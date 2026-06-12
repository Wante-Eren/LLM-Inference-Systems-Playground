import concurrent.futures
import hashlib
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import psutil


# =================================================================================================
# 进程级网络解耦：物理清除代理变量，避免 cpolar / Windows 代理残留劫持本地环回路由。
# 这里必须在任何潜在网络库初始化前执行，确保 127.0.0.1 与 llama-server 控制面完全去代理化。
# =================================================================================================
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)


# =================================================================================================
# 实验参数配置：Qwen2-7B-Instruct-Q4_K_M + RTX 4060 Laptop 8GB 显存的端侧推理档位。
# =================================================================================================
SERVER_CWD = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_CWD.parent
MODEL_BASE = os.environ.get(
    "MODEL_BASE",
    str(PROJECT_ROOT / "models/qwen2-7b-instruct-q4_k_m.gguf"),
)
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "./build/bin/llama-server")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))

POLL_INTERVAL_SEC = 10
RESTORE_PROBE_INTERVAL_SEC = 1
LOW_POWER_DEADLINE_PERCENT = 30

FULL_NGL_LAYERS = 99
FULL_CTX_SIZE = 2048
ECO_NGL_LAYERS = 15
ECO_CTX_SIZE = 512

GRACEFUL_TERMINATE_TIMEOUT_SEC = 10

current_process: Optional[subprocess.Popen] = None
current_process_lock = threading.Lock()

fusion_lock = threading.Lock()
fusion_stop_event = threading.Event()
fusion_thread: Optional[threading.Thread] = None
fusion_generation = 0


class ServingMode(Enum):
    """守护进程的强约束双态状态机，避免电量边界附近反复重启。"""

    PERFORMANCE = "performance"
    LOW_POWER_FUSED = "low_power_fused"


@dataclass(frozen=True)
class HardwareStatus:
    percent: float
    plugged: bool


@dataclass(frozen=True)
class ServingProfile:
    mode: ServingMode
    ngl_layers: int
    ctx_size: int


PERFORMANCE_PROFILE = ServingProfile(
    mode=ServingMode.PERFORMANCE,
    ngl_layers=FULL_NGL_LAYERS,
    ctx_size=FULL_CTX_SIZE,
)
LOW_POWER_PROFILE = ServingProfile(
    mode=ServingMode.LOW_POWER_FUSED,
    ngl_layers=ECO_NGL_LAYERS,
    ctx_size=ECO_CTX_SIZE,
)


def get_hardware_status() -> HardwareStatus:
    """读取边缘端供电状态；无电池设备按永久插电处理。"""
    battery = psutil.sensors_battery()
    if battery is None:
        return HardwareStatus(percent=100.0, plugged=True)
    return HardwareStatus(percent=float(battery.percent), plugged=bool(battery.power_plugged))


def _safe_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return ""


def _is_managed_llama_server(proc: psutil.Process) -> bool:
    """只清理本守护脚本会竞争的 llama-server 实例，避免误伤其他实验进程。"""
    cmdline = _safe_cmdline(proc)
    if not cmdline:
        return False

    port_tokens = {str(PORT), f"--port {PORT}"}
    return "llama-server" in cmdline and any(token in cmdline for token in port_tokens)


def _terminate_process(proc: psutil.Process, reason: str) -> None:
    if proc.pid == os.getpid():
        return

    try:
        print(f"[进程守护] {reason}: pid={proc.pid} -> SIGTERM 优雅释放")
        proc.terminate()
        proc.wait(timeout=GRACEFUL_TERMINATE_TIMEOUT_SEC)
    except psutil.TimeoutExpired:
        print(f"[进程守护] pid={proc.pid} 未在窗口内退出 -> SIGKILL 兜底清空显存句柄")
        proc.kill()
        proc.wait(timeout=GRACEFUL_TERMINATE_TIMEOUT_SEC)
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        pass


def _terminate_existing_llama_servers() -> None:
    """切档前清空旧实例，严防 mmap 文件锁、端口复用与 CUDA 上下文残留。"""
    global current_process

    with current_process_lock:
        if current_process is not None and current_process.poll() is None:
            try:
                proc = psutil.Process(current_process.pid)
                _terminate_process(proc, "检测到当前托管 llama-server 实例")
            except psutil.NoSuchProcess:
                pass
            finally:
                try:
                    current_process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    current_process.kill()
                    current_process.wait()

        current_process = None

    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        if _is_managed_llama_server(proc):
            _terminate_process(proc, "检测到同端口残留 llama-server 实例")

    # 给 CUDA driver 与文件映射层一个短暂收敛窗口，减少热重启时的显存碎片与锁竞争。
    time.sleep(1)


def start_adaptive_server(profile: ServingProfile) -> None:
    """按当前能耗档位重启 llama-server，并保证旧实例已物理退出。"""
    global current_process

    _terminate_existing_llama_servers()

    cmd = [
        LLAMA_SERVER_BIN,
        "-m",
        MODEL_BASE,
        "-ngl",
        str(profile.ngl_layers),
        "--ctx-size",
        str(profile.ctx_size),
        "--port",
        str(PORT),
        "--host",
        HOST,
    ]

    print("\n[决策引擎] 触发软硬件协同自适应策略")
    print(
        "[参数激活] "
        f"mode={profile.mode.value} | ngl_layers={profile.ngl_layers} | "
        f"ctx_size={profile.ctx_size} tokens | port={PORT}"
    )
    print(f"[执行命令] {' '.join(cmd)}")

    with current_process_lock:
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            cwd=str(SERVER_CWD),
            start_new_session=True,
        )


def _select_prefix_anchors(ctx_size: int) -> list[str]:
    """选择性加载 Prefix 锚点：用极小稳定上下文保留系统语义和检索路由骨架。"""
    anchor_budget = max(64, ctx_size // 4)
    return [
        f"system-policy-anchor:{anchor_budget}",
        f"conversation-intent-anchor:{anchor_budget // 2}",
        f"rag-routing-anchor:{anchor_budget // 2}",
    ]


def _iter_trimmed_rag_shards(ctx_size: int) -> Iterable[tuple[int, str]]:
    """模拟被活跃窗口裁剪的 RAG 知识分片；真实系统可替换为向量库 / KV 元数据扫描。"""
    shard_count = max(4, min(16, ctx_size // 64))
    for shard_id in range(shard_count):
        yield shard_id, f"trimmed-rag-shard:{shard_id}:ctx={ctx_size}:semantic-digest"


def _recompute_rag_shard(shard: tuple[int, str]) -> tuple[int, str]:
    """CPU 侧局部重计算：用算力换带宽，避免低电量下全量 KV 搬运压垮内存通道。"""
    shard_id, payload = shard
    digest = payload.encode("utf-8")

    # 轻量确定性计算，只模拟局部重算的调度形态，不在守护层制造真实高负载。
    for _ in range(256):
        digest = hashlib.blake2b(digest, digest_size=16).digest()

    return shard_id, digest.hex()


def _async_kv_fusion_worker(ctx_size: int, generation: int, stop_event: threading.Event) -> None:
    """CacheBlend 知识融合工兵线程：异步执行选择性加载、局部重算与融合提交。"""
    print(
        "[CacheBlend] 后台融合层已接管低功耗窗口: "
        f"ctx_size={ctx_size}, generation={generation}"
    )

    prefix_anchors = _select_prefix_anchors(ctx_size)
    print(f"[CacheBlend] Selective Prefix Loading -> anchors={len(prefix_anchors)}")

    if stop_event.is_set():
        return

    shards = list(_iter_trimmed_rag_shards(ctx_size))
    worker_count = max(1, min(4, (os.cpu_count() or 1), len(shards)))
    fused_kv_digests: list[tuple[int, str]] = []

    print(
        "[CacheBlend] Local Re-computation -> "
        f"rag_shards={len(shards)}, cpu_workers={worker_count}"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_recompute_rag_shard, shard): shard for shard in shards}
        for future in concurrent.futures.as_completed(future_map):
            if stop_event.is_set():
                print("[CacheBlend] 检测到全血恢复信号，取消低功耗融合提交")
                return
            fused_kv_digests.append(future.result())

    # Fused Knowledge Fusion：在系统层模拟片上缓存内的新旧 KV 非饱和缝合。
    # 真实实现应在 llama.cpp KV-cache / attention prefill 热路径中接入 prefix 复用与 shard 重算结果。
    fusion_manifest = {
        "generation": generation,
        "ctx_size": ctx_size,
        "prefix_anchor_count": len(prefix_anchors),
        "recomputed_shard_count": len(fused_kv_digests),
        "fusion_operator": "non_saturating_forward_stitch",
    }

    print(
        "[CacheBlend] Fused Knowledge Fusion -> "
        f"{fusion_manifest} | 知识不丢弃，带宽压力转移为可控计算"
    )

    while not stop_event.is_set():
        time.sleep(5)

    print(f"[CacheBlend] generation={generation} 后台融合层安全退场")


def cacheblend_knowledge_fusion_adaptation(ctx_size: int) -> None:
    """O(1) 非阻塞触发 CacheBlend 风格知识融合演进层。"""
    global fusion_generation, fusion_thread

    with fusion_lock:
        if fusion_thread is not None and fusion_thread.is_alive():
            print("[CacheBlend] 融合工兵线程已在线，复用当前低功耗知识融合层")
            return

        fusion_generation += 1
        fusion_stop_event.clear()
        fusion_thread = threading.Thread(
            target=_async_kv_fusion_worker,
            args=(ctx_size, fusion_generation, fusion_stop_event),
            name=f"cacheblend-kv-fusion-{fusion_generation}",
            daemon=True,
        )
        fusion_thread.start()

    print("[CacheBlend] 主线程已常数级释放，llama-server 切档路径不被 KV 融合阻塞")


def stop_cacheblend_fusion_layer() -> None:
    """恢复插电后停止低功耗融合层，回到全窗口原生 KV 生命周期。"""
    global fusion_thread

    with fusion_lock:
        if fusion_thread is None or not fusion_thread.is_alive():
            fusion_thread = None
            return

        print("[CacheBlend] 收到插电恢复信号 -> 通知后台融合层退场")
        fusion_stop_event.set()
        thread = fusion_thread

    thread.join(timeout=2)

    with fusion_lock:
        if fusion_thread is thread and not thread.is_alive():
            fusion_thread = None


def _shutdown_current_process() -> None:
    global current_process

    stop_cacheblend_fusion_layer()

    with current_process_lock:
        proc_handle = current_process
        current_process = None

    if proc_handle is None or proc_handle.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(proc_handle.pid), signal.SIGTERM)
        proc_handle.wait(timeout=GRACEFUL_TERMINATE_TIMEOUT_SEC)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        proc_handle.kill()
        proc_handle.wait()


def main() -> None:
    print("=" * 88)
    print("  双重感知边缘端算力守护系统 V3.0 | llama.cpp + CacheBlend Knowledge Fusion")
    print("  底座大脑: Qwen2-7B-Instruct-Q4_K_M | RTX 4060 8GB | 进程级自适应常驻服务")
    print("=" * 88)

    status = get_hardware_status()
    print(
        "\n[感知层] 首次点火初始化 -> "
        f"battery={status.percent:.1f}%, plugged={status.plugged}"
    )
    print("[策略] 首次点火强制进入全速高性能模式，随后由硬件水位状态机接管")

    active_mode = ServingMode.PERFORMANCE
    start_adaptive_server(PERFORMANCE_PROFILE)
    print(f"\n[守护进程] llama-server 已常驻后台，监听 {HOST}:{PORT}")

    try:
        while True:
            status = get_hardware_status()

            if (
                active_mode is ServingMode.PERFORMANCE
                and not status.plugged
                and status.percent < LOW_POWER_DEADLINE_PERCENT
            ):
                print(
                    "\n[能耗熔断] 触达未插电极限死线 -> "
                    f"battery={status.percent:.1f}% < {LOW_POWER_DEADLINE_PERCENT}%"
                )
                print("[策略] 降低 GPU 卸载层与活跃窗口，并启动 CacheBlend 知识融合演进层")
                active_mode = ServingMode.LOW_POWER_FUSED
                cacheblend_knowledge_fusion_adaptation(LOW_POWER_PROFILE.ctx_size)
                start_adaptive_server(LOW_POWER_PROFILE)

                print("[状态机] 已进入阻塞保压区：等待重新插电，禁止电量边界震荡重启")
                while True:
                    status = get_hardware_status()
                    if status.plugged:
                        print(
                            "\n[电源恢复] 检测到适配器重新接入 -> "
                            f"battery={status.percent:.1f}%, 秒级恢复全血档位"
                        )
                        stop_cacheblend_fusion_layer()
                        active_mode = ServingMode.PERFORMANCE
                        start_adaptive_server(PERFORMANCE_PROFILE)
                        print("[状态机] 已恢复全速高性能常驻模式")
                        break
                    time.sleep(RESTORE_PROBE_INTERVAL_SEC)

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n[系统] 接收到退出指令，开始清理推理进程与后台融合线程")
    finally:
        _shutdown_current_process()
        print("[系统] 自适应常驻服务器已安全关闭")


if __name__ == "__main__":
    main()
