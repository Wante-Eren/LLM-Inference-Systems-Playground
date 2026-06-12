#!/usr/bin/env python3
import argparse
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


# 进程级网络解耦：演示脚本同样继承 dynamic_serving.py 的代理隔离策略。
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_BASE = os.environ.get(
    "MODEL_BASE",
    str(PROJECT_ROOT / "models/qwen2-7b-instruct-q4_k_m.gguf"),
)
LLAMA_SERVER_BIN = Path(
    os.environ.get("LLAMA_SERVER_BIN", str(SCRIPT_DIR / "build/bin/llama-server"))
)
CACHEBLEND_FUSION_BIN = Path(
    os.environ.get(
        "CACHEBLEND_FUSION_BIN",
        str(SCRIPT_DIR / "build/bin/llama-cacheblend-static-fusion"),
    )
)

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))

FULL_NGL_LAYERS = 99
FULL_CTX_SIZE = 2048
ECO_NGL_LAYERS = 15
ECO_CTX_SIZE = 512

LOW_POWER_PERCENT = 25.0
FULL_POWER_PERCENT = 100.0

FUSION_DEFAULT_TOKENS = 8
FUSION_DEFAULT_NGL = 0
FUSION_TIMEOUT_SEC = 240

# Qwen2-7B: n_layer=28, n_embd_k_gqa=512, n_embd_v_gqa=512, KV cache 默认 F16。
QWEN2_N_LAYER = 28
QWEN2_N_EMBD_K_GQA = 512
QWEN2_N_EMBD_V_GQA = 512
KV_BYTES_PER_ELEMENT = 2


class ServingMode(Enum):
    PERFORMANCE = "PERFORMANCE"
    LOW_POWER_FUSED = "LOW_POWER_FUSED"


@dataclass(frozen=True)
class MockPowerStatus:
    percent: float
    plugged: bool


@dataclass(frozen=True)
class ServingProfile:
    mode: ServingMode
    ngl_layers: int
    ctx_size: int


@dataclass
class DemoState:
    power: MockPowerStatus
    active_profile: ServingProfile
    host_shadow_tokens: int = 0
    host_shadow_ready: bool = False
    low_power_epoch: int = 0
    fusion_count: int = 0
    last_fusion_ok: Optional[bool] = None
    last_fusion_summary: str = "尚未触发"


PERFORMANCE_PROFILE = ServingProfile(ServingMode.PERFORMANCE, FULL_NGL_LAYERS, FULL_CTX_SIZE)
LOW_POWER_PROFILE = ServingProfile(ServingMode.LOW_POWER_FUSED, ECO_NGL_LAYERS, ECO_CTX_SIZE)


class TerminalKeyReader:
    """非阻塞单键读取器：按 d/c/q 即时注入状态，不需要回车。"""

    def __init__(self) -> None:
        self.is_tty = sys.stdin.isatty()
        self.old_settings = None

    def __enter__(self) -> "TerminalKeyReader":
        if self.is_tty:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.is_tty and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout: float) -> Optional[str]:
        if not self.is_tty:
            time.sleep(timeout)
            return None

        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        return sys.stdin.read(1).lower()


def kv_cache_mib(ctx_size: int) -> float:
    bytes_total = (
        ctx_size
        * QWEN2_N_LAYER
        * (QWEN2_N_EMBD_K_GQA + QWEN2_N_EMBD_V_GQA)
        * KV_BYTES_PER_ELEMENT
    )
    return bytes_total / 1024.0 / 1024.0


def status_text(power: MockPowerStatus) -> str:
    plugged = "已插电" if power.plugged else "未插电"
    return f"{power.percent:.1f}% / {plugged}"


def print_banner(args: argparse.Namespace) -> None:
    print("=" * 96)
    print("  CacheBlend Static Fusion Demo | Edge LLM Serving Visualization")
    print("  手动按键注入: d = 断电低功耗, c = 充电恢复融合, q = 退出")
    print("=" * 96)
    print(f"[模型] {args.model}")
    print(f"[C++ 融合器] {args.fusion_bin}")
    print(f"[模式] {'联动 llama-server 切档' if args.launch_server else '安全可视化模式（不启动 llama-server）'}")
    print()


def print_dashboard(state: DemoState) -> None:
    profile = state.active_profile
    print(
        "[状态看盘] "
        f"mode={profile.mode.value:<15} | "
        f"power={status_text(state.power):<14} | "
        f"ctx={profile.ctx_size:<4} | "
        f"ngl={profile.ngl_layers:<2} | "
        f"active_KV={kv_cache_mib(profile.ctx_size):6.2f} MiB | "
        f"host_shadow={state.host_shadow_tokens:4d} tokens | "
        f"fusion={state.last_fusion_summary}"
    )


def low_power_short_prompt(state: DemoState) -> str:
    return (
        "[seq_0 LOW_POWER_SHORT_CONTEXT] "
        f"epoch={state.low_power_epoch}; active_ctx={ECO_CTX_SIZE}; "
        "the edge agent kept only the freshest short dialogue span under battery deadline."
    )


def host_long_prompt(state: DemoState) -> str:
    return (
        "[seq_1 HOST_TRUNCATED_LONG_CONTEXT] "
        f"epoch={state.low_power_epoch}; shadow_tokens={state.host_shadow_tokens}; "
        "historical RAG shards were moved to host-side memory and await static KV fusion."
    )


def run_cacheblend_static_fusion(args: argparse.Namespace, state: DemoState) -> bool:
    fusion_bin = Path(args.fusion_bin)
    if not fusion_bin.exists():
        print(f"[错误] 找不到 C++ 融合器: {fusion_bin}")
        state.last_fusion_summary = "C++ binary missing"
        return False

    cmd = [
        str(fusion_bin),
        "-m",
        args.model,
        "--n-tokens",
        str(args.fusion_tokens),
        "-ngl",
        str(args.fusion_ngl),
        "--n-verify",
        str(args.n_verify),
        "--prompt-a",
        low_power_short_prompt(state),
        "--prompt-b",
        host_long_prompt(state),
    ]

    print()
    print(
        "[高光] 检测到电源恢复！正在调用底层 C++ 扫描 56 组 K/V 载荷偏移，"
        "执行 0.5*A + 0.5*B 物理缝合..."
    )
    print(f"[跨语言调用] {' '.join(cmd)}")

    started = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.fusion_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"[错误] C++ 融合器超时，超过 {args.fusion_timeout}s 未返回")
        state.last_fusion_summary = "timeout"
        return False

    elapsed = time.time() - started
    combined = result.stdout + "\n" + result.stderr
    verify_lines = [line for line in combined.splitlines() if line.startswith("[verify]")]
    fused_lines = [line for line in combined.splitlines() if "fused 56 K/V payload ranges" in line]
    injected_lines = [line for line in combined.splitlines() if "injected fused state into seq_id=2" in line]

    if args.show_cpp_log:
        print("-" * 96)
        print(combined.strip())
        print("-" * 96)
    else:
        for line in fused_lines[-1:]:
            print(f"[C++ 摘要] {line}")
        for line in injected_lines[-1:]:
            print(f"[C++ 摘要] {line}")
        for line in verify_lines[-2:]:
            print(f"[C++ 验证] {line}")

    ok = result.returncode == 0 and bool(fused_lines) and bool(injected_lines) and bool(verify_lines)
    state.fusion_count += 1
    state.last_fusion_ok = ok
    state.last_fusion_summary = f"{'OK' if ok else 'FAILED'} #{state.fusion_count} ({elapsed:.1f}s)"

    if ok:
        print("[高光] 融合状态已注入 seq_2，全血续写成功！")
    else:
        print(f"[错误] C++ 融合器返回异常: returncode={result.returncode}")
        if not args.show_cpp_log:
            tail = "\n".join(combined.splitlines()[-20:])
            print("[C++ 日志尾部]")
            print(tail)

    return ok


def enter_low_power(state: DemoState) -> None:
    if state.active_profile.mode is ServingMode.LOW_POWER_FUSED:
        print("[状态机] 已处于 LOW_POWER_FUSED，忽略重复 d 注入")
        return

    state.low_power_epoch += 1
    state.power = MockPowerStatus(LOW_POWER_PERCENT, False)
    state.active_profile = LOW_POWER_PROFILE
    state.host_shadow_tokens = FULL_CTX_SIZE - ECO_CTX_SIZE
    state.host_shadow_ready = True

    print()
    print("[注入] d -> 模拟断电: 25% / 未插电")
    print(
        f"[警告] 活跃上下文由 {FULL_CTX_SIZE} 压缩至 {ECO_CTX_SIZE}，"
        f"已有 {state.host_shadow_tokens} 个历史 Token 被物理截断至 Host 侧！"
    )
    print("[CacheBlend] seq_0=低功耗短上下文，seq_1=Host 侧长上下文阴影，等待恢复时融合")


def enter_performance(args: argparse.Namespace, state: DemoState) -> None:
    if state.active_profile.mode is ServingMode.PERFORMANCE:
        state.power = MockPowerStatus(FULL_POWER_PERCENT, True)
        print("[状态机] 已处于 PERFORMANCE，刷新为 100% / 已插电")
        return

    print()
    print("[注入] c -> 模拟充电恢复: 100% / 已插电")
    state.power = MockPowerStatus(FULL_POWER_PERCENT, True)

    if state.host_shadow_ready:
        run_cacheblend_static_fusion(args, state)
    else:
        print("[提示] 没有 Host 侧长上下文阴影，跳过 C++ 融合")

    state.active_profile = PERFORMANCE_PROFILE
    state.host_shadow_tokens = 0
    state.host_shadow_ready = False
    print(f"[状态机] 恢复全血档: ctx={FULL_CTX_SIZE}, ngl={FULL_NGL_LAYERS}, active_KV={kv_cache_mib(FULL_CTX_SIZE):.2f} MiB")


class OptionalServerManager:
    def __init__(self, enabled: bool, model: str) -> None:
        self.enabled = enabled
        self.model = model
        self.process: Optional[subprocess.Popen] = None

    def start(self, profile: ServingProfile) -> None:
        if not self.enabled:
            print(f"[服务层] 可视化模式：不启动 llama-server，仅模拟 profile={profile.mode.value}")
            return

        self.stop()
        cmd = [
            str(LLAMA_SERVER_BIN),
            "-m",
            self.model,
            "-ngl",
            str(profile.ngl_layers),
            "--ctx-size",
            str(profile.ctx_size),
            "--port",
            str(PORT),
            "--host",
            HOST,
        ]
        print(f"[服务层] 启动 llama-server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = None
            return
        print("[服务层] 停止当前 llama-server 子进程")
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=8)
        except Exception:
            self.process.kill()
            self.process.wait()
        finally:
            self.process = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CacheBlend serving state-machine visualization demo")
    parser.add_argument("--model", default=MODEL_BASE, help="GGUF model path")
    parser.add_argument("--fusion-bin", default=str(CACHEBLEND_FUSION_BIN), help="llama-cacheblend-static-fusion binary path")
    parser.add_argument("--fusion-tokens", type=int, default=FUSION_DEFAULT_TOKENS, help="equal token count used by the C++ Level 1 PoC")
    parser.add_argument("--fusion-ngl", type=int, default=FUSION_DEFAULT_NGL, help="GPU offload layers for the C++ PoC; default 0 for safe demos")
    parser.add_argument("--n-verify", type=int, default=1, help="number of verification tokens decoded by the C++ PoC")
    parser.add_argument("--fusion-timeout", type=int, default=FUSION_TIMEOUT_SEC, help="timeout for the C++ PoC subprocess")
    parser.add_argument("--show-cpp-log", action="store_true", help="print the full C++ subprocess log")
    parser.add_argument("--launch-server", action="store_true", help="also start/restart llama-server on mode changes")
    parser.add_argument("--dashboard-interval", type=float, default=2.0, help="seconds between dashboard refreshes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = DemoState(
        power=MockPowerStatus(FULL_POWER_PERCENT, True),
        active_profile=PERFORMANCE_PROFILE,
    )
    server = OptionalServerManager(args.launch_server, args.model)

    print_banner(args)
    server.start(state.active_profile)
    print_dashboard(state)

    print("\n[键盘] 按 d 触发低功耗，按 c 恢复并调用 C++ 融合，按 q 退出。")
    next_dashboard_at = time.time() + args.dashboard_interval

    try:
        with TerminalKeyReader() as keys:
            while True:
                key = keys.read_key(timeout=0.1)

                if key == "d":
                    enter_low_power(state)
                    server.start(state.active_profile)
                    print_dashboard(state)
                    next_dashboard_at = time.time() + args.dashboard_interval
                elif key == "c":
                    enter_performance(args, state)
                    server.start(state.active_profile)
                    print_dashboard(state)
                    next_dashboard_at = time.time() + args.dashboard_interval
                elif key == "q":
                    print("\n[系统] 收到 q，退出演示")
                    break
                elif key in {"\x03", "\x04"}:
                    print("\n[系统] 收到终端中断，退出演示")
                    break

                if time.time() >= next_dashboard_at:
                    print_dashboard(state)
                    next_dashboard_at = time.time() + args.dashboard_interval
    finally:
        server.stop()

    print("[系统] demo_serving_visualization 已安全关闭")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
