# dynamic_serving.py 说明文档

本文档说明 `/home/wante/DPL/Embodied-Oms/llama.cpp/dynamic_serving.py` 的功能、运行逻辑和本次重构引入的关键机制。

## 一句话概览

`dynamic_serving.py` 是一个面向笔记本边缘端推理的 `llama.cpp` 常驻守护进程脚本。它负责根据电源状态自动切换 Qwen2-7B-Instruct-Q4_K_M 的推理档位：插电时使用高性能 GPU 卸载和较大上下文窗口，低电量未插电时进入功耗熔断档，同时启动一个 CacheBlend 风格的异步知识融合模拟层，避免简单截断上下文带来的 RAG 知识断流。

## 服务对象

脚本默认管理以下本地推理服务：

```bash
./build/bin/llama-server \
  -m /home/wante/DPL/Embodied-Oms/models/qwen2-7b-instruct-q4_k_m.gguf \
  --port 8080 \
  --host 127.0.0.1
```

默认模型为：

```text
/home/wante/DPL/Embodied-Oms/models/qwen2-7b-instruct-q4_k_m.gguf
```

默认端口为：

```text
127.0.0.1:8080
```

脚本会把子进程工作目录固定到 `dynamic_serving.py` 所在的 `llama.cpp` 目录，因此从仓库根目录或 `llama.cpp` 目录启动都能正确找到 `./build/bin/llama-server`。

## 网络与代理隔离

脚本启动后第一时间清除以下环境变量：

```python
http_proxy
https_proxy
HTTP_PROXY
HTTPS_PROXY
```

这样做是为了避免 Windows 代理残留、cpolar 内网穿透场景或 shell 环境变量污染影响本地推理服务访问。当前版本把 `llama-server` 绑定到 `127.0.0.1`，控制面只走本地环回地址，默认不暴露到局域网。

## 两种推理档位

脚本定义了两个 `ServingProfile`。

高性能档：

```text
mode       = performance
ngl_layers = 99
ctx_size   = 2048
```

这个档位用于插电状态和首次启动状态。目标是尽可能让 RTX 4060 承担更多层计算，并保留较完整的活跃上下文窗口。

低功耗融合档：

```text
mode       = low_power_fused
ngl_layers = 15
ctx_size   = 512
```

这个档位只在极限低电量未插电状态触发。目标是减少 GPU 卸载压力和活跃 KV cache 窗口，降低功耗和显存压力。

## 硬件状态机

脚本使用 `psutil.sensors_battery()` 读取电池状态，并每 10 秒轮询一次。

首次启动时，无论当前电量如何，脚本都会先启动高性能档：

```text
performance -> ngl_layers=99, ctx_size=2048
```

之后进入硬件状态机。只有满足下面两个条件时，才会触发降级：

```text
not plugged
percent < 30
```

也就是说，只有未插电并且电量低于 30% 时，才进入低功耗融合档。这个条件避免了旧版本中“只要拔电就降级”或“电量边界附近频繁重启”的控制流震荡。

降级后，脚本进入阻塞保压区：

```text
low_power_fused -> 等待重新插电
```

在这个阶段不会继续根据电量变化反复重启服务，而是每 1 秒检测一次是否重新插电。检测到插电后，立即停止低功耗融合层，并恢复高性能档。

## 进程生命周期管理

`start_adaptive_server(profile)` 是核心启动函数。每次切换档位前，它都会先调用 `_terminate_existing_llama_servers()` 清理旧实例。

清理逻辑包括：

- 如果当前脚本已经托管了一个 `llama-server` 子进程，则先发送 `SIGTERM`。
- 等待进程优雅退出，默认最多等待 10 秒。
- 如果超时未退出，则使用 kill 兜底。
- 扫描系统进程，清理同端口 `8080` 的残留 `llama-server` 实例。
- 清理后等待 1 秒，让 CUDA driver、mmap 文件映射和端口状态有时间收敛。

这样做是为了减少以下问题：

- 端口占用。
- 模型文件 mmap 锁竞争。
- CUDA 上下文残留。
- 热重启导致的显存碎片或显存死锁。

脚本退出时，`finally` 块会调用 `_shutdown_current_process()`，停止后台融合线程并关闭当前 `llama-server`。

## CacheBlend 知识融合演进层

本次重构新增了：

```python
cacheblend_knowledge_fusion_adaptation(ctx_size)
```

这个函数用于在低功耗降级时启动一个后台线程：

```python
_async_kv_fusion_worker(ctx_size, generation, stop_event)
```

它的设计目标是模拟 CacheBlend 的 Cached Knowledge Fusion 思路：当活跃上下文窗口从 2048 降到 512 时，不把被裁剪的 RAG 历史知识简单丢掉，而是在守护进程侧模拟一种“选择性保留核心 Prefix + 后台局部重计算 + 融合提交”的策略。

需要注意：当前 Python 层无法直接改写 `llama.cpp` 内部真实 KV cache。这里实现的是系统控制层模拟和调度骨架，方便后续接入真实的 `llama.cpp` KV-cache hook、RAG 元数据索引或 prefill 热路径。

## 融合层内部流程

低功耗触发后，主线程调用：

```python
cacheblend_knowledge_fusion_adaptation(512)
```

这个函数只负责创建后台线程并立刻返回，所以不会阻塞 `llama-server` 切档路径。

后台线程会执行三个阶段。

第一阶段：Selective Prefix Loading。

```python
_select_prefix_anchors(ctx_size)
```

它模拟选择少量稳定 Prefix 锚点，例如系统策略锚点、会话意图锚点、RAG 路由锚点。这样做的含义是：低功耗时不搬运全量历史 KV，而是优先保留最能稳定语义和检索路由的核心前缀。

第二阶段：Local Re-computation。

```python
_iter_trimmed_rag_shards(ctx_size)
_recompute_rag_shard(shard)
```

脚本会模拟生成被活跃窗口裁剪掉的 RAG 知识分片，并使用 `ThreadPoolExecutor` 在 CPU 线程池中并行执行轻量确定性重计算。这里用 `hashlib.blake2b` 模拟重计算工作负载。

它表达的系统思想是：低电量时内存带宽和显存搬运更脆弱，因此不要做高延迟、全量 KV 搬运，而是用边缘端相对可控的 CPU compute 去替换 bandwidth 压力。

第三阶段：Fused Knowledge Fusion。

后台线程最终生成一个 `fusion_manifest`，模拟将 Prefix 锚点和重计算后的 RAG shard 融合为新的知识缓存状态。

当前 manifest 包含：

```text
generation
ctx_size
prefix_anchor_count
recomputed_shard_count
fusion_operator = non_saturating_forward_stitch
```

这一步对应系统层语义上的“新旧 KV 非饱和前向缝合”。真实生产实现可以把这里替换成 llama.cpp 内部 KV-cache、attention prefill 或 RAG cache 索引的具体融合逻辑。

## 为什么不是简单截断上下文

旧版本在拔电后直接把 `ctx_size` 降到 512。这样虽然降低了功耗，但会导致：

- Agent 会话历史断流。
- RAG 检索到的长文本知识被硬裁剪。
- 低功耗状态下首字延迟和语义稳定性不可控。

新版本仍然会在低功耗档把活跃窗口降到 512，但同时用后台融合层模拟保留核心 Prefix 和重计算外部知识分片。它表达的是一种 MLSys 层面的折中：

```text
少搬 KV，少吃带宽；
保留锚点，维持语义；
局部重算，用 compute 替换 bandwidth；
后台融合，不阻塞主服务切档。
```

## 运行方式

推荐在仓库根目录运行：

```bash
python llama.cpp/dynamic_serving.py
```

或者进入 `llama.cpp` 目录运行：

```bash
cd /home/wante/DPL/Embodied-Oms/llama.cpp
python dynamic_serving.py
```

启动后脚本会自动拉起 `llama-server`，监听：

```text
http://127.0.0.1:8080
```

按 `Ctrl+C` 退出时，脚本会清理当前 `llama-server` 子进程和后台融合线程。

## 当前实现边界

这份脚本做的是守护进程级控制，不是对 `llama.cpp` C++ KV cache 的真实内核改造。

当前已实现：

- 电源感知。
- 稳定双态状态机。
- `llama-server` 进程生命周期管理。
- 代理环境变量清理。
- 本地环回绑定。
- CacheBlend 风格异步知识融合模拟层。
- CPU 并行局部重计算的调度骨架。
- 融合 manifest 的日志可观测性。

当前未实现：

- 真实读取和写入 llama.cpp 内部 KV cache。
- 真实 RAG 文档 shard 元数据扫描。
- 真实 Prefix cache 的二进制加载。
- 真实 fused attention / prefill kernel。
- 与上层 Agent 记忆系统或向量数据库的 API 对接。

后续如果要把模拟层推进到真实系统层，可以从以下方向接入：

- 在 RAG 层维护可裁剪 shard 的元数据 manifest。
- 在 llama.cpp server 请求链路中暴露 prefix cache 标识。
- 在 prefill 阶段支持选择性 prefix KV 复用。
- 在低功耗档把后台重计算结果写回一个可复用的本地 cache index。
- 用实际 TTFT、tokens/s、电池放电速率和显存占用数据闭环调参。

