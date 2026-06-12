# demo_serving_visualization.py 说明文档

本文档记录今天下午围绕 CacheBlend 静态 KV 融合与边缘端服务状态机展示所做的完整改进。对应主脚本为：

```text
/home/wante/DPL/Embodied-Oms/llama.cpp/demo_serving_visualization.py
```

相关 C++ PoC 位于：

```text
/home/wante/DPL/Embodied-Oms/llama.cpp/examples/cacheblend-static-fusion/
```

可执行文件为：

```text
/home/wante/DPL/Embodied-Oms/llama.cpp/build/bin/llama-cacheblend-static-fusion
```

## 背景目标

原始 `dynamic_serving.py` 已经能够根据电池状态在两个服务档位之间切换：

```text
PERFORMANCE      : 2048 ctx, 99 GPU offload layers
LOW_POWER_FUSED  : 512 ctx, 15 GPU offload layers
```

但开发设备 RTX 4060 Laptop 日常永久插电，无法方便地靠真实拔电触发状态机。因此今天新增了一个面向导师展示的可视化脚本：

```text
demo_serving_visualization.py
```

它不依赖真实电池状态，而是通过终端按键手动注入电源事件：

```text
d -> 模拟断电，切入低功耗 512 ctx
c -> 模拟充电恢复，触发 C++ 静态 KV 融合 PoC
q -> 退出演示
```

这样可以稳定复现“断电截断 -> Host 侧保留历史 -> 充电恢复 -> C++ 融合 -> seq_2 续写”的闭环。

## 下午完成的核心改进

今天下午主要完成了两条链路。

第一条是 C++ 底层 PoC：

```text
examples/cacheblend-static-fusion/cacheblend-static-fusion.cpp
```

它真实调用 llama.cpp C API，把两段 prompt 分别 decode 到 `seq_id=0` 和 `seq_id=1`，导出 host-side sequence state buffer，扫描每层 K/V payload offset，对 F16/F32 KV payload 执行：

```text
fused = 0.5 * state_a + 0.5 * state_b
```

然后把融合后的完整 state buffer 注入 `seq_id=2`，并继续 decode 1 到 2 个 token，验证模型能正常前向传播。

第二条是 Python 展示层：

```text
demo_serving_visualization.py
```

它在状态机恢复全血时通过 `subprocess.run()` 跨语言调用 C++ 可执行文件，把低功耗短上下文和 Host 侧长上下文阴影作为 Prompt A/B 传给 C++ PoC，完成现场演示闭环。

## C++ Level 1 PoC 做了什么

我们实现的是 Level 1: State-Buffer 离线融合 PoC。它故意先限制在等长 token 条件下，避免在第一版中处理复杂的不同长度切片、RoPE 位置对齐和 KV offset 重排。

流程如下：

1. 加载 GGUF 模型。
2. 创建 `llama_context`，设置 `n_seq_max = 3`。
3. 构造等长 Prompt A 和 Prompt B。
4. Prompt A decode 到 `seq_id=0`。
5. Prompt B decode 到 `seq_id=1`。
6. 调用 `llama_state_seq_get_size()` 获取两个序列的 state size。
7. 调用 `llama_state_seq_get_data()` 导出两个 host buffer。
8. 解析 state buffer 的二进制布局。
9. 找到每层 K/V payload 的 `payload_offset` 和 `payload_bytes`。
10. 创建 `fused_buffer`，先全量 `memcpy(state_a)` 作为安全拓扑基底。
11. 仅在 K/V payload 范围内执行 F16/F32 数值融合。
12. 调用 `llama_state_seq_set_data(..., seq_id=2)` 注入融合 state。
13. 从 `seq_id=2` 继续 decode，打印 token/logit 验证结果。

这条链路已经实测跑通，退出码为 0。

实测关键输出如下：

```text
seq_id=0 state size = 459544 bytes
seq_id=1 state size = 459544 bytes
fused 56 K/V payload ranges, total payload bytes = 458752
injected fused state into seq_id=2, bytes=459544
[verify] step=0 seq_id=2 input_token=17105 pos=8 argmax_token=25 argmax_logit=6.669005 sampled_token=25 piece=':'
CacheBlend static fusion PoC completed without crash.
```

其中 56 组 K/V payload 来自：

```text
28 层 K payload + 28 层 V payload = 56
```

当前 Qwen2-7B 的 KV cache 类型为 F16，因此 C++ PoC 会把 `ggml_fp16_t` 转成 float，执行线性融合后再写回 F16。

## demo_serving_visualization.py 做了什么

该脚本是一个展示用状态机外壳，重点是把服务侧策略、按键注入和 C++ KV 融合串起来。

启动后默认状态为：

```text
power = 100% / 已插电
mode  = PERFORMANCE
ctx   = 2048
ngl   = 99
```

控制台会持续打印状态看盘：

```text
[状态看盘] mode=PERFORMANCE | power=100.0% / 已插电 | ctx=2048 | ngl=99 | active_KV=112.00 MiB | host_shadow=0 tokens | fusion=尚未触发
```

这里的 `active_KV` 是按 Qwen2-7B 的 KV cache 结构估算的：

```text
n_layer          = 28
n_embd_k_gqa     = 512
n_embd_v_gqa     = 512
bytes_per_elem   = 2   # F16
KV MiB = ctx_size * n_layer * (K + V) * 2 / 1024 / 1024
```

所以：

```text
2048 ctx -> 约 112 MiB 活跃 KV
512  ctx -> 约  28 MiB 活跃 KV
```

## 低功耗注入 d

按下 `d` 后，脚本会模拟：

```text
power = 25% / 未插电
mode  = LOW_POWER_FUSED
ctx   = 512
ngl   = 15
```

同时打印：

```text
[警告] 活跃上下文由 2048 压缩至 512，已有 1536 个历史 Token 被物理截断至 Host 侧！
```

这里 `1536` 来自：

```text
2048 - 512 = 1536
```

语义上对应：

```text
seq_0 -> 低功耗期间保留的短上下文
seq_1 -> 被截断并迁移到 Host 侧的长上下文阴影
```

注意：Python 展示脚本本身不直接读写 llama.cpp KV cache。它负责构造展示状态和调用 C++ PoC。真实 state-buffer 读写和 K/V payload 融合发生在 C++ 可执行文件内部。

## 充电恢复注入 c

按下 `c` 后，脚本模拟：

```text
power = 100% / 已插电
mode  = PERFORMANCE
ctx   = 2048
ngl   = 99
```

如果此前已经按过 `d`，即存在 Host 侧长上下文阴影，脚本会自动调用：

```text
./build/bin/llama-cacheblend-static-fusion
```

调用参数中会传入：

```text
--prompt-a "[seq_0 LOW_POWER_SHORT_CONTEXT] ..."
--prompt-b "[seq_1 HOST_TRUNCATED_LONG_CONTEXT] ..."
```

并打印核心高光日志：

```text
[高光] 检测到电源恢复！正在调用底层 C++ 扫描 56 组 K/V 载荷偏移，执行 0.5*A + 0.5*B 物理缝合...
```

C++ PoC 成功后，脚本会提取摘要：

```text
[C++ 摘要] fused 56 K/V payload ranges, total payload bytes = ...
[C++ 摘要] injected fused state into seq_id=2, bytes=...
[C++ 验证] [verify] step=...
```

最后打印：

```text
[高光] 融合状态已注入 seq_2，全血续写成功！
```

## 安全展示模式与联动服务模式

默认运行：

```bash
cd /home/wante/DPL/Embodied-Oms/llama.cpp
python demo_serving_visualization.py
```

默认是安全展示模式：

```text
不启动 llama-server
不抢占 8080 端口
不常驻占用显存
只在按 c 恢复时调用 C++ 融合 PoC
```

这是为了防止展示脚本和 C++ PoC 同时加载模型、争抢 8GB 显存。

如果需要现场联动 `llama-server` 切档，可以加：

```bash
python demo_serving_visualization.py --launch-server
```

该模式会在状态切换时启动对应档位的 `llama-server`：

```text
PERFORMANCE     -> -ngl 99 --ctx-size 2048
LOW_POWER_FUSED -> -ngl 15 --ctx-size 512
```

演示时建议优先使用默认安全模式，确认 C++ 融合输出稳定后，再考虑 `--launch-server`。

## 常用运行命令

基础演示：

```bash
cd /home/wante/DPL/Embodied-Oms/llama.cpp
python demo_serving_visualization.py
```

显示完整 C++ 日志：

```bash
python demo_serving_visualization.py --show-cpp-log
```

使用更短的 C++ PoC token 数，减少演示等待：

```bash
python demo_serving_visualization.py --fusion-tokens 8 --n-verify 1
```

让 C++ PoC 使用 CPU 路径，避免显存压力：

```bash
python demo_serving_visualization.py --fusion-ngl 0
```

让 C++ PoC 使用更多 GPU offload：

```bash
python demo_serving_visualization.py --fusion-ngl 99
```

联动启动 `llama-server`：

```bash
python demo_serving_visualization.py --launch-server
```

## 交互流程建议

向导师演示时，可以按下面顺序走：

1. 启动脚本。
2. 展示初始状态看盘：`PERFORMANCE / 100% / 2048 ctx / active_KV 112 MiB`。
3. 按 `d`。
4. 展示低功耗切换：`LOW_POWER_FUSED / 25% / 512 ctx / active_KV 28 MiB`。
5. 强调日志：`1536 个历史 Token 被物理截断至 Host 侧`。
6. 按 `c`。
7. 展示 C++ 跨语言调用。
8. 展示 `fused 56 K/V payload ranges`。
9. 展示 `injected fused state into seq_id=2`。
10. 展示 `[verify]` token/logit 输出。
11. 总结：低功耗下压缩活跃窗口，恢复时通过 C++ state-buffer 级 KV 静态融合把长短上下文重新缝合。

## 当前技术边界

当前版本已经真实完成：

- Python 手动电量 mock。
- `PERFORMANCE` 与 `LOW_POWER_FUSED` 状态机。
- 终端单键实时注入。
- 活跃 KV cache 估算。
- Host shadow token 可视化。
- 跨语言调用 C++ PoC。
- C++ state-buffer 扫描。
- 56 组 K/V payload offset 定位。
- F16/F32 类型安全线性融合。
- `seq_id=2` state 注入。
- 融合后 forward decode 验证。

当前版本仍然没有做：

- 真实从运行中的 `llama-server` 提取 KV cache。
- 真实把 Python 服务侧的 seq_0/seq_1 state 文件传给 C++。
- 不等长 KV 的切片对齐。
- RoPE position 重映射。
- RAG shard 到 KV payload 的精确语义索引。
- 在线请求级 CacheBlend 调度。

也就是说，今天下午完成的是一个重要的 Level 1 闭环：

```text
等长 state-buffer -> K/V payload 静态融合 -> seq_2 注入 -> 可继续 decode
```

下一阶段可以推进到：

```text
不等长 state-buffer -> prefix 对齐 -> host-side 长上下文 shard 选择 -> 在线恢复融合
```

## 为什么这一步重要

这次改进把原先 Python 中“CacheBlend 知识融合”的纯模拟日志，推进到了真实 llama.cpp C++ 内存层：

```text
不是只打印融合发生了，
而是真的导出 KV state，
真的扫描 K/V payload offset，
真的执行 F16/F32 数值融合，
真的写回 seq_2，
真的继续 forward decode。
```

因此它已经具备向导师展示的工程可信度：Python 展示层负责状态机与演示叙事，C++ PoC 负责底层 KV state 融合验证，两者通过 `subprocess.run()` 串成一个可复现闭环。

