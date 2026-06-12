<div align="center">

# Embodied-Oms

### 面向消费级 RTX 笔记本的边缘大模型自适应服务与 KV Cache 融合实验平台

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![llama.cpp](https://img.shields.io/badge/runtime-llama.cpp-111111)](https://github.com/ggml-org/llama.cpp)
[![CUDA](https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个把 **电源感知服务调度、llama.cpp 状态缓冲区融合、Triton 算子实验和本地 Agent**
放进同一条端侧推理链路的研究型项目。

</div>

---

## 演示视频

完整实验演示视频已作为 GitHub Release 附件发布，包含本地 Agent、边缘端服务状态切换与实验验证过程：

> [观看或下载：Embodied-Oms 实验验证视频（MP4）](https://github.com/Wante-Eren/-Embodied-Oms-/releases/download/demo-v1/default.mp4)

视频文件超过 GitHub 普通仓库的单文件限制，因此未写入 Git 历史，避免克隆源码时下载大型媒体文件。

## 项目简介

Embodied-Oms 面向显存和电池预算都有限的消费级边缘设备，探索一个具体问题：

> 当笔记本从高性能插电状态切换到低功耗状态时，如何降低 GPU 与 KV Cache 压力，同时尽量保留长上下文知识，并在资源恢复后继续推理？

项目以 RTX 4060 Laptop、Qwen2-7B-Instruct-Q4_K_M 和 `llama.cpp` 为主要实验环境，包含四条相互衔接的技术路径：

- **电源感知推理守护进程**：根据电池状态在 2048 ctx / 99 层 GPU 卸载与 512 ctx / 15 层 GPU 卸载之间稳定切档。
- **CacheBlend 风格静态融合 PoC**：在 C++ 层导出两段等长序列状态，定位逐层 K/V payload，执行 `0.5 * KV_A + 0.5 * KV_B`，再注入第三个序列继续解码。
- **可视化演示状态机**：通过 `d` / `c` 按键模拟断电与恢复，并跨语言调用 C++ 融合器。
- **Triton 注意力实验**：提供一个轻量 Transformer 编码层和自定义融合注意力 Kernel，用于学习与基准测试。

## 系统架构

```text
                         +-----------------------------+
                         |       Gradio Local Agent    |
                         | love_agent.py / LangChain   |
                         +--------------+--------------+
                                        |
                               OpenAI-compatible API
                                        |
+-------------------+       +-----------v------------+       +--------------------+
| Battery / Mock IO | ----> | Adaptive Serving FSM   | ----> | llama.cpp Runtime  |
| psutil, d/c keys  |       | 2048 ctx <-> 512 ctx   |       | Qwen2 GGUF / CUDA  |
+-------------------+       +-----------+------------+       +----------+---------+
                                        |                               |
                                        | subprocess                    | seq state API
                                        v                               v
                         +--------------+-------------------------------+--+
                         | CacheBlend Static Fusion PoC                    |
                         | Export A/B -> Parse 56 K/V ranges -> Fuse       |
                         | -> Inject seq_2 -> Decode verification           |
                         +--------------------------------------------------+
```

## 核心模块

| 模块 | 入口 | 作用 |
| --- | --- | --- |
| 自适应服务守护 | `llama.cpp/dynamic_serving.py` | 轮询电源状态、稳定切档、管理 `llama-server` 生命周期 |
| 演示可视化 | `llama.cpp/demo_serving_visualization.py` | 手动模拟断电/恢复并调用 C++ 融合 PoC |
| 静态融合 PoC | `llama.cpp/examples/cacheblend-static-fusion/` | 导出、解析、融合、注入 sequence state |
| 本地 Agent | `love_agent.py` | Gradio + LangChain + 本地 OpenAI-compatible API |
| Triton 实验 | `my_transformer/`、`benchmark.py` | 自定义注意力 Kernel 与 PyTorch 基线对比 |
| 一键启动 | `start_bus.sh` | 拉起服务守护进程与本地 Agent |

更完整的设计说明见：

- [`dynamic_serving.md`](llama.cpp/dynamic_serving.md)
- [`demo_serving_visualization.md`](llama.cpp/demo_serving_visualization.md)

## 已跑通的静态 KV 融合 PoC

C++ PoC 使用 `n_seq_max >= 3` 创建三个逻辑序列：

```text
seq_0 = Prompt A
seq_1 = Prompt B
seq_2 = Fused State
```

其执行路径如下：

1. 将等长 Prompt A/B 分别 prefill 到 `seq_0` 和 `seq_1`。
2. 使用 `llama_state_seq_get_size()` 与 `llama_state_seq_get_data()` 导出 host state buffer。
3. 解析 sequence state，定位 Qwen2-7B 的 `28 K + 28 V = 56` 组 payload。
4. 复制 A 的完整元数据拓扑，仅对 K/V payload 做 F16/F32 加权融合。
5. 使用 `llama_state_seq_set_data()` 将结果注入 `seq_2`。
6. 向 `seq_2` 输入一个 token，验证融合状态能够继续前向传播。

一次已验证运行的关键输出：

```text
seq_id=0 state size = 459544 bytes
seq_id=1 state size = 459544 bytes
fused 56 K/V payload ranges, total payload bytes = 458752
injected fused state into seq_id=2, bytes=459544
CacheBlend static fusion PoC completed without crash.
```

## 重要实验边界

本项目当前是研究 PoC，不应把静态线性平均解释为无损知识合并：

- `0.5 * KV_A + 0.5 * KV_B` 是表示空间中的逐位置平均，不等价于模型阅读了 `A + B`。
- 当前 PoC 要求两段序列等长、位置对齐、模型与 KV 类型一致。
- 成功解码证明状态导出、解析、融合和注入链路可执行，不证明语义召回准确率。
- Python 中的异步 CacheBlend worker 是控制层模拟；真实 KV payload 融合发生在 C++ PoC 中。
- 演示脚本中的 Host shadow token 是可视化状态，不是运行中 `llama-server` 的真实 KV 快照。
- 已记录的基础验证使用 `-ngl 0`，因此不能据此宣称 GPU/PCIe 性能收益。

## 快速开始

### 1. 准备环境

```bash
git clone https://github.com/Wante-Eren/-Embodied-Oms-.git
cd ./-Embodied-Oms-

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

模型权重不会提交到 Git。请自行准备 GGUF 模型，并通过环境变量指定：

```bash
export MODEL_BASE=/absolute/path/to/qwen2-7b-instruct-q4_k_m.gguf
```

### 2. 编译 llama.cpp 与静态融合示例

CPU 构建：

```bash
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF
cmake --build llama.cpp/build --target llama-server llama-cacheblend-static-fusion -j
```

CUDA 构建：

```bash
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --target llama-server llama-cacheblend-static-fusion -j
```

### 3. 运行 C++ 静态融合 PoC

```bash
./llama.cpp/build/bin/llama-cacheblend-static-fusion \
  -m "$MODEL_BASE" \
  --n-tokens 8 \
  -ngl 0 \
  --n-verify 1
```

### 4. 运行交互式演示

```bash
python llama.cpp/demo_serving_visualization.py
```

终端按键：

```text
d  模拟 25% 未插电，切入 LOW_POWER_FUSED
c  模拟恢复插电，调用 C++ 静态融合器
q  退出
```

### 5. 运行电源感知服务

```bash
python llama.cpp/dynamic_serving.py
```

或同时启动本地 Agent：

```bash
./start_bus.sh
```

访问 `http://127.0.0.1:7860/`。

### 6. 通过内网穿透进行远程演示

项目支持使用 cpolar 将本机 Gradio 演示界面临时分享给导师或协作者。先确保本地服务与 Agent 已启动：

```bash
./start_bus.sh
```

然后在另一个终端创建临时公网隧道：

```bash
cpolar http 7860
```

将 cpolar 输出的 HTTPS 地址分享给远程观众即可。推荐只穿透 Gradio 前端端口 `7860`，不要直接暴露 llama.cpp 推理接口 `8080`。

当前 Gradio 演示界面没有用户认证能力，因此远程演示应使用临时隧道，并在演示结束后立即关闭 cpolar。不要在公开网络环境中输入敏感信息或长期暴露服务。

### 7. 运行 Triton 基准

```bash
python benchmark.py
```

该实验需要 CUDA、PyTorch 与 Triton。自定义 Kernel 是教学与研究实现，尚未覆盖生产级 causal mask、数值稳定 softmax 和任意 shape。

## 服务状态机

| 状态 | 触发条件 | GPU 卸载 | 活跃上下文 |
| --- | --- | ---: | ---: |
| `PERFORMANCE` | 首次启动或已插电 | 99 层 | 2048 tokens |
| `LOW_POWER_FUSED` | 未插电且电量 `< 30%` | 15 层 | 512 tokens |

进入低功耗状态后，守护进程会阻塞等待重新插电，避免阈值附近频繁重启导致显卡功耗与 CUDA 上下文震荡。

## 配置

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_BASE` | `models/qwen2-7b-instruct-q4_k_m.gguf` | 本地 GGUF 模型路径 |
| `LLAMA_SERVER_BIN` | `llama.cpp/build/bin/llama-server` | 服务二进制路径 |
| `CACHEBLEND_FUSION_BIN` | `llama.cpp/build/bin/llama-cacheblend-static-fusion` | 融合 PoC 二进制路径 |
| `HOST` | `127.0.0.1` | 本地服务绑定地址 |
| `PORT` | `8080` | OpenAI-compatible API 端口 |
| `EMBODIED_OMS_HOME` | 脚本所在目录 | `start_bus.sh` 使用的项目根目录 |

## 路线图

- [x] 电源感知双态服务守护与进程生命周期管理
- [x] 等长 sequence state 的离线 K/V payload 融合
- [x] 融合状态注入与继续解码验证
- [x] 手动电源事件演示闭环
- [ ] 不等长 KV 的位置与切片对齐
- [ ] 排除重复 seed token 的严谨续写路径
- [ ] 真实 `llama-server` 会话状态导出与恢复
- [ ] 语义质量、TTFT、能耗与 PCIe 开销基准
- [ ] 面向真实 CacheBlend 的选择性重计算与融合策略

## 致谢与许可

项目中的 `llama.cpp/` 基于 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，其原始代码与许可证保留在对应目录中。CacheBlend 相关思路受论文 *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion* 启发；当前实现是独立的教学与研究 PoC，并非论文官方实现。

本仓库新增代码以 [MIT License](LICENSE) 开源。模型权重遵循各自模型发布方的许可证，不包含在本仓库中。
