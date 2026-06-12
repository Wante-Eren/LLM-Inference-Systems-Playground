import torch
import triton
import triton.language as tl

# 1. 硬件层 Kernel：最小改动，只做最核心的 QK^T + Softmax + V 融合
@triton.jit
def _fused_attn_kernel_simple(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
    # N_CTX=S BLOCK_M=S, BLOCK_N=S, BLOCK_D=D
):
    # 计算当前处理的 Batch * Head 的一维逻辑 ID
    """
    `[B, S, d_model]` 重新组织为 `[B, H, S, D]` 的多头表示，d_model = 表示每个向量维度
    其中 `B` 表示 batch size（批大小），`H` 表示 attention head number（注意力头数量），
    `S` 表示 sequence length（序列长度），`D` 表示每个 head 的 hidden dimension（隐藏维度）。
    """
    pid = tl.program_id(0)
    
    # 极简指针定位：一网打尽所有多维偏移
    # 假设每一步处理固定的维度大小
    q_offset = pid * N_CTX * BLOCK_D
    
    offs_m = tl.arange(0, BLOCK_M) # BLOCK_M = S = 2 -> 得到一维行索引: [0, 1]
    offs_n = tl.arange(0, BLOCK_N) # BLOCK_N = S = 2 -> 得到一维列索引: [0, 1]
    offs_d = tl.arange(0, BLOCK_D) # BLOCK_D = D = 3 -> 得到一维特征索引: [0, 1, 2]
    
    # 从 4060 显存中流式加载数据到片上高速缓存 SRAM
    q = tl.load(Q_ptr + q_offset + offs_m[:, None] * BLOCK_D + offs_d[None, :])
    k = tl.load(K_ptr + q_offset + offs_n[:, None] * BLOCK_D + offs_d[None, :])
    v = tl.load(V_ptr + q_offset + offs_n[:, None] * BLOCK_D + offs_d[None, :])
    
    # 2. 在 SRAM 内部一气呵成完成矩阵计算
    scores = tl.dot(q, tl.trans(k)) / 8.0
    # 【修复点1】：算完指数后，强制转回 fp16，为了能跟 fp16 的 v 对齐做矩阵乘法
    attn_weights = tl.math.exp(scores).to(tl.float16) 
    
    output = tl.dot(attn_weights, v)
    
    # 3. 结果写回主显存
    # 【修复点2】：由于 tl.dot 累加器默认会输出 fp32，写回显存前再强转回 fp16
    tl.store(Out_ptr + q_offset + offs_m[:, None] * BLOCK_D + offs_d[None, :], output.to(tl.float16))

# 2. 上层 PyTorch 包装层接口
def triton_fused_attention(q, k, v):
    # 转换为 4060 最爱的半精度（Float16）并保证显存连续
    q = q.contiguous().half()
    k = k.contiguous().half()
    v = v.contiguous().half()
    
    B, H, S, D = q.shape
    out = torch.empty_like(q)
    
    # 极简硬件网格：为 4060 分配 B * H 个逻辑任务块
    grid = (B * H, )
    
    # 触发 Triton 2.1.0 编译器轰鸣起飞
    _fused_attn_kernel_simple[grid](
        q, k, v, out,
        S,
        BLOCK_M=S, BLOCK_N=S, BLOCK_D=D
    )
    return out