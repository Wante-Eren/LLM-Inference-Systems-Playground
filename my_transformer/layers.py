import math
import torch
import torch.nn as nn
from .triton_ops import triton_fused_attention

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    公式: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, use_triton=False):
        # 【最小改动】：加入硬件拦截分支，如果 use_triton=True，且在 GPU 上，直接走底层流
        if use_triton and q.is_cuda:
            context = triton_fused_attention(q, k, v)
            return context, None
            
        # 输入维度: [batch_size, n_heads, seq_len, d_k]
        d_k = q.size(-1)
        
        # 1. 计算注意力得分 (Q * K^T) / sqrt(d_k)
        # [B, H, S, d_k] x [B, H, d_k, S] -> [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 如果有掩码（比如 Padding Mask），将掩码为 0 的地方赋予一个极小值
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax 归一化得到权重
        attn_weights = self.softmax(scores)
        
        # 4. 权重与 V 相乘得到上下文向量: [B, H, S, S] x [B, H, S, d_v] -> [B, H, S, d_v]
        context = torch.matmul(attn_weights, v)
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除！"

        # 定义 Q, K, V 的线性映射层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # 最后的输出线性层
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    # 【核心修复】：加上 use_triton=False 参数
    def forward(self, q, k, v, mask=None, use_triton=False):
        batch_size = q.size(0)

        # 1. 线性变换 -> 拆分成多头 -> 维度置换
        # 维度变化: [B, S, d_model] -> [B, S, H, d_k] -> [B, H, S, d_k]
        Q = self.W_Q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 调整 mask 维度以匹配多头计算: [B, 1, S, S]
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 3. 计算点积注意力
        # 【核心修复】：向下层传递 use_triton
        context, attn_weights = self.attention(Q, K, V, mask, use_triton=use_triton)

        # 4. 把多头合并回来 (Concat)
        # [B, H, S, d_v] -> [B, S, H, d_v] -> [B, S, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 通过最后的线性层
        output = self.fc(context)
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络 (FFN)
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [B, S, d_model] -> [B, S, d_ff] -> [B, S, d_model]
        return self.fc2(self.relu(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    """
    完整的 Transformer 编码器层 (Standard Post-LN 结构)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # 【核心修复】：加上 use_triton=False 参数
    def forward(self, x, mask=None, use_triton=False):
        # 1. Sub-layer 1: Multi-Head Attention + Residual + LayerNorm
        # 【核心修复】：向下层传递 use_triton
        attn_out, attn_weights = self.mha(x, x, x, mask, use_triton=use_triton)
        x = self.layernorm1(x + self.dropout(attn_out))
        
        # 2. Sub-layer 2: Feed Forward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_out))
        
        return x, attn_weights
        
if __name__ == "__main__":
    print("="*20 + " 开始运行 Transformer 模块测试 " + "="*20)
    
    # 1. 模拟超参数设置
    batch_size = 2    # 批次大小
    seq_len = 5       # 句子最大长度
    d_model = 512     # 嵌入维度
    n_heads = 8       # 注意力头数
    d_ff = 2048       # FFN 内部隐层维度
    
    # 2. 构造随机输入数据 [Batch_size, Seq_len, d_model]
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    print(f"输入张量的初始维度 (Input Shape): {dummy_input.shape}")
    
    # 3. 构造一个 Padding Mask
    mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ]).unsqueeze(1) # [B, 1, S]
    mask = mask.repeat(1, seq_len, 1) 
    print(f"掩码张量的维度 (Mask Shape): {mask.shape}")

    # 4. 实例化我们自己写的 Transformer 编码层
    encoder_layer = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=0.1)
    encoder_layer.eval() 
    
    # 5. 前向传播
    with torch.no_grad():
        output, weights = encoder_layer(dummy_input, mask=mask)
        
    print("\n" + "-"*50)
    print(f"前向传播成功！")
    print(f"输出张量的最终维度 (Output Shape): {output.shape}  <-- 必须与 Input 一致")
    print(f"注意力权重矩阵维度 (Attention Weights Shape): {weights.shape} <-- [B, H, S, S]")
    print("-"*50)
    
    # 6. 验证 Mask 是否生效
    print("\n[验证] 检查第二句话的 Attention 权重是否成功屏蔽了 Padding 部分:")
    sample_weights = weights[1, 0, 0, :]
    print(f"第二句话首个 Token 对整句的注意力分布:\n{sample_weights}")
    print("提示：如果最后两个数值极度接近于 0 (例如小于 1e-5)，说明 Mask 完美生效！")
    
    print("="*60)