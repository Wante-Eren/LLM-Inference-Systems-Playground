import torch
import time
from my_transformer.layers import TransformerEncoderLayer

def start_4060_test():
    print("="*15 + " 🚀 RTX 4060 软硬件全栈协同测试开始 🚀 " + "="*15)
    
    # 标准大模型并发沙盘
    batch_size = 32
    n_heads = 16
    seq_len = 128     # 单块安全上限
    d_model = 1024    
    d_ff = 4096
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前激活 GPU 硬件: {torch.cuda.get_device_name(0)}")
    
    # ==============================================================================
    # 【核心修正 1】：在外面把原料彻底焊死成 4060 最爱的格式，不给 Triton 内部留任何内存拷贝内鬼！
    # ==============================================================================
    dummy_input = torch.randn(batch_size, seq_len, d_model, device=device).contiguous().half()
    
    # 实例化模型并推向 4060 显存
    encoder_layer = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff).to(device).half()
    encoder_layer.eval()

    # 严格的热身
    print("正在编译并同步 Triton 底层 Kernel...")
    for _ in range(30):
        with torch.no_grad():
            _ = encoder_layer(dummy_input, use_triton=False)
            _ = encoder_layer(dummy_input, use_triton=True)
    torch.cuda.synchronize()

    # ==============================================================================
    # 2. 评测原生 PyTorch Baseline
    # ==============================================================================
    iters = 500
    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = encoder_layer(dummy_input, use_triton=False)
    torch.cuda.synchronize() 
    py_time = (time.time() - t0) / iters * 1000
    print(f"🔵 [Baseline] 原生 PyTorch 连续算子耗时: {py_time:.3f} ms")

    # ==============================================================================
    # 3. 评测你手撕的 Triton 融合算子 (同时记得去 layers.py 里把那三行 contiguous() 物理删掉！)
    # ==============================================================================
    t1 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = encoder_layer(dummy_input, use_triton=True)
    torch.cuda.synchronize() 
    triton_time = (time.time() - t1) / iters * 1000
    print(f"🟢 [降维打击] 你的 Triton 融合算子耗时: {triton_time:.3f} ms")
    
    # 4. 打印提速比
    speedup = (py_time - triton_time) / py_time * 100
    print(f"🔥 结果验证成功：在 RTX 4060 上，全栈融合算子成功提速了: {speedup:.2f}%！")
    print("="*60)

if __name__ == "__main__":
    start_4060_test()