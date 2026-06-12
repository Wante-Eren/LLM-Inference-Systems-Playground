import os
from modelscope import snapshot_download

print("=" * 10 + " 正在通过魔搭国内骨干网极速抓取 TinyLlama " + "=" * 10)

# 明确指定下载到项目目录下的 models 文件夹
model_dir = snapshot_download(
    'Huber/TinyLlama-1.1B-Chat-v1.0', 
    cache_dir='./models'
)

print(f"\n🎉 极速下载完成！模型已存入: {model_dir}")