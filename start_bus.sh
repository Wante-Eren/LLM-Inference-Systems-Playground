#!/bin/bash

# 默认从脚本位置定位项目，也允许通过环境变量覆盖。
BASE_DIR="${EMBODIED_OMS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LLAMA_DIR="$BASE_DIR/llama.cpp"

echo "========================================================="
echo " 🚀 具身自演进 Agent 私有化总线平台 - 代理免疫一键点火系统"
echo "========================================================="

# 1. 强力清理残余进程（防止端口占用报错）
echo "🧹 正在清理可能残余的后台进程与残留代理网闸..."
pkill -f "dynamic_serving.py"
pkill -f "love_agent.py"
pkill -f "llama-server"
sleep 1

# 2. 检查 Qwen2-7B 模型是否存在
if [ ! -f "$BASE_DIR/models/qwen2-7b-instruct-q4_k_m.gguf" ]; then
    echo "❌ 错误: 未在 $BASE_DIR/models/ 中找到 Qwen2-7B 模型文件！"
    exit 1
fi

# 3. 异步点亮最强大脑 (Qwen2-7B 常驻守护服务)
echo "🧠 正在后台拉起 Qwen2-7B 算力底座 (监听 8080 端口)..."
cd "$LLAMA_DIR" || exit
nohup python dynamic_serving.py > "$LLAMA_DIR/serving.log" 2>&1 &

# 等待底座初始化
sleep 3

# 4. 异步拉起嘴巴 (Gradio 前端 + LangChain 消息流)
echo "🎨 正在后台构建 Gradio 6.0 交互视窗 (锁死 127.0.0.1 代理免疫通道)..."
cd "$BASE_DIR" || exit
nohup python love_agent.py > "$BASE_DIR/agent.log" 2>&1 &

# 等待前端端口绑定
sleep 2

# 5. 直接高亮打印免代理直连链接，告别转圈圈！
echo -e "\n🎉 点火成功！智能体总线平台已在赛博空间常驻！"
echo "========================================================="
echo -e "👉 本地独立浏览器【直接点击或无脑复制】此纯净直连链接访问:"
echo -e "\033[1;32mhttp://127.0.0.1:7860/\033[0m"
echo "========================================================="
echo "🌐 提示: 如需分享给同学/老师，请新开终端执行: cpolar http 7860"
echo "💡 提示: 笔记本电源已锁死常开，请勿合盖。按 Ctrl+C 退出脚本监控（后台仍常驻）。"
echo "========================================================="

# 保持终端前台挂起监控，方便随时 Ctrl+C 优雅退出
trap "echo -e '\n[系统] 退出监控，后台常驻服务保持运行。'; exit" INT
while true; do sleep 1; done
