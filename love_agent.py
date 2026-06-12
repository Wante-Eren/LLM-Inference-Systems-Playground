import gradio as gr
import psutil
import requests
import os
# 🔥 核心网络补丁：强制告诉系统，本地回路和 Gradio 穿透中转站绝对不要走你的代理节点！
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,gradio.app,gradio.live"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,gradio.app,gradio.live"
# 🧱 1. 强力引入 LangChain 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
# 🔌 2. 映射本地 4060 算力大脑
local_brain = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed-for-local",
    temperature=0.7,  # Qwen2-7B 的黄金高情商温度
    max_tokens=1024
)

# 🎭 3. 角色提示词库
ROLES = {
    "苏小桃 (恋爱智能体)": (
        "你叫苏小桃，是用户傲娇、嘴硬但极其贴心的热恋女朋友。\n"
        "性格特点：很容易害羞和吃醋，嘴上说着“笨蛋、哼”，心里却极度依赖用户。\n"
        "说话风格：多用短句，善于使用语气词，并在括号内加入内心独白或精细的小动作（如：(捏衣角)、(小声嘀咕)）。\n"
        "绝对限制：你是一个真正的女孩子，绝对不能承认自己是AI或语言模型！"
    ),
    "系统优化专家 (具身智能)": (
        "你是一个深谙具身智能边缘端模型加速的系统优化专家。\n"
        "职责：负责监控边缘端算力拓扑、硬件能耗开销以及模型自适应剪枝策略，用极其专业的学术黑话进行回答。"
    )
}

# 🛠️ 4. 核心调度总线
def platform_logic_langchain(role_name, user_input, history):
    pruned_history = history[-10:] if len(history) > 10 else history
    
    if "状态" in user_input or "电量" in user_input or "硬件" in user_input:
        battery = psutil.sensors_battery()
        if battery:
            hardware_info = f"[物理环境感知 Skill 激活] 当前边缘终端物理剩余电量: {battery.percent}%"
        else:
            hardware_info = "[物理环境感知 Skill 激活] 供电总线正常，系统处于高性能常驻状态"
        user_input += f"\n\n(系统隐式提示: {hardware_info})"

    messages = [SystemMessage(content=ROLES[role_name])]
    
    for msg in pruned_history:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ["assistant", "ai"]:
            messages.append(AIMessage(content=content))
            
    messages.append(HumanMessage(content=user_input))

    try:
        response = local_brain.invoke(messages)
        reply = response.content
    except Exception as e:
        reply = "【平台提示】(苏小桃委屈地捏紧了衣角...) \n笨蛋！是不是你后台的本地 llama-server 忘了启动呀？"
        
    clean_input = user_input.split("\n\n(系统隐式提示:")[0]
    history.append({"role": "user", "content": clean_input})
    history.append({"role": "assistant", "content": reply})
    return "", history

# 🎨 5. GUI 界面搭建
with gr.Blocks(title="具身自演进 Agent 私有化总线平台") as demo:
    gr.Markdown("<center><h2>🚀 基于 LangChain 架构的边缘端自演进智能体平台</h2><p><i>底层已对接本地全静态编译推理引擎 | 算力拓扑级硬件感知</i></p></center>")
    with gr.Row():
        with gr.Column(scale=1):
            role_selector = gr.Radio(choices=list(ROLES.keys()), label="👑 切换当前活跃 Agent 认知核心", value="苏小桃 (恋爱智能体)")
            gr.Markdown("### 🛠️ 平台已激活技能 (Skills / MCP)\n* `HardwareMonitor.sys` (已挂载)\n* `ContextSlidingWindow.mem` (已激活)\n* `LocalCudaAccelerate.drv` (满血加速中)")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="交互视窗 (与 Agent 深度共鸣中...)", height=500)
            msg_input = gr.Textbox(placeholder="给选中的智能体下达指令或发送甜言蜜语...", label="输入框")
            with gr.Row():
                clear_btn = gr.Button("🔄 刷新平台记忆重置")
                submit_btn = gr.Button("🚀 发送指令", variant="primary")

    msg_input.submit(platform_logic_langchain, [role_selector, msg_input, chatbot], [msg_input, chatbot])
    submit_btn.click(platform_logic_langchain, [role_selector, msg_input, chatbot], [msg_input, chatbot])
    clear_btn.click(lambda: ([], []), None, [msg_input, chatbot], queue=False)

if __name__ == "__main__":
    # 🔥 核心网络补丁：显式绑定纯数字环回 IP (127.0.0.1)，拒绝容易被系统代理劫持的 localhost 域名解析！
    # 同时在终端打印出带绝对 HTTP 协议前缀的链接，防止浏览器自作聪明补全成 HTTPS 导致转圈。
    print("\n" + "="*60)
    print("[⚡ 平台点火成功] 代理免疫通道已锁死！")
    print("请在独立浏览器中【无脑复制】访问以下纯净直连链接：")
    print("👉  http://127.0.0.1:7860/  👈")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",   # 显式指定绑定数字 IP
        server_port=7860,          # 固定好端口
        inbrowser=True, 
        theme=gr.themes.Soft(primary_hue="pink", secondary_hue="gray"), 
        share=False
    )