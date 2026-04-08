# 消除Hugging Face tokenizers并行提示
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 设置国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import streamlit as st
import uuid
import time
import json
import smtplib
import tempfile
from datetime import datetime
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from openai import OpenAI
from ltp import LTP
from aip import AipNlp

# 自编模块
from chat import smart_chat_stream
from state_aware import get_max_dependency_depth, get_sentiment_analysis, get_novelty, get_prior_knowledge
from strategy import ProactiveScaffoldingSystem
from logger import logger
from utils import process_single_image

# ==============================================================================
# 0. 基础配置与工具函数
# ==============================================================================

# ── 启动时在终端选择实验信息（只读取一次，使用环境变量持久化）─────────────────
_GROUP_ENV_KEY = "EXPERIMENT_GROUP"
_PARTICIPANT_ENV_KEY = "PARTICIPANT_ID"
if _GROUP_ENV_KEY not in os.environ:
    print("\n" + "="*50)
    print("  人智交互自适应支持系统 — 实验信息配置")
    print("="*50)
    # 选择组别
    print("  [1] 实验组（启用自适应策略）")
    print("  [2] 对照组（三维度固定为 0.5）")
    print("-"*50)
    while True:
        choice = input("  请输入组别编号 (1 或 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  ❌ 输入无效，请重新输入 1 或 2")
    os.environ[_GROUP_ENV_KEY] = "control" if choice == "2" else "experiment"
    group_label = "对照组" if choice == "2" else "实验组"
    # 输入被试编号
    print("-"*50)
    while True:
        pid = input("  请输入被试编号 (如 P01): ").strip()
        if pid:
            break
        print("  ❌ 被试编号不能为空，请重新输入")
    os.environ[_PARTICIPANT_ENV_KEY] = pid
    print(f"\n  ✅ 已选择【{group_label}】，被试编号【{pid}】，启动系统...\n")

EXPERIMENT_GROUP = os.environ[_GROUP_ENV_KEY]      # "experiment" 或 "control"
IS_CONTROL_GROUP = (EXPERIMENT_GROUP == "control")
PARTICIPANT_ID   = os.environ[_PARTICIPANT_ENV_KEY] # 被试编号，如 P01

# ── 本地日志文件路径（与 app.py 同目录下的 experiment_logs/ 文件夹）─────────
_LOG_ENV_KEY = "EXPERIMENT_LOG_FILE"
if _LOG_ENV_KEY not in os.environ:
    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_logs")
    os.makedirs(_log_dir, exist_ok=True)
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_filename = f"{PARTICIPANT_ID}_{EXPERIMENT_GROUP}_{_timestamp}.jsonl"
    os.environ[_LOG_ENV_KEY] = os.path.join(_log_dir, _log_filename)
    print(f"  📁 实验数据将保存至: {os.environ[_LOG_ENV_KEY]}\n")

LOCAL_LOG_FILE = os.environ[_LOG_ENV_KEY]

if not hasattr(st, "secrets"):
    st.secrets = {}

try:
    deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    baidu_app_id = st.secrets.get("APP_ID", "")
    baidu_api_key = st.secrets.get("API_KEY", "")
    baidu_secret_key = st.secrets.get("SECRET_KEY", "")
    baidu_vl_api_key = st.secrets.get("BAIDU_VL_API_KEY", "")
    EMAIL_SENDER = st.secrets.get("EMAIL_SENDER", "")
    EMAIL_PASSWORD = st.secrets.get("EMAIL_PASSWORD", "")
    EMAIL_RECEIVER = st.secrets.get("EMAIL_RECEIVER", "")
except Exception:
    (deepseek_api_key,baidu_app_id,baidu_api_key,baidu_secret_key,
     baidu_vl_api_key,EMAIL_SENDER,EMAIL_PASSWORD,EMAIL_RECEIVER) = '', '', '', '', '', '', '', ''

try:
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    baidu_client = AipNlp(baidu_app_id, baidu_api_key, baidu_secret_key)
    vl_client = OpenAI(base_url='https://qianfan.baidubce.com/v2',api_key=baidu_vl_api_key)
except Exception as e:
    client, baidu_client, vl_client = None, None, None

@st.cache_resource
def load_ltp_model():
    logger.info("📦 开始加载 LTP 模型 (系统初始化)...")
    logger.info("✅ LTP 模型加载完成！")
    return LTP()
ltp_model = load_ltp_model()

def save_log_local(log_data: dict) -> bool:
    """
    将单条实验日志以 JSON Lines 格式追加写入本地文件。
    每行一条记录，程序崩溃也不丢失已有数据。
    """
    try:
        with open(LOCAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        logger.info(f"💾 日志已追加写入本地文件: {LOCAL_LOG_FILE}")
        return True
    except Exception as e:
        logger.error(f"❌ 本地日志写入失败: {e}")
        return False


def send_log_email(log_data):
    # 检查配置是否存在
    if not EMAIL_SENDER or not EMAIL_PASSWORD or EMAIL_SENDER == "NO":
        logger.info("邮件配置缺失或已禁用（EMAIL_SENDER=NO），跳过发送。")
        return False
    
    try:
        # 1. 准备邮件内容
        json_str = json.dumps(log_data, indent=4, ensure_ascii=False)
        subject = f"【实验日志】{log_data.get('participant_id', 'Unknown')}-{log_data.get('task_number', 'Unknown')}(Session {log_data.get('session_id', 'Unknown')[:8]}...)"
        
        message = MIMEText(json_str, 'plain', 'utf-8')
        message['From'] = formataddr(("AI实验系统", EMAIL_SENDER))
        message['To'] = formataddr(("研究员", EMAIL_RECEIVER))
        message['Subject'] = Header(subject, 'utf-8')

        # 2. 连接 SMTP 服务器 (推荐使用 587 端口 + starttls)
        smtp_server = "smtp.qq.com"
        
        server = smtplib.SMTP(smtp_server, 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        
        # 3. 登录并发送
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], message.as_string())
        server.quit()
        
        logger.info("📧 邮件发送成功！")
        return True

    except Exception as e:
        logger.error(f"❌ 邮件发送失败！详细错误: {e}")
        return False
    
# ==============================================================================
# 1. 后端逻辑
# ==============================================================================

def get_response_meta(task_key, task_text, user_query, chat_messages, mind_map_text):
    """
    计算感知特征与交互策略，返回元数据 dict 和 mode_settings。
    不再在此处调用大模型（大模型的流式调用移到前端渲染部分，以便实时显示）。
    【注意：为了节省时间和避免程序错误，思维导图过于复杂时直接截断，因此先验知识评分只是定性判断方便决策，不可作为真正的评分数据。】
    """
    start_ts = datetime.now()
    logger.info(f"📨 收到用户请求: {user_query[:30]}...")

    logger.info("⚙️ 正在计算感知特征 (A1, A2, A3, C1, C2)...")
    try:
        a1 = get_max_dependency_depth(user_query, ltp_model)
        a2 = get_sentiment_analysis(user_query, baidu_client)
        a3 = get_novelty(task_text, user_query, baidu_client)
        c1 = len([m for m in chat_messages if m.get('role') == 'user'])
        c2 = get_prior_knowledge(task_key, mind_map_text, client)
        logger.info(f"📊 特征计算结果: A1={a1}, A2={a2:.4f}, A3={a3:.4f}, C1={c1}, C2={c2}")
    except Exception as e:
        logger.error(f"❌ 特征计算失败，使用默认安全值: {e}")
        a1, a2, a3, c1, c2 = 0, 0, 0, 0, 0

    engine = ProactiveScaffoldingSystem(task_key=task_key, control_group=IS_CONTROL_GROUP)
    user_features = {'syntax': a1, 'sentiment': a2, 'novelty': a3, 'turn': c1, 'priorknowledge': c2}
    mode_settings, strategy_meta = engine.decide_interaction_strategy(a1=a1, a2=a2, a3=a3, c1=c1, c2=c2)

    packet_meta = {
        "request_timestamp": start_ts.strftime('%Y-%m-%d %H:%M:%S.%f'),
        "user_query": user_query,
        "user_query_length": len(user_query),
        "task_context": task_text,
        "mindmap_char_count": len(mind_map_text) if mind_map_text else 0,
        "user_features": user_features,
        "calibrated_state": strategy_meta["calibrated_state"],
        "interaction_settings": mode_settings,
        "matched_rules": strategy_meta["matched_rules"],
        "matched_rule_source": strategy_meta["matched_rule_source"],
        "is_risk_detected": strategy_meta["is_risk_detected"],
        "final_strategy_source": strategy_meta["final_strategy_source"],
    }
    return packet_meta, mode_settings


# ==============================================================================
# 2. 前端界面
# ==============================================================================

st.set_page_config(page_title="人智交互自适应支持系统", page_icon="🎓", layout="wide")

TASKS = {
    "task1": "你在听新闻的过程中，主持人提到了“低空经济”这个概念。请使用生成式人工智能进行搜索，了解“低空经济”这一新产业的定义、特征、分类、地位以及发展措施。请用思维导图的形式呈现最终的搜索结果。",
    "task2": "你家里的老人需要购买一部智能手机，但对智能手机的类型、性能、主要功能、价位、是否适合老人使用等的信息都不了解。请使用生成式人工智能进行搜索，给出你所认为的适合老人使用的智能手机。请用思维导图的形式呈现最终的搜索结果。",
    "task3": "想象一下，您最近出现了发热、恶心、呕吐、腹泻等症状，偶尔还会打鼾。你想知道什么疾病都会导致上述症状？请使用生成式人工智能进行搜索，请用思维导图的形式呈现最终的搜索结果。"
}
TASK_LABELS = {"任务一：低空经济": "task1", "任务二：智能手机": "task2", "任务三：判断病情": "task3"}

# --- Session State 初始化 ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {} 
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "experiment_logs" not in st.session_state:
    st.session_state.experiment_logs = []
if "current_turn_mindmap" not in st.session_state:
    st.session_state.current_turn_mindmap = None 
if "current_turn_mindmap_input_type" not in st.session_state:
    st.session_state.current_turn_mindmap_input_type = None
    
# --- 侧边栏 ---
with st.sidebar:
    st.title("🗂️ 历史会话")
    
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())

        st.session_state.conversations[new_id] = {
            "title": f"对话 {len(st.session_state.conversations)+1}", 
            "messages": [], 
            "task": None,
            "last_modified": time.time()
        }
        st.session_state.current_chat_id = new_id
        st.session_state.current_turn_mindmap = None
        st.session_state.current_turn_mindmap_input_type = None
        st.rerun()

    st.divider()
    
    chat_ids = sorted(
        st.session_state.conversations.keys(),
        key=lambda k: st.session_state.conversations[k].get("last_modified", 0),
        reverse=True
    )
    
    if chat_ids:
        if st.session_state.current_chat_id not in chat_ids:
             st.session_state.current_chat_id = chat_ids[0]
        
        display_options = {cid: data['title'] for cid, data in st.session_state.conversations.items()}
        
        selected_id = st.radio(
            "切换会话：", 
            options=chat_ids, 
            format_func=lambda x: display_options[x], 
            key="chat_sel",
            index=chat_ids.index(st.session_state.current_chat_id)
        )
        
        if selected_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_id
            st.session_state.current_turn_mindmap = None
            st.session_state.current_turn_mindmap_input_type = None
            st.rerun()
    else:
        st.info("暂无历史会话，请点击上方按钮新建。")

# --- 主界面 ---
st.title("🎓 人智交互自适应支持系统")
st.caption("Powered by DeepSeek & Streamlit")

current_id = st.session_state.current_chat_id

if current_id and current_id in st.session_state.conversations:
    current_chat = st.session_state.conversations[current_id]
    
    if current_chat.get("task") is None:
        st.info("👋 欢迎！为了提供更精准的回答，请先选择您当前进行的任务场景。")
        with st.container(border=True):
            st.subheader("请选择当前任务：")
            
            selected_label = st.radio(
                "任务列表",
                options=list(TASK_LABELS.keys()),
                index=0,
                key=f"task_radio_{current_id}",
                horizontal=True
            )
            
            preview_task_key = TASK_LABELS[selected_label]
            st.markdown(f"**📝 任务描述：**\n\n> {TASKS[preview_task_key]}")
            
            st.write("")
            
            if st.button("✅ 确认并开始对话", type="primary"):
                current_chat["task"] = current_chat["task"] = TASK_LABELS[selected_label]
                current_chat["last_modified"] = time.time()
                st.rerun()
    else:
        task_key = current_chat["task"]
        task_text = TASKS[task_key]
        current_label = [k for k, v in TASK_LABELS.items() if v == task_key][0]
        with st.expander(f"📌 当前任务：{current_label} (点击查看详情)", expanded=False):
            st.info(task_text)

        for msg in current_chat["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        need_mindmap = st.session_state.current_turn_mindmap is None

        if need_mindmap:
            with st.container(border=True):
                    st.warning("🔒 提问锁定：为了更好地辅助您，请先更新您当前的知识状态（思维导图）。")
                    
                    dynamic_key_suffix = f"{current_id}_{len(current_chat['messages'])}"

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 方式一：上传截图")
                        uploaded_file = st.file_uploader("上传思维导图图片", type=["png", "jpg", "jpeg"], key=f"mm_uploader_{dynamic_key_suffix}")
                        
                    with col2:
                        st.markdown("#### 方式二：粘贴文本")
                        text_input = st.text_area("直接粘贴Markdown格式的节点文本", height=150, key=f"mm_text_{dynamic_key_suffix}")
                    
                    if st.button("🚀 提交并开始提问", type="primary", use_container_width=True):
                        if uploaded_file is not None:
                            with st.spinner("正在识别思维导图结构..."):
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                        tmp_file.write(uploaded_file.getvalue())
                                        tmp_file_path = tmp_file.name
                                    extracted_text = process_single_image(vl_client, tmp_file_path)
                                    os.remove(tmp_file_path)
                                    if extracted_text:
                                        st.session_state.current_turn_mindmap = extracted_text
                                        st.session_state.current_turn_mindmap_input_type = "image"
                                        st.rerun()
                                    else: st.error("识别失败，请重试。")
                                except Exception as e: st.error(f"处理出错: {e}")
                        elif text_input.strip():
                            st.session_state.current_turn_mindmap = text_input
                            st.session_state.current_turn_mindmap_input_type = "text"
                            st.rerun()
                        else:
                            st.toast("请至少上传一张图片或输入一段文本", icon="⚠️")
        else:
            st.success(f"✅ 当前知识状态已更新 (字符数: {len(st.session_state.current_turn_mindmap)})")

        prompt_placeholder = "思维导图已记录，请输入您的问题..." if not need_mindmap else "🚫 请先在上方提交思维导图以解锁提问"
        
        prompt = st.chat_input(prompt_placeholder)

        if prompt:
            if need_mindmap:
                st.toast("🔒 请先提交思维导图！", icon="🚫")
            else:
                # 显示用户消息
                with st.chat_message("user"):
                    st.markdown(prompt)
                current_chat["messages"].append({"role": "user", "content": prompt})
                
                # 先计算特征和策略（含 spinner 提示），再流式渲染模型输出
                with st.spinner("正在分析您的请求，请稍候..."):
                    current_mindmap = st.session_state.current_turn_mindmap
                    packet_meta, mode_settings = get_response_meta(
                        task_key, task_text, prompt, current_chat["messages"], current_mindmap
                    )

                full_reasoning = ""
                full_content = ""
                usage_info = {}

                with st.chat_message("assistant"):
                    # 推理过程：用可折叠的 st.status 展示
                    with st.status("系统推理中...", expanded=True) as status_box:
                        reasoning_placeholder = st.empty()

                    # 正式回复：在 status 块外创建占位符，用于逐步追加文本
                    content_placeholder = st.empty()

                    for chunk in smart_chat_stream(current_chat["messages"], mode_settings, client):
                        if chunk["type"] == "reasoning":
                            full_reasoning += chunk["text"]
                            # 在 status 块内实时更新推理内容
                            reasoning_placeholder.markdown(
                                f"> {full_reasoning.replace(chr(10), chr(10) + '> ')}"
                            )
                        elif chunk["type"] == "content":
                            if not full_content:
                                # 第一个 content chunk 到达时，折叠推理框
                                status_box.update(label="✅ 推理完成", state="complete", expanded=False)
                            full_content += chunk["text"]
                            # ★ 核心修复：每收到一个 content chunk 就立即更新占位符，实现流式效果
                            content_placeholder.markdown(full_content + "▌")  # 光标符增强流式感
                        elif chunk["type"] == "done":
                            usage_info = chunk.get("usage", {})

                    # 流结束后去掉光标符，渲染最终完整内容
                    content_placeholder.markdown(full_content)

                # 整理完整 packet 并记录
                end_ts = datetime.now()
                start_ts = datetime.strptime(packet_meta['request_timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                latency_s = round((end_ts - start_ts).total_seconds(), 3)
                logger.info(f"⏱️ 本次交互总耗时: {latency_s:.2f}s")

                # 当前对话轮次（以 user 消息数计）
                turn_index = len([m for m in current_chat["messages"] if m.get("role") == "user"])

                # 注意：为了节省时间和避免程序错误，思维导图过于复杂时直接截断，因此先验知识评分只是定性判断方便决策，不可作为真正的评分数据。

                packet = {
                    # ── 会话标识 ──────────────────────────────────────
                    "session_id": current_id,
                    "participant_id": PARTICIPANT_ID,        # 被试编号，如 P01
                    "experiment_group": EXPERIMENT_GROUP,   # "experiment"=实验组 / "control"=对照组
                    "task_number": task_key,
                    "turn_index": turn_index,
                    "model_chat": "deepseek-reasoner",      # 主对话模型
                    "model_scorer": "deepseek-chat",        # 先验知识评分模型 (C2)

                    # ── 时间戳与性能 ──────────────────────────────────
                    "request_timestamp": packet_meta["request_timestamp"],
                    "response_timestamp": end_ts.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "response_latency_s": latency_s,

                    # ── 用户输入 ──────────────────────────────────────
                    "user_query": packet_meta["user_query"],
                    "user_query_length": packet_meta["user_query_length"],
                    "mindmap_input_type": st.session_state.current_turn_mindmap_input_type,
                    "mindmap_char_count": packet_meta["mindmap_char_count"],
                    "mindmap_text": current_mindmap,

                    # ── 状态感知特征（原始值 + 校准后隶属度）────────
                    "user_features": packet_meta["user_features"],
                    "calibrated_state": packet_meta["calibrated_state"],

                    # ── 策略决策过程 ──────────────────────────────────
                    "matched_rule_source": packet_meta["matched_rule_source"],
                    "matched_rules": packet_meta["matched_rules"],
                    "is_risk_detected": packet_meta["is_risk_detected"],
                    "final_strategy_source": packet_meta["final_strategy_source"],
                    "interaction_settings": packet_meta["interaction_settings"],

                    # ── 模型输出 ──────────────────────────────────────
                    "response_content": full_content,
                    "response_content_length": len(full_content),
                    "reasoning_content": full_reasoning,
                    "reasoning_content_length": len(full_reasoning),

                    # ── Token 消耗（来自 deepseek-reasoner）──────────
                    "prompt_tokens": usage_info.get("prompt_tokens", None),
                    "completion_tokens": usage_info.get("completion_tokens", None),
                    "total_tokens": (
                        usage_info["prompt_tokens"] + usage_info["completion_tokens"]
                        if usage_info.get("prompt_tokens") is not None and usage_info.get("completion_tokens") is not None
                        else None
                    ),
                }

                current_chat["messages"].append({
                    "role": "assistant",
                    "content": full_content,
                    "reasoning": full_reasoning
                })

                st.session_state.experiment_logs.append(packet)
                current_chat["last_modified"] = time.time()

                try:
                    save_log_local(packet)
                except Exception as e:
                    logger.error(f"本地日志写入异常: {e}")

                try:
                    send_log_email(packet)
                except Exception:
                    pass

                # 重置思维导图状态，强制下一轮重新提交
                st.session_state.current_turn_mindmap = None
                st.session_state.current_turn_mindmap_input_type = None

                if len(current_chat["messages"]) == 2:
                    current_chat["title"] = prompt[:10] + "..."

                time.sleep(0.5)
                st.rerun()

else:
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <h3>👋 欢迎使用</h3>
            <p>请点击左侧侧边栏的 <b>"➕ 新建对话"</b> 按钮开始交互。</p>
        </div>
        """,
        unsafe_allow_html=True
    )