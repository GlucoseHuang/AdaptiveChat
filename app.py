"""人智交互自适应支持系统 — 自由对话版。

在原实验原型基础上改造:
- 去掉实验任务选择、思维导图上传、组别/被试提示
- 去掉本地日志落盘与邮件推送
- 保留: 状态感知 → 策略选择 → 自适应回复 的核心三段式闭环
- 保留: 多会话侧边栏、deepseek-reasoner 思维链流式输出
- 新增: 每轮 LLM 自动识别任务类型(事实型 / 解释型 / 探索型),
        选用对应的策略规则库
- 新增: 每轮回复前渲染一张"自适应感知"卡片,顶部一行策略摘要,
        内嵌 <details> 折叠区显示任务判断 + 5 维校准 + 分层推理
        + 命中规则 + 策略向量

先验知识 C2 简化处理:
  - 第 1 轮视为"低"
  - 第 2 轮及以后视为"高"
这样无需思维导图输入也能让策略系统保持运转。
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# HF 镜像只在用户终端显式设置时才生效 (HF_ENDPOINT=...);不强制改默认值
# (hf-mirror.com 镜像缺少 LTP/small 的 config.json,会导致 LTP 模型加载失败)

import streamlit as st
import uuid
import time
from openai import OpenAI
from ltp import LTP
from aip import AipNlp

from chat import smart_chat_stream, build_bubble_text, classify_task_type, TASK_LABEL
from state_aware import get_max_dependency_depth, get_sentiment_analysis, get_novelty
from strategy import ProactiveScaffoldingSystem
from logger import logger

# ==============================================================================
# 0. 配置与客户端
# ==============================================================================

# 任务类型由 LLM 每轮自动识别(见 classify_task_type),此处不再硬编码

# C2 校准锚点:简化处理,首轮视为"低",后续视为"高";100 在校准后会落在"高"集合
C2_RAW_LOW = 0.0
C2_RAW_HIGH = 100.0

if not hasattr(st, "secrets"):
    st.secrets = {}

try:
    deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    baidu_app_id = st.secrets.get("APP_ID", "")
    baidu_api_key = st.secrets.get("API_KEY", "")
    baidu_secret_key = st.secrets.get("SECRET_KEY", "")
except Exception:
    deepseek_api_key = baidu_app_id = baidu_api_key = baidu_secret_key = ""

try:
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    baidu_client = AipNlp(baidu_app_id, baidu_api_key, baidu_secret_key)
except Exception as e:
    logger.error(f"客户端初始化失败: {e}")
    client, baidu_client = None, None


@st.cache_resource
def load_ltp_model():
    logger.info("开始加载 LTP 模型...")
    return LTP()


ltp_model = load_ltp_model()


# ==============================================================================
# 1. 后端逻辑
# ==============================================================================

def compute_user_state(user_query, prior_user_messages):
    """实时计算 A1/A2/A3;返回 (a1, a2, a3, c1, c2)。

    prior_user_messages 是当前轮之前所有的用户消息内容(已按时间顺序)。
    """
    a1 = get_max_dependency_depth(user_query, ltp_model)
    a2 = get_sentiment_analysis(user_query, baidu_client)

    if prior_user_messages:
        reference = " ".join(prior_user_messages[-3:])  # 取最近 3 条作为新颖性参照
        try:
            a3 = get_novelty(reference, user_query, baidu_client)
        except Exception as e:
            logger.warning(f"A3 新颖性计算失败,使用中性值 0.5: {e}")
            a3 = 0.5
    else:
        a3 = 0.5  # 首轮无参照,赋予中性隶属度

    c1 = len(prior_user_messages) + 1  # 当前轮也算在内
    c2 = C2_RAW_LOW if c1 == 1 else C2_RAW_HIGH

    logger.info(
        f"特征计算: A1={a1}, A2={a2:.4f}, A3={a3:.4f}, C1={c1}, C2={c2}"
    )
    return a1, a2, a3, c1, c2


def _membership_tag(v):
    """把 [0,1] 隶属度映射为高/中/低 + 颜色档。"""
    if v >= 0.67:
        return "高", "#2f855a"
    if v <= 0.33:
        return "低", "#c53030"
    return "中", "#b7791f"


def _strategy_row(v):
    """0/0.5/1 → (档位文字, 颜色)。"""
    return {0: ("低", "#c53030"), 0.5: ("中", "#b7791f"), 1: ("高", "#2f855a")}.get(v, ("?", "#4a5568"))


# 维度含义表(隶属度 → 自然语言标签)
_STATE_DIM_LABELS = {
    "A1": "提问复杂度",
    "A2": "情感倾向",
    "A3": "新颖性",
    "C1": "对话轮次",
    "C2": "先验知识",
}


def render_adaptive_bubble(bubble_text, strategy_meta, strategy, turn_index):
    """渲染统一的自适应感知卡片:顶部一行策略摘要,内嵌可展开的决策详情。

    整张卡只有一个 border / 一个阴影,信息层次为:
    - 顶行:🧠 自适应感知 · {摘要} · 第 N 轮
    - 折叠区:<details> 摘要下方,点击展开查看 4 段决策详情

    气泡摘要与展开后的策略向量由**同一份 strategy_meta + strategy 推导**,
    不会出现「气泡说短句、详情显示 syntax=1」的不一致情况。
    """
    calibrated = strategy_meta.get("calibrated_state", {}) or {}
    matched_rules = strategy_meta.get("matched_rules", []) or []
    matched_src = strategy_meta.get("matched_rule_source", "")
    final_src = strategy_meta.get("final_strategy_source", matched_src)
    risk = strategy_meta.get("is_risk_detected", False)

    # ---------- ① 5 维状态校准表 ----------
    state_rows = []
    for dim in ["A1", "A2", "A3", "C1", "C2"]:
        v = calibrated.get(dim)
        if v is None:
            continue
        tag, color = _membership_tag(v)
        state_rows.append(
            f"<tr><td style='padding:3px 10px 3px 0; color:#4a5568;'>"
            f"{_STATE_DIM_LABELS[dim]} <span style='color:#a0aec0;'>({dim})</span></td>"
            f"<td style='padding:3px 10px; font-family:monospace;'>{v:.3f}</td>"
            f"<td style='padding:3px 0; color:{color}; font-weight:600;'>{tag}</td></tr>"
        )
    state_table = (
        "<table style='border-collapse:collapse; font-size:0.9em;'>"
        + "".join(state_rows) + "</table>"
    )

    # ---------- ② 分层推理路径 ----------
    if matched_src == "R_full+":
        path_html = (
            "<div style='margin:3px 0;'>"
            "<span style='color:#2f855a;'>①</span> 匹配 <b>R_full+</b> 高收益组态 → "
            f"命中 {len(matched_rules)} 条</div>"
        )
    elif matched_src == "R_core+":
        path_html = (
            "<div style='margin:3px 0;'>"
            "<span style='color:#b7791f;'>①</span> R_full+ 无匹配 → 匹配 <b>R_core+</b> 核心条件 → "
            f"命中 {len(matched_rules)} 条</div>"
        )
    elif matched_src == "control_group":
        path_html = (
            "<div style='margin:3px 0; color:#718096;'>"
            "<span style='color:#718096;'>①</span> 对照组模式:跳过策略选择,使用固定中值</div>"
        )
    else:
        path_html = (
            "<div style='margin:3px 0;'>"
            "<span style='color:#c53030;'>①</span> R_full+ / R_core+ 均无匹配 → "
            "使用 <b>S_default</b></div>"
        )

    if matched_src in ("R_core+", "default"):
        risk_html = (
            "<span style='color:#2f855a;'>✓ 通过</span>"
            if not risk else
            "<span style='color:#c53030;'>⚠ 触发低收益风险规则,已切换至安全策略(汉明距离最小)</span>"
        )
        path_html += (
            "<div style='margin:3px 0;'>"
            "<span style='color:#4a5568;'>②</span> 风险检查 → " + risk_html + "</div>"
        )
        if risk and final_src == "safe_search":
            path_html += (
                "<div style='margin:3px 0;'>"
                "<span style='color:#4a5568;'>③</span> 在 27 个候选策略中按汉明距离选择最近的安全策略</div>"
            )

    # ---------- ③ 命中组态规则 ----------
    if matched_rules:
        rule_blocks = []
        for i, r in enumerate(matched_rules, 1):
            c = r.get("C", {})
            s = r.get("S", {})
            gamma = r.get("consistency", 0)
            cov = r.get("raw_coverage", 0)
            c_html = " · ".join(
                f"{k}={v}" for k, v in c.items() if v != 0.5
            ) or "(无约束)"
            s_html = " · ".join(f"{k}={v}" for k, v in s.items())
            rule_blocks.append(
                f"<div style='margin:5px 0; padding:5px 9px; background:#fff; "
                f"border-left:2px solid #6b8cff; border-radius:3px;'>"
                f"<div style='font-size:0.85em; color:#4a5568;'>"
                f"<b>规则 #{i}</b> · γ={gamma:.3f} · raw_cov={cov:.3f}</div>"
                f"<div style='font-size:0.78em; color:#718096; margin-top:1px;'>"
                f"条件: {c_html}</div>"
                f"<div style='font-size:0.78em; color:#718096;'>"
                f"策略: {s_html}</div></div>"
            )
        rules_html = "".join(rule_blocks)
    else:
        rules_html = (
            "<div style='color:#a0aec0; font-size:0.85em; font-style:italic;'>"
            "本轮无组态规则命中(对照组 / 默认回退)。</div>"
        )

    # ---------- ④ 本轮执行策略向量 ----------
    strat_rows = []
    strat_labels = [
        ("syntax", "句法复杂度", {0: "短句分点", 0.5: "通顺自然", 1: "正式专业"}),
        ("concept", "新概念密度", {0: "不引新概念", 0.5: "适度补背景", 1: "多引新概念"}),
        ("proactivity", "系统主动性", {0: "答完即停", 0.5: "视情况追问", 1: "主动追问"}),
    ]
    for key, name, meaning in strat_labels:
        v = strategy.get(key)
        tag, color = _strategy_row(v)
        m = meaning.get(v, "—")
        strat_rows.append(
            f"<tr><td style='padding:3px 10px 3px 0;'>{name}</td>"
            f"<td style='padding:3px 10px; font-family:monospace;'>{v}</td>"
            f"<td style='padding:3px 10px; color:{color}; font-weight:600;'>{tag}</td>"
            f"<td style='padding:3px 0; color:#718096;'>{m}</td></tr>"
        )
    strat_table = (
        "<table style='border-collapse:collapse; font-size:0.9em;'>"
        + "".join(strat_rows) + "</table>"
    )

    # ---------- ⑤ 任务类型判断 ----------
    task_key_seen = strategy_meta.get("task_key")
    task_reason_seen = strategy_meta.get("task_reason", "—")
    task_fallback = strategy_meta.get("task_fallback", False)
    task_label_seen = TASK_LABEL.get(task_key_seen, task_key_seen or "—")
    fallback_badge = (
        "<span style='color:#c53030; font-size:0.85em;'>⚠ LLM 判断失败,使用默认 task2</span>"
        if task_fallback else
        "<span style='color:#2f855a; font-size:0.85em;'>✓ LLM 正常判断</span>"
    )
    task_table = (
        "<table style='border-collapse:collapse; font-size:0.9em;'>"
        f"<tr><td style='padding:3px 10px 3px 0;'>任务类型</td>"
        f"<td style='padding:3px 10px; font-weight:600; color:#6b8cff;'>{task_label_seen}</td>"
        f"<td style='padding:3px 0; color:#718096; font-family:monospace;'>({task_key_seen or '—'})</td></tr>"
        f"<tr><td style='padding:3px 10px 3px 0;'>判断依据</td>"
        f"<td colspan='2' style='padding:3px 0; color:#4a5568;'>{task_reason_seen}</td></tr>"
        f"<tr><td style='padding:3px 10px 3px 0;'>调用状态</td>"
        f"<td colspan='2' style='padding:3px 0;'>{fallback_badge}</td></tr>"
        "</table>"
    )

    # ---------- 单卡 HTML ----------
    html = f"""
<div style="
    background: linear-gradient(90deg, #f5f7ff 0%, #f9fafe 100%);
    border-left: 3px solid #6b8cff;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 4px 0 6px 0;
    font-size: 0.85em;
    color: #4a5568;
    max-width: 92%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
">
  <div>
    <span style="color:#6b8cff; font-weight:600;">🧠 自适应感知</span>
    <span style="color:#cbd5e0; margin:0 6px;">·</span>
    <span style="color:#2d3748;">{bubble_text}</span>
    <span style="color:#a0aec0; font-size:0.9em; margin-left:8px;">第 {turn_index} 轮</span>
  </div>
  <details style="margin-top:6px;">
    <summary style="
        cursor: pointer;
        color: #6b8cff;
        font-size: 0.9em;
        padding: 2px 0;
    ">决策详情</summary>
    <div style="
        margin-top: 8px;
        padding: 10px 0 2px 0;
        border-top: 1px dashed #cbd5e0;
        color: #2d3748;
    ">
      <div style="color:#6b8cff; font-weight:600; margin-bottom:4px; font-size:0.95em;">① 5 维状态校准</div>
      {state_table}
      <div style="color:#6b8cff; font-weight:600; margin:10px 0 4px; font-size:0.95em;">② 分层推理路径</div>
      {path_html}
      <div style="color:#6b8cff; font-weight:600; margin:10px 0 4px; font-size:0.95em;">③ 命中组态规则</div>
      {rules_html}
      <div style="color:#6b8cff; font-weight:600; margin:10px 0 4px; font-size:0.95em;">④ 本轮执行策略向量</div>
      {strat_table}
      <div style="color:#6b8cff; font-weight:600; margin:10px 0 4px; font-size:0.95em;">⑤ 任务类型判断</div>
      {task_table}
    </div>
  </details>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# 2. 前端界面
# ==============================================================================

st.set_page_config(page_title="自适应对话", page_icon="✨", layout="wide")

# --- Session State 初始化 ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "pending_rerun" not in st.session_state:
    st.session_state.pending_rerun = False

# --- 侧边栏 ---
with st.sidebar:
    st.title("会话")

    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {
            "title": f"对话 {len(st.session_state.conversations) + 1}",
            "messages": [],
            "last_modified": time.time(),
        }
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()

    chat_ids = sorted(
        st.session_state.conversations.keys(),
        key=lambda k: st.session_state.conversations[k].get("last_modified", 0),
        reverse=True,
    )

    if chat_ids:
        if st.session_state.current_chat_id not in chat_ids:
            st.session_state.current_chat_id = chat_ids[0]

        display_options = {cid: data["title"] for cid, data in st.session_state.conversations.items()}

        selected_id = st.radio(
            "切换会话:",
            options=chat_ids,
            format_func=lambda x: display_options[x],
            key="chat_sel",
            index=chat_ids.index(st.session_state.current_chat_id),
        )

        if selected_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_id
            st.rerun()
    else:
        st.info("暂无会话,点击上方按钮新建。")

# --- 主界面 ---
st.title("✨ 自适应对话系统")
st.caption("每一轮回复前,系统会先告诉你它识别到的任务类型、感知到的状态、打算怎么答。")

current_id = st.session_state.current_chat_id

if not current_id or current_id not in st.session_state.conversations:
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <h3>👋 你好</h3>
            <p>请点击左侧侧边栏的 <b>"➕ 新建对话"</b> 开始。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

current_chat = st.session_state.conversations[current_id]

# --- 渲染历史消息(含之前轮次的自适应感知卡片) ---
for msg in current_chat["messages"]:
    if msg.get("strategy_meta") and msg.get("strategy") and msg.get("adaptive_bubble"):
        render_adaptive_bubble(
            bubble_text=msg["adaptive_bubble"],
            strategy_meta=msg["strategy_meta"],
            strategy=msg["strategy"],
            turn_index=msg.get("turn_index", 1),
        )
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 用户输入 ---
prompt = st.chat_input("输入你的问题,回车发送...")

if prompt:
    # 1. 展示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    current_chat["messages"].append({"role": "user", "content": prompt})

    # 2. 提取上下文(本轮之前的用户提问)用于任务识别
    prior_user_messages = [
        m["content"] for m in current_chat["messages"][:-1] if m.get("role") == "user"
    ]
    recent_context = prior_user_messages[-2:]

    # 3. LLM 识别本轮任务类型(事实型 / 解释型 / 探索型)
    with st.spinner("正在识别任务类型..."):
        task_key, task_reason, used_fallback = classify_task_type(
            user_query=prompt,
            recent_context=recent_context,
            client=client,
        )

    # 4. 计算用户状态 + 选择策略(用识别出来的 task_key 选对应规则库)
    a1, a2, a3, c1, c2 = compute_user_state(prompt, prior_user_messages)

    with st.spinner("正在分析提问状态..."):
        engine = ProactiveScaffoldingSystem(task_key=task_key)
        mode_settings, strategy_meta = engine.decide_interaction_strategy(
            a1=a1, a2=a2, a3=a3, c1=c1, c2=c2
        )
    # 把任务识别结果挂到 strategy_meta 上,UI 渲染时统一从这里取
    strategy_meta["task_key"] = task_key
    strategy_meta["task_reason"] = task_reason
    strategy_meta["task_fallback"] = used_fallback

    # 3. 构造自适应感知卡片(确定性:从 strategy_meta + strategy 推导,
    #    气泡摘要与展开后的详情面板**同源**,保证内容一致)
    bubble_text = build_bubble_text(strategy_meta, mode_settings)
    render_adaptive_bubble(
        bubble_text=bubble_text,
        strategy_meta=strategy_meta,
        strategy=mode_settings,
        turn_index=int(c1),
    )

    # 4. 流式生成主回复
    full_reasoning = ""
    full_content = ""
    usage_info = {}

    with st.chat_message("assistant"):
        with st.status("系统推理中...", expanded=True) as status_box:
            reasoning_placeholder = st.empty()

        content_placeholder = st.empty()

        for chunk in smart_chat_stream(current_chat["messages"], mode_settings, client):
            if chunk["type"] == "reasoning":
                full_reasoning += chunk["text"]
                reasoning_placeholder.markdown(
                    f"> {full_reasoning.replace(chr(10), chr(10) + '> ')}"
                )
            elif chunk["type"] == "content":
                if not full_content:
                    status_box.update(label="✅ 推理完成", state="complete", expanded=False)
                full_content += chunk["text"]
                content_placeholder.markdown(full_content + "▌")
            elif chunk["type"] == "done":
                usage_info = chunk.get("usage", {})

        content_placeholder.markdown(full_content)

    # 5. 写回消息历史(同时把气泡、策略元数据存入)
    current_chat["messages"].append({
        "role": "assistant",
        "content": full_content,
        "reasoning": full_reasoning,
        "adaptive_bubble": bubble_text,
        "strategy": mode_settings,
        "strategy_meta": strategy_meta,
        "turn_index": int(c1),
    })
    current_chat["last_modified"] = time.time()

    # 6. 若这是该对话的第 1 轮助手回复,更新标题为提问前 10 字
    user_msg_count = sum(1 for m in current_chat["messages"] if m.get("role") == "user")
    if user_msg_count == 1:
        current_chat["title"] = prompt[:10] + ("..." if len(prompt) > 10 else "")

    st.rerun()
