"""对话生成与自适应元解释模块。"""

import json
from logger import logger

# 句法复杂度 (System Syntax) 的指令库
SYNTAX_PROMPTS = {
    1: "在回答中可以使用较为正式和专业的表达方式,在保证论证过程清晰、结构明确的基础上,允许采用多从句结构与逻辑递进的论述形式。",
    0.5: "使用通顺、自然且结构清晰的语言进行解释,避免过度口语化或高度学术化表达,优先采用易于理解的完整句式。",
    0: "使用短句和明确的结构(如分点或步骤说明)进行回答,避免复杂句法结构和嵌套从句,确保文本高度可读。",
}

# 新异概念密度 (Concept Density) 的指令库
CONCEPT_PROMPTS = {
    1: "在回答中可以引入新的概念、理论或视角,对相关背景进行扩展性说明,但需确保新概念之间具有清晰关联,避免无关发散。",
    0.5: "聚焦于问题核心,在必要时引入少量关键新的概念或背景信息以支持理解,控制信息增量,保持整体信息密度适中。",
    0: "仅围绕问题所涉及的既有概念进行回答,不引入新的专业术语或理论框架,直接给出解决当前问题所需的信息。",
}

# 系统主动性 (Proactivity) 的指令库
PROACTIVITY_PROMPTS = {
    1: "在完成回答后,主动提出一个有针对性的澄清性问题、反思性提问或下一步探索建议,引导用户进一步思考或补充信息。",
    0.5: "视回答完整程度决定是否提出简要的后续问题或建议;若问题已被充分解决,可不额外引导。",
    0: "在完成回答后直接结束,不提出反问或额外建议,避免对用户后续行动产生干预。",
}

# 策略档位 → 气泡用短描述(供 build_bubble_text 使用)
_SYNTAX_SHORT = {0: "短句分点", 0.5: "通顺自然", 1: "正式专业"}
_CONCEPT_SHORT = {0: "不引新概念", 0.5: "适度补背景", 1: "多引新概念"}
_PROACTIVITY_SHORT = {0: "答完即停", 0.5: "视情况追问", 1: "主动追问或给建议"}

# 决策来源 → 短标签
_SOURCE_SHORT = {
    "R_full+": "高收益组态命中",
    "R_core+": "核心条件推断",
    "safe_search": "风险规避回退",
    "default": "默认策略",
    "control_group": "对照组固定值",
}

# 任务类型 → 中文短标签(论文中三类任务情境的对外称呼)
TASK_LABEL = {
    "task1": "事实型",
    "task2": "解释型",
    "task3": "探索型",
}


def build_bubble_text(strategy_meta, strategy):
    """从 strategy_meta + strategy 确定性构造气泡摘要文本。

    与下方展开的决策详情**同源**——由相同的 strategy / final_strategy_source
    推导,保证气泡描述与详情面板内容一致(不会出现气泡说「短句分点」但
    详情显示 syntax=1 的情况)。
    """
    final_src = strategy_meta.get("final_strategy_source", "default")
    s = strategy.get("syntax", 0.5)
    c = strategy.get("concept", 0.5)
    p = strategy.get("proactivity", 0.5)
    parts = [_SYNTAX_SHORT.get(s, "—"),
             _CONCEPT_SHORT.get(c, "—"),
             _PROACTIVITY_SHORT.get(p, "—")]
    source = _SOURCE_SHORT.get(final_src, final_src)

    segments = [" · ".join(parts)]
    task_key = strategy_meta.get("task_key")
    if task_key in TASK_LABEL:
        segments.append(f"任务:{TASK_LABEL[task_key]}")
    if source:
        segments.append(f"来源:{source}")
    return "  ·  ".join(segments)


# 任务类型识别 LLM 的 system prompt
_TASK_CLASSIFIER_PROMPT = """你是自适应对话系统的「任务情境识别」模块。系统的策略空间按任务类型分组(事实型 / 解释型 / 探索型,各自的组态规则和默认策略都不同),所以每一轮需要先判断本轮提问属于哪一类。

【三类任务定义】
1. 事实型 (task1) — 用户在做**事实查找或数据获取**
   - 典型问法: "X 是什么?/X 多少?/X 在哪?/X 什么时候?"
   - 期望系统行为: 直接、完整地给出信息,不绕弯
   - 失败模式: 信息不足型失败(系统没充分供给事实)

2. 解释型 (task2) — 用户想**理解某事的原因、机制、原理、关系**
   - 典型问法: "为什么...?/怎么...的?/X 的原理是什么?/X 和 Y 的区别?"
   - 期望系统行为: 适度展开,引导深入思考,可以追问澄清
   - 失败模式: 认知惰性型失败(用户问得浅,系统也没引导深层加工)

3. 探索型 (task3) — 用户在做**开放式判断、决策支持、或综合多源信息**
   - 典型问法: "我应该...怎么办?/如果...会怎样?/帮我分析.../对比 X 和 Y 哪个更适合我"
   - 期望系统行为: 整合多源信息,降低复杂度,辅助决策,**回答要简洁不能太长**
   - 失败模式: 信息过载型失败(系统输出太复杂反而帮倒忙)

【判断要点】
- 看**当前提问的认知目标**,不是话题领域
- 短问句(几个字)也要给出最匹配的判断
- 同一对话内,允许从一类切到另一类(用户可能从"巴黎人口多少"转到"如果要去巴黎工作要考虑什么")

【输出格式(严格)】
- 只输出一个 JSON,不要其他任何文字、解释、markdown
- 格式: {"task": "task1|task2|task3", "reason": "中文短句,不超过 12 字"}
- reason 要能让用户看懂为什么这样分,例如: "询问具体数据"、"询问原理机制"、"开放式决策咨询"
"""


def classify_task_type(user_query, recent_context, client):
    """调用 deepseek-chat 判断当前提问属于哪一类任务情境。

    Args:
        user_query: str, 当前用户提问
        recent_context: list[str], 最近 2 轮用户提问(用于判断话题延续性,可为空)
        client: OpenAI/DeepSeek 兼容客户端

    Returns:
        (task_key, reason, used_fallback)
        task_key ∈ {"task1", "task2", "task3"}
        reason: 中文短句(<=12 字)
        used_fallback: True 表示 LLM 调用失败,回退到默认;False 表示正常判断
    """
    if recent_context:
        ctx_lines = "\n".join(f"  - {m}" for m in recent_context[-2:])
    else:
        ctx_lines = "  (无,这是对话首轮)"

    user_prompt = (
        "【最近 2 轮用户提问(用于判断话题延续性)】\n"
        f"{ctx_lines}\n\n"
        "【本轮用户提问】\n"
        f"  \"{user_query}\"\n\n"
        "请只输出 JSON,不要任何其他文字。"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _TASK_CLASSIFIER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=80,
            temperature=0.1,
        )
        content = (response.choices[0].message.content or "").strip()

        # 尝试从内容里抠 JSON(防止模型包了 markdown 围栏或前缀)
        if "{" in content and "}" in content:
            content = content[content.find("{"): content.rfind("}") + 1]
        data = json.loads(content)
        task = str(data.get("task", "")).strip()
        reason = str(data.get("reason", "")).strip()[:12]

        if task not in TASK_LABEL:
            raise ValueError(f"返回了非预期 task 值: {task!r}")

        logger.info(f"🎯 [任务识别] → {task}({TASK_LABEL[task]}):{reason}")
        return task, (reason or "未提供"), False

    except Exception as e:
        logger.error(f"任务类型识别失败,回退到 task2: {e}")
        return "task2", "默认(识别失败)", True


def smart_chat_stream(chat_messages, mode_settings, client):
    """流式调用 deepseek-reasoner 生成主回复,逐块 yield 事件。

    yield 字典格式:
        {"type": "reasoning", "text": "..."}  推理过程片段
        {"type": "content",   "text": "..."}  正式回复片段
        {"type": "done", "usage": {...}}      流式结束
    """

    system_prompt_content = (
        "在回答问题时,请严格遵守以下回复规范:\n"
        f"1. [语言风格]: {SYNTAX_PROMPTS[mode_settings['syntax']]}\n"
        f"2. [内容深度]: {CONCEPT_PROMPTS[mode_settings['concept']]}\n"
        f"3. [交互策略]: {PROACTIVITY_PROMPTS[mode_settings['proactivity']]}\n"
        "[全局约束] 直接输出最终回复,不要在回复中提及或解释你正在遵守的任何规则,也不要暗示自己在按照特定设定行事。"
    )

    logger.debug(f"System Prompt 配置: {mode_settings}")

    messages = [{"role": "system", "content": system_prompt_content}] + chat_messages

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True,
        )

        usage_info = {}

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            reasoning_chunk = getattr(delta, "reasoning_content", None)
            content_chunk = getattr(delta, "content", None)

            if reasoning_chunk:
                yield {"type": "reasoning", "text": reasoning_chunk}
            if content_chunk:
                yield {"type": "content", "text": content_chunk}

            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_info = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                }

        if usage_info:
            logger.info(
                f"Token 消耗: Prompt={usage_info['prompt_tokens']}, "
                f"Completion={usage_info['completion_tokens']}"
            )

        yield {"type": "done", "usage": usage_info}

    except Exception as e:
        logger.error(f"DeepSeek API 调用异常: {e}")
        yield {"type": "content", "text": f"Error: {e}"}
        yield {"type": "done", "usage": {}}
