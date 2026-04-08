from logger import logger

# 句法复杂度 (System Syntax) 的指令库
SYNTAX_PROMPTS = {
    1: "在回答中可以使用较为正式和专业的表达方式，在保证论证过程清晰、结构明确的基础上，允许采用多从句结构与逻辑递进的论述形式。",
    0.5: "使用通顺、自然且结构清晰的语言进行解释，避免过度口语化或高度学术化表达，优先采用易于理解的完整句式。",
    0: "使用短句和明确的结构（如分点或步骤说明）进行回答，避免复杂句法结构和嵌套从句，确保文本高度可读。"
}

# 新异概念密度 (Concept Density) 的指令库
CONCEPT_PROMPTS = {
    1: "在回答中可以引入新的概念、理论或视角，对相关背景进行扩展性说明，但需确保新概念之间具有清晰关联，避免无关发散。",
    0.5: "聚焦于问题核心，在必要时引入少量关键新的概念或背景信息以支持理解，控制信息增量，保持整体信息密度适中。",
    0: "仅围绕问题所涉及的既有概念进行回答，不引入新的专业术语或理论框架，直接给出解决当前问题所需的信息。"
}

# 系统主动性 (Proactivity) 的指令库
PROACTIVITY_PROMPTS = {
    1: "在完成回答后，主动提出一个有针对性的澄清性问题、反思性提问或下一步探索建议，引导用户进一步思考或补充信息。",
    0.5: "视回答完整程度决定是否提出简要的后续问题或建议；若问题已被充分解决，可不额外引导。",
    0: "在完成回答后直接结束，不提出反问或额外建议，避免对用户后续行动产生干预。"
}

def smart_chat_stream(chat_messages, mode_settings, client):
    """
    流式版本：以生成器形式逐块 yield 内容。

    每次 yield 一个 dict，格式为：
        {"type": "reasoning", "text": "..."} —— 推理过程片段
        {"type": "content",   "text": "..."} —— 正式回复片段
        {"type": "done", "usage": {...}}      —— 流式结束，携带 token 用量

    参数:
    - chat_messages: 对话列表 (list of dicts)
    - mode_settings: 模式的具体设置 (dict)
    - client: OpenAI/DeepSeek 客户端实例
    """

    # 1. 动态构建 System Prompt
    system_prompt_content = (
        f"在回答问题时，请严格遵守以下回复规范：\n"
        f"1. [语言风格]: {SYNTAX_PROMPTS[mode_settings['syntax']]}\n"
        f"2. [内容深度]: {CONCEPT_PROMPTS[mode_settings['concept']]}\n"
        f"3. [交互策略]: {PROACTIVITY_PROMPTS[mode_settings['proactivity']]}\n"
        f"[全局约束] 直接输出最终回复，不要在回复中提及或解释你正在遵守的任何规则，也不要暗示自己在按照特定设定行事。"
    )

    logger.debug(f"📝 System Prompt 配置: {mode_settings}")

    # 2. 构建 API 请求消息列表
    messages = [{"role": "system", "content": system_prompt_content}] + chat_messages

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True
        )

        usage_info = {}

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None

            if delta is None:
                continue

            reasoning_chunk = getattr(delta, 'reasoning_content', None)
            content_chunk = getattr(delta, 'content', None)

            if reasoning_chunk:
                yield {"type": "reasoning", "text": reasoning_chunk}

            if content_chunk:
                yield {"type": "content", "text": content_chunk}

            if hasattr(chunk, 'usage') and chunk.usage is not None:
                usage_info = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens
                }

        if usage_info:
            logger.info(f"💰 Token 消耗: Prompt={usage_info['prompt_tokens']}, Completion={usage_info['completion_tokens']}")

        yield {"type": "done", "usage": usage_info}

    except Exception as e:
        logger.error(f"❌ DeepSeek API 调用异常: {str(e)}")
        yield {"type": "content", "text": f"Error: {str(e)}"}
        yield {"type": "done", "usage": {}}