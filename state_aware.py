from ltp import StnSplit
import collections
import json
import re
import os
from logger import logger

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 system_prompts.json 的绝对路径
json_path = os.path.join(current_dir, "system_prompts.json")

with open(json_path, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPTS = json.load(f)

def calculate_tree_depth(tree, node):
    """递归计算树的深度"""
    if node not in tree:
        return 1  # 叶子节点，深度包含自身记为 1
    
    # 递归寻找所有子节点的最大深度
    max_child_depth = 0
    for child in tree[node]:
        max_child_depth = max(max_child_depth, calculate_tree_depth(tree, child))
        
    return max_child_depth + 1

### 计算变量A1
def get_max_dependency_depth(text, ltp_model):
    sents = StnSplit().split(text)
    result = ltp_model.pipeline(sents, tasks=["cws", "dep"])
    dependencies = result.dep
    max_depth = []
    for sent_idx, dependency in enumerate(dependencies):
        # sent_deps 格式通常为: [(head_index, label), ...] 
        # 注意：LTP 的索引是 1-based，0 表示虚拟根节点 ROOT
        
        # 构建邻接表 (Tree Structure)
        tree = collections.defaultdict(list)
        root_node = None
        
        heads = dependency["head"]
        labels = dependency["label"]
        sent_deps = [(heads[i], labels[i]) for i in range(len(heads))]
        
        for i, (head, label) in enumerate(sent_deps):
            curr_word_idx = i + 1  # 当前词的索引 (1-based)
            
            if head == 0:
                root_node = curr_word_idx # 找到根节点
            else:
                tree[head].append(curr_word_idx) # 父节点 -> 子节点列表
        
        # 3. 如果解析失败没有根节点，深度为0
        if root_node is None:
            max_depth.append(0)

        # 4. 使用 DFS 计算最大深度
        max_depth.append(calculate_tree_depth(tree, root_node))
    A1 = max(max_depth)
    return A1

### 计算变量A2
def get_sentiment_analysis(text, client):
    d = client.sentimentClassify(text)
    A2 = d["items"][0]["positive_prob"]
    return A2

### 计算变量A3
def get_novelty(task, text, client):
    r = client.simnet(task, text)
    A3 = 1 - r["score"]
    return A3

def clean_json_string(json_str):
    """清洗返回的字符串，去除 Markdown 标记"""
    if not json_str:
        return None
    # 去除 ```json 和 ```
    clean_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
    clean_str = re.sub(r'^```\s*', '', clean_str, flags=re.MULTILINE)
    clean_str = re.sub(r'\s*```$', '', clean_str, flags=re.MULTILINE)
    return clean_str.strip()

### 计算先验知识
""" 
    注意：为了节省时间和避免程序错误，思维导图过于复杂时直接截断，因此先验知识评分只是定性判断方便决策，不可作为真正的评分数据。
"""

def get_prior_knowledge(task_key: str, mind_map_text: str, client) -> float:
    """
    先验知识评分（快速版）。

    与原版 get_prior_knowledge() 接口完全一致：
      - 输入：task_key ("task1"/"task2"/"task3")、思维导图文本、DeepSeek client
      - 输出：float，知识单元计数分值（与原 total_score 同量纲）

    优化点：
      - System prompt 只要求模型输出 {"total_score": N}，无 knowledge_breakdown
      - max_tokens=20，模型仅需生成约 5 个 token，延迟从数十秒降至 1-2 秒
      - 截断阈值从 1000 压缩至 500 字符（对 θ_c 阈值判断已足够）
    """
    # 超长截断警告（阈值已从 1000 收紧至 500）
    TRUNCATE_LIMIT = 500
    if len(mind_map_text) >= TRUNCATE_LIMIT:
        logger.warning("⚠️ 思维导图过长，截断后再评分...")

    SYSTEM_PROMPT = SYSTEM_PROMPTS[task_key]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请对以下思维导图文本进行评分：\n\n{mind_map_text[:TRUNCATE_LIMIT]}"}
        ],
        max_tokens=20,          # 只需输出 {"total_score": N}，约 5 token
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw_response = response.choices[0].message.content

    if raw_response:
        try:
            cleaned = clean_json_string(raw_response)
            json_data = json.loads(cleaned)
            score = float(json_data.get("total_score", 0))
            logger.info(f"先验知识计算成功! 得分: {score}")
            return score
        except (json.JSONDecodeError, ValueError):
            logger.error("❌ JSON 解析失败! (模型输出格式错误)")
            return 0.0
    else:
        logger.error("❌ API 请求无响应")
        return 0.0
    
# def get_prior_knowledge(task_key, mind_map_text, client):
#     SYSTEM_PROMPT = SYSTEM_PROMPTS[task_key]
#     response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": f"请对以下思维导图文本进行评分：\n\n{mind_map_text}"}
#             ],
#             temperature=0,
#             response_format={ "type": "json_object" } # 强制 JSON 模式
#         )
#     raw_response = response.choices[0].message.content

#     if raw_response:
#             # 解析 JSON
#             try:
#                 cleaned_json_str = clean_json_string(raw_response)
#                 json_data = json.loads(cleaned_json_str)
#                 score = json_data.get('total_score', 0)
#                 logger.info(f" 先验知识计算成功! 得分: {score}")
#                 return score
#             except json.JSONDecodeError:
#                 logger.error("❌ JSON 解析失败! (模型输出格式错误)")
#     else:
#         logger.error("❌ API 请求无响应")