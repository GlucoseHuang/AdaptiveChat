"""实时用户状态感知模块。

在新版自由对话系统里,负责计算以下三个随用户提问实时变化的特征:
- A1 句法依赖深度(基于哈工大 LTP 依存句法分析)
- A2 情感倾向(基于百度 NLP 的 sentimentClassify)
- A3 问题新颖性(基于百度 NLP 的 SimNet,参考文本为历史对话)

先验知识 C2 与对话轮次 C1 不再由本模块计算,改由 app.py 直接根据
对话状态推导(首轮 C2 视为低,后续轮次视为高),以替代原系统中
基于思维导图的评分流程。
"""

from ltp import StnSplit
import collections
from logger import logger


def calculate_tree_depth(tree, node):
    """递归计算树的深度。"""
    if node not in tree:
        return 1

    max_child_depth = 0
    for child in tree[node]:
        max_child_depth = max(max_child_depth, calculate_tree_depth(tree, child))

    return max_child_depth + 1


def get_max_dependency_depth(text, ltp_model):
    """计算句子的最大依存句法深度 A1。"""
    sents = StnSplit().split(text)
    result = ltp_model.pipeline(sents, tasks=["cws", "dep"])
    dependencies = result.dep

    max_depth = []
    for dependency in dependencies:
        tree = collections.defaultdict(list)
        root_node = None

        heads = dependency["head"]
        labels = dependency["label"]

        for i, (head, _label) in enumerate(zip(heads, labels)):
            curr_word_idx = i + 1
            if head == 0:
                root_node = curr_word_idx
            else:
                tree[head].append(curr_word_idx)

        if root_node is None:
            max_depth.append(0)
            continue

        max_depth.append(calculate_tree_depth(tree, root_node))

    if not max_depth:
        return 0
    return max(max_depth)


def get_sentiment_analysis(text, client):
    """调用百度 NLP sentimentClassify 获取正向情感概率 A2。"""
    d = client.sentimentClassify(text)
    return d["items"][0]["positive_prob"]


def get_novelty(reference, text, client):
    """计算问题新颖性 A3 = 1 - SimNet(reference, text).score。

    reference 一般是历史对话拼接文本,text 是当前用户提问。
    """
    r = client.simnet(reference, text)
    return 1 - r["score"]
