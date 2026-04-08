import math
from itertools import product
from logger import logger


class ProactiveScaffoldingSystem:
    """
    人智交互自适应支持系统 — 交互策略选择模块
    实现论文 5.2.2「交互策略选择的形式化建模」中定义的映射 f: U -> S
    """

    # ============================================================
    # 策略空间 S = {0, 0.5, 1}^3，三个策略维度各取三值
    # 对应论文：S_i ∈ S = {0, 0.5, 1}^3
    # 执行时含义：0=低, 0.5=中, 1=高
    # ============================================================
    STRATEGY_VALUES = [0, 0.5, 1]
    STRATEGY_DIMS = ["syntax", "concept", "proactivity"]

    # 对照组固定策略：三个维度全部设置为0.5
    CONTROL_GROUP_STRATEGY = {"syntax": 0.5, "concept": 0.5, "proactivity": 0.5}

    def __init__(self, task_key: str, priority_metric: str = "consistency",
                 control_group: bool = False):
        self.task_key = task_key
        self.metric = priority_metric  # 'consistency' 或 'raw_coverage'
        self.control_group = control_group  # True=对照组，False=实验组

        # ============================================================
        # 0. 模糊集直接校准锚点
        #    θ_f (95%锚点), θ_c (50%交叉点), θ_n (5%锚点)
        #    论文 (2)：基于 log-odds 的直接校准方法（Ragin, 2008）
        #    c_{t,j} 将每个状态维度映射到 [0,1] 模糊集隶属度
        # ============================================================
        # 格式：每个维度 -> (θ_n, θ_c, θ_f)，即 (5%, 50%, 95%) 锚点
        self.calibration_anchors = {
            "task1": {
                "A1": (3.000,6.000,8.000),
                "A2": (0.035,0.775,0.930),
                "A3": (0.329,0.450,0.839),
                # 对于对话轮数，根据第四章数据，认为第一轮为前期，之后轮次均为后期
                "C1": (0.0, 1.5, 2.0), 
                "C2": (0.0, 1.0, 61.5),
            },
            "task2": {
                "A1": (3.550,8.000,9.450),
                "A2": (0.463,0.989,0.999),
                "A3": (0.079,0.437,0.771),
                "C1": (0.0, 1.5, 2.0),
                "C2": (0.0, 7.0, 49.950),
            },
            "task3": {
                "A1": (5.000,6.000,8.000),
                "A2": (0.047,0.496,0.941),
                "A3": (0.000,0.352,0.845),
                "C1": (0.0, 1.5, 2.0),
                "C2": (0.000,5.000,20.000),
            },
        }

        # ============================================================
        # 1. 默认策略 S_default
        #    论文：当 R_core+ 也无匹配、或 S_safe=∅ 时使用
        # ============================================================
        self.default_strategy = {
        "task1" : {"syntax": 0.5, "concept": 1, "proactivity": 0},
        "task2" : {"syntax": 0.5, "concept": 0.5, "proactivity": 1},
        "task3" : {"syntax": 0, "concept": 0.5, "proactivity": 0.5}
        }
        # ============================================================
        # 2. 规则集合，按论文定义的三类子集：
        #    R_full+  : 高知识增益的完整组态（论文中的"中间解"）
        #    R_core+  : 高知识增益组态的核心条件（论文中的"简单解/简约解"）
        #    R^-      : 低知识增益的组态（风险规则）
        #
        # 每条规则 R_i = (C_i, S_i, γ_i)，其中：
        #   C_i: dict，键为状态维度，值为 {0, 0.5, 1}
        #        0   = 该条件在组态中被判定为不存在
        #        0.5 = 中间状态（不决定性因素，match 中永远一致）
        #        1   = 该条件在组态中被判定为存在
        #        注：缺失的键视同 0.5（不做约束）
        #   S_i: dict，键为策略维度，值为 {0, 0.5, 1}
        #        0 = 低, 0.5 = 中, 1 = 高
        #   γ_i: float，fsQCA 一致性指标，用于排序
        # ============================================================

        # ------- R_full+ ：高知识增益完整组态 -------
        self.R_full_pos = {
            "task1": [
                {"C": {"A1": 1, "A2": 1, "A3": 1, "C1": 0, "C2": 1},
                 "S": {"syntax": 0.5, "concept": 1, "proactivity": 0},
                 "consistency": 0.961955, "raw_coverage": 0.157627},
                {"C": {"A2": 1, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 1, "concept": 0, "proactivity": 0},
                 "consistency": 0.896463, "raw_coverage": 0.172015},
                {"C": {"A1": 1, "A2": 0, "A3": 0, "C1": 0, "C2": 0},
                 "S": {"syntax": 0, "concept": 1, "proactivity": 0},
                 "consistency": 0.880625, "raw_coverage": 0.237284},
                {"C": {"A1": 1, "A2": 1, "A3": 0, "C1": 0, "C2": 0},
                 "S": {"syntax": 1, "concept": 0, "proactivity": 0},
                 "consistency": 0.886747, "raw_coverage": 0.218447},
                {"C": {"A1": 0, "A2": 1, "A3": 1, "C1": 0, "C2": 0},
                 "S": {"syntax": 1, "concept": 1, "proactivity": 0},
                 "consistency": 0.884634, "raw_coverage": 0.233322},
                {"C": {"A1": 1, "A2": 0, "A3": 1, "C1": 0, "C2": 0},
                 "S": {"syntax": 1, "concept": 1, "proactivity": 1},
                 "consistency": 0.936332, "raw_coverage": 0.043299},
            ],
            "task2": [
                {"C": {"A2": 0, "A3": 0, "C1": 0, "C2": 0},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 1},
                 "consistency": 0.834868, "raw_coverage": 0.134697},
                {"C": {"A1": 1, "A2": 0, "A3": 1, "C1": 0, "C2": 0},
                 "S": {"syntax": 1, "concept": 1, "proactivity": 0},
                 "consistency": 0.921155, "raw_coverage": 0.098653},
                {"C": {"A1": 0, "A2": 1, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 0},
                 "consistency": 0.820152, "raw_coverage": 0.106983},
                {"C": {"A1": 1, "A2": 1, "A3": 1, "C1": 0, "C2": 0},
                 "S": {"syntax": 0, "concept": 1, "proactivity": 1},
                 "consistency": 0.898135, "raw_coverage": 0.137428},
                {"C": {"A1": 1, "A2": 1, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 1, "concept": 0, "proactivity": 1},
                 "consistency": 0.921002, "raw_coverage": 0.092663},
            ],
            "task3": [
                {"C": {"A1": 1, "A2": 1, "A3": 1, "C1": 0, "C2": 0},
                 "S": {"syntax": 0, "concept": 0.5, "proactivity": 1},
                 "consistency": 0.829733, "raw_coverage": 0.176986},
                {"C": {"A1": 0, "A2": 0, "A3": 0, "C1": 0, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 0},
                 "consistency": 0.801543, "raw_coverage": 0.105087},
                {"C": {"A1": 1, "A2": 0, "A3": 1, "C1": 0, "C2": 1},
                 "S": {"syntax": 0, "concept": 1, "proactivity": 0},
                 "consistency": 0.916864, "raw_coverage": 0.099190},
            ],
        }

        # ------- R_core+ ：高知识增益核心条件（简约解）-------
        self.R_core_pos = {
            "task1": [
                {"C": {"A2": 1},
                 "S": {"syntax": 1, "concept": 0.5, "proactivity": 0.5},
                 "consistency": 0.843024, "raw_coverage": 0.532213},
                {"C": {"A2": 0},
                 "S": {"syntax": 0.5, "concept": 1, "proactivity": 0.5},
                 "consistency": 0.769609, "raw_coverage": 0.479851},
                {"C": {"A3": 1},
                 "S": {"syntax": 0.5, "concept": 1, "proactivity": 0.5},
                 "consistency": 0.771166, "raw_coverage": 0.506202},
            ],
            "task2": [
                {"C": {"A1": 1},
                 "S": {"syntax": 0.5, "concept": 0.5, "proactivity": 0.5},
                 "consistency": 0.725852, "raw_coverage": 0.679304},
                {"C": {"A2": 1},
                 "S": {"syntax": 0, "concept": 0.5, "proactivity": 0.5},
                 "consistency": 0.750530, "raw_coverage": 0.606061},
                {"C": {"C2": 0},
                 "S": {"syntax": 0.5, "concept": 0, "proactivity": 1},
                 "consistency": 0.670347, "raw_coverage": 0.288186},
            ],
            "task3": [
                {"C": {"C1": 0},
                 "S": {"syntax": 0, "concept": 0.5, "proactivity": 0.5},
                 "consistency": 0.721605, "raw_coverage": 0.537805},
            ],
        }

        # ------- R^- ：低知识增益风险规则 -------
        # 结构同上，C 为用户状态条件，S 为对应低效策略配置
        self.R_neg = {
            "task1": [
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 0, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 0}},
                {"C": {"A1": 1, "A2": 0, "A3": 0, "C1": 0, "C2": 1},
                 "S": {"syntax": 1, "concept": 0, "proactivity": 1}},
                {"C": {"A1": 1, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 0}},
                {"C": {"A1": 0, "A2": 1, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 0}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 1, "concept": 0, "proactivity": 0}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 1}},
            ],
            "task2": [
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0.5, "concept": 0, "proactivity": 0.5}},
                {"C": {"A1": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 1, "concept": 0, "proactivity": 0.5}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 0}},
                {"C": {"A1": 0, "A2": 0, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0, "proactivity": 0}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0.5, "proactivity": 0}},
                {"C": {"A1": 0, "A3": 1, "C1": 0, "C2": 1},
                 "S": {"syntax": 1, "concept": 1, "proactivity": 0}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 1, "concept": 0.5, "proactivity": 1}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 0, "C2": 0},
                 "S": {"syntax": 0, "concept": 1, "proactivity": 1}},
            ],
            "task3": [
                {"C": {"A1": 0, "A2": 1, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 0.5, "proactivity": 0}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 1, "concept": 1, "proactivity": 0.5}},
                {"C": {"A1": 0, "A2": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0.5, "concept": 1, "proactivity": 1}},
                {"C": {"A1": 0, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 1, "concept": 1, "proactivity": 1}},
                {"C": {"A1": 0, "A2": 1, "A3": 0, "C1": 0, "C2": 0},
                 "S": {"syntax": 1, "concept": 0, "proactivity": 0}},
                {"C": {"A1": 1, "A2": 1, "A3": 1, "C1": 1, "C2": 1},
                 "S": {"syntax": 0, "concept": 1, "proactivity": 1}},
            ],
        }

        # 预先按 γ 降序排列，Match 后取第一个即为 γ 最大的规则
        self.R_full_pos[self.task_key].sort(key=lambda x: x[self.metric], reverse=True)
        self.R_core_pos[self.task_key].sort(key=lambda x: x[self.metric], reverse=True)

    # ============================================================
    # 论文 (2)：模糊集直接校准 —— log-odds 方法（Ragin, 2008）
    # 公式：
    #   若 U_{t,j} >= θ_c : c = exp(ln19 * (U-θ_c)/(θ_f-θ_c)) / (1 + ...)
    #   若 U_{t,j} <  θ_c : c = exp(ln19 * (U-θ_c)/(θ_c-θ_n)) / (1 + ...)
    # ============================================================
    @staticmethod
    def _calibrate(value: float, theta_n: float, theta_c: float, theta_f: float) -> float:
        """将原始观测值校准为 [0,1] 模糊集隶属度。"""
        LN19 = math.log(19)  # ln(0.95/0.05)
        if theta_f == theta_c or theta_c == theta_n:
            # 退化情况：直接线性化
            if value >= theta_c:
                denom = theta_f - theta_c if theta_f != theta_c else 1e-9
                ratio = (value - theta_c) / denom
            else:
                denom = theta_c - theta_n if theta_c != theta_n else 1e-9
                ratio = (value - theta_c) / denom
            exp_val = math.exp(LN19 * ratio)
        else:
            if value >= theta_c:
                ratio = (value - theta_c) / (theta_f - theta_c)
            else:
                ratio = (value - theta_c) / (theta_c - theta_n)
            exp_val = math.exp(LN19 * ratio)
        return exp_val / (1.0 + exp_val)

    def _calibrate_state(self, a1: float, a2: float, a3: float,
                         c1: float, c2: float) -> dict:
        """
        将原始用户状态向量校准为模糊集隶属度向量 Ũ_t = (c_{t,1}, ..., c_{t,5})。
        论文 (2)：每种任务类型采用各自相应的校准锚点。
        """
        anchors = self.calibration_anchors[self.task_key]
        return {
            "A1": self._calibrate(a1, *anchors["A1"]),
            "A2": self._calibrate(a2, *anchors["A2"]),
            "A3": self._calibrate(a3, *anchors["A3"]),
            "C1": self._calibrate(c1, *anchors["C1"]),
            "C2": self._calibrate(c2, *anchors["C2"]),
        }

    # ============================================================
    # 论文 (3)：一致性判定函数 δ(c_i, u_i)
    # δ = 1 当且仅当 |c_i - u_i| <= 0.5，否则 δ = 0
    # 逻辑含义：
    #   c_i=1  → 要求 u_i >= 0.5（条件"存在"时当前状态须不低于交叉点）
    #   c_i=0  → 要求 u_i <= 0.5（条件"不存在"时当前状态须不高于交叉点）
    #   c_i=0.5→ 无约束，始终 δ=1（中间状态为不决定性因素）
    # ============================================================
    @staticmethod
    def _delta(c_i: float, u_i: float) -> int:
        return 1 if abs(c_i - u_i) <= 0.5 else 0

    # ============================================================
    # 论文 (3)：匹配算子 Match(C, Ũ) = ∏ δ(c_i, u_i)
    # C 中未指定的维度视同 c_i=0.5（不做约束，δ 恒为 1）
    # ============================================================
    def _match_rule(self, C: dict, U_tilde: dict) -> bool:
        """
        判断当前校准状态 U_tilde 是否与组态条件 C 匹配。
        C 中未出现的维度等价于 c_i=0.5，不构成约束。
        """
        for dim in self.STRATEGY_DIMS[:0]:  # 状态维度
            pass
        state_dims = ["A1", "A2", "A3", "C1", "C2"]
        for dim in state_dims:
            c_i = C.get(dim, 0.5)   # 缺失键 → 中间状态 0.5 → 无约束
            u_i = U_tilde[dim]
            if self._delta(c_i, u_i) == 0:
                return False
        return True

    # ============================================================
    # 论文 (4)：汉明距离 d_H(S_a, S_b) = Σ I(S_{a,l} ≠ S_{b,l})
    # ============================================================
    @staticmethod
    def _hamming(s_a: dict, s_b: dict) -> int:
        return sum(1 for dim in ProactiveScaffoldingSystem.STRATEGY_DIMS
                   if s_a.get(dim) != s_b.get(dim))

    # ============================================================
    # 论文 (4)：构建安全策略集合 S_safe，并取汉明距离最近的策略
    # S_safe = {S ∈ S | ¬∃ R_k ∈ R^-, Match(C_k, Ũ)=1 ∧ S_k = S}
    # ============================================================
    def _find_safe_strategy(self, U_tilde: dict, S_initial: dict) -> dict:
        """
        风险检查阶段：
        1. 枚举整个策略空间 S = {0, 0.5, 1}^3（共 27 种组合）
        2. 过滤掉所有触发风险规则的策略，得到 S_safe
        3. 若 S_safe ≠ ∅：返回 S_safe 中与 S_initial 汉明距离最近的策略
        4. 若 S_safe = ∅：返回 S_default
        """
        # 枚举 S = {0, 0.5, 1}^3
        all_strategies = [
            dict(zip(self.STRATEGY_DIMS, combo))
            for combo in product(self.STRATEGY_VALUES, repeat=3)
        ]

        # 筛选安全策略：对任意 R_k ∈ R^-，若 Match(C_k, Ũ)=1，则 S_k 是当前危险策略
        # 当前用户状态下的所有危险策略集合
        dangerous_strategies = set()
        for rule in self.R_neg[self.task_key]:
            if self._match_rule(rule["C"], U_tilde):
                # 将 S_k 转为 frozenset 便于集合操作
                s_k = tuple(rule["S"].get(d, 0.5) for d in self.STRATEGY_DIMS)
                dangerous_strategies.add(s_k)

        S_safe = [
            s for s in all_strategies
            if tuple(s.get(d, 0.5) for d in self.STRATEGY_DIMS) not in dangerous_strategies
        ]

        if not S_safe:
            # S_safe = ∅：使用默认策略
            logger.warning("⚠️ S_safe = ∅，无安全策略，回退至 S_default")
            return dict(self.default_strategy[self.task_key])

        # argmin_{S ∈ S_safe} d_H(S, S_initial)
        best = min(S_safe, key=lambda s: self._hamming(s, S_initial))
        return best

    # ============================================================
    # 主接口：decide_interaction_strategy
    # 实现论文映射 f(Ũ_t | R_full+, R_core+, R^-)
    # ============================================================
    def decide_interaction_strategy(self, a1: float, a2: float,
                                    a3: float, c1: float, c2: float) -> tuple:
        """
        输入原始用户状态 (a1, a2, a3, c1, c2)，输出最终执行策略 S_t* 及决策元数据。

        策略选择四阶段流程（论文 5.2.2）：
          (1) 规则形式化表示 — 见 __init__ 中的规则集配置
          (2) 用户状态模糊集校准
          (3) 高知识增益组态匹配（R_full+ → R_core+）
          (4) 低知识增益风险检查

        返回值：
          (S_star, strategy_meta)
          S_star       : dict，键为 'syntax', 'concept', 'proactivity'，值为 0 / 0.5 / 1
          strategy_meta: dict，包含决策过程的详细元数据，用于实验日志记录
            - calibrated_state   : 模糊集校准后的隶属度向量 Ũ_t
            - matched_rules      : 命中的组态规则列表（来自 R_full+ 或 R_core+）
            - matched_rule_source: 命中规则所属集合（"R_full+" / "R_core+" / "default"）
            - is_risk_detected   : 是否触发低知识增益风险规则（bool）
            - final_strategy_source: 最终策略来源（"R_full+" / "R_core+" / "safe_search" / "default"）
        """

        # ── 对照组：跳过策略计算，直接返回固定中值策略 ──────────
        if self.control_group:
            U_tilde = self._calibrate_state(a1, a2, a3, c1, c2)
            strategy_meta = {
                "calibrated_state": {k: round(v, 6) for k, v in U_tilde.items()},
                "matched_rules": [],
                "matched_rule_source": "control_group",
                "is_risk_detected": False,
                "final_strategy_source": "control_group",
            }
            logger.info("🔵 [对照组] 使用固定策略 syntax=0.5, concept=0.5, proactivity=0.5")
            return dict(self.CONTROL_GROUP_STRATEGY), strategy_meta

        # ── 初始化元数据收集器 ────────────────────────────────────
        strategy_meta = {
            "calibrated_state": {},
            "matched_rules": [],
            "matched_rule_source": "default",
            "is_risk_detected": False,
            "final_strategy_source": "default",
        }

        # ── 阶段 (2)：模糊集校准，得到 Ũ_t ──────────────────────
        U_tilde = self._calibrate_state(a1, a2, a3, c1, c2)
        strategy_meta["calibrated_state"] = {k: round(v, 6) for k, v in U_tilde.items()}
        logger.info(
            f"🔍 [状态校准] 原始值: A1={a1}, A2={a2:.4f}, A3={a3:.4f}, "
            f"C1={c1:.4f}, C2={c2:.4f}\n"
            f"   → 校准后 Ũ = {U_tilde}"
        )

        # ── 阶段 (3)：高知识增益组态匹配 ────────────────────────

        # 步骤 3a：匹配 R_full+（已按 γ 降序排列，首个命中即 γ 最大）
        M_full = [r for r in self.R_full_pos[self.task_key]
                  if self._match_rule(r["C"], U_tilde)]

        if M_full:
            # S_t* = argmax_{R_i ∈ M} γ_i 的 S_i
            best_rule = max(M_full, key=lambda r: r[self.metric])
            S_star = best_rule["S"]
            # 记录所有命中规则
            strategy_meta["matched_rules"] = [
                {"C": r["C"], "S": r["S"],
                 "consistency": r["consistency"], "raw_coverage": r["raw_coverage"]}
                for r in M_full
            ]
            strategy_meta["matched_rule_source"] = "R_full+"
            strategy_meta["final_strategy_source"] = "R_full+"
            logger.info(
                f"✅ 命中 R_full+ 规则（γ={best_rule['consistency']:.4f}）: "
                f"C={best_rule['C']} → S={S_star}"
            )
            logger.info(f"🏁 [最终策略] {S_star}")
            return dict(S_star), strategy_meta

        # 步骤 3b：R_full+ 无匹配，匹配 R_core+
        M_core = [r for r in self.R_core_pos[self.task_key]
                  if self._match_rule(r["C"], U_tilde)]

        if M_core:
            best_rule = max(M_core, key=lambda r: r[self.metric])
            S_initial = dict(best_rule["S"])
            strategy_meta["matched_rules"] = [
                {"C": r["C"], "S": r["S"],
                 "consistency": r["consistency"], "raw_coverage": r["raw_coverage"]}
                for r in M_core
            ]
            strategy_meta["matched_rule_source"] = "R_core+"
            logger.info(
                f"🔶 命中 R_core+ 规则（γ={best_rule['consistency']:.4f}）: "
                f"C={best_rule['C']} → S_initial={S_initial}"
            )
        else:
            # 步骤 3c：R_core+ 也无匹配，使用默认策略
            S_initial = dict(self.default_strategy[self.task_key])
            strategy_meta["matched_rules"] = []
            strategy_meta["matched_rule_source"] = "default"
            logger.info(f"⬜ R_core+ 无匹配，S_initial = S_default = {S_initial}")

        # ── 阶段 (4)：低知识增益风险检查 ────────────────────────

        # 判断是否存在 R_k ∈ R^-，使 Match(C_k, Ũ)=1 ∧ S_k = S_initial
        risk_triggered = any(
            self._match_rule(r["C"], U_tilde) and r["S"] == S_initial
            for r in self.R_neg[self.task_key]
        )
        strategy_meta["is_risk_detected"] = risk_triggered

        if not risk_triggered:
            # 无风险：S_t* = S_initial
            S_star = S_initial
            strategy_meta["final_strategy_source"] = strategy_meta["matched_rule_source"]
            logger.info(f"🛡️ 风险检查通过，S_t* = S_initial = {S_star}")
        else:
            # 有风险：在安全策略空间中取汉明距离最近的策略
            logger.warning(
                f"⚠️ S_initial={S_initial} 触发风险规则（R^-），进入安全策略搜索..."
            )
            S_star = self._find_safe_strategy(U_tilde, S_initial)
            strategy_meta["final_strategy_source"] = "safe_search"
            logger.info(f"🛡️ 安全策略已确定: {S_star}")

        logger.info(
            f"🏁 [最终策略] syntax={S_star['syntax']} | "
            f"concept={S_star['concept']} | proactivity={S_star['proactivity']}"
        )
        return S_star, strategy_meta