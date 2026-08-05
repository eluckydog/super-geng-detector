#!/usr/bin/env python3
"""
scripts/evidence_engine.py — 超级耿同学证据链合成引擎

将分散的检测结果合成为统一的后验造假概率：
- 贝叶斯网络建模证据间的条件依赖
- Bayes Factor 聚合
- Shapley 值各证据贡献度分解
- 置信区间（后验 HDI）
"""
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy import stats as sp_stats
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# Module 1: Evidence Likelihood Model
# ═══════════════════════════════════════════════════════════

def _alarm_log_lr(key: str) -> float:
    """根据证据 key 返回严重红旗的 log LR 权重。"""
    # (关键词 -> log LR)，命中则返回，否则用默认权重
    keyword_lr = [
        (("PRNU", "copy_move"), 20),
        (("GRIM",), 18),
        (("SD", "hierarchical"), 16),
        (("Benford",), 15),
        (("HMM", "AI"), 12),
        (("因果", "causal"), 8),
    ]
    for keywords, lr in keyword_lr:
        if any(k in key for k in keywords):
            return math.log(lr)
    return math.log(10)  # 默认严重权重


def evidence_to_likelihood(evidence_item: dict) -> float:
    """
    将单个证据项转化为似然比 P(evidence | fraud) / P(evidence | honest)。

    返回 log LR。
    """
    verdict = evidence_item.get("verdict", "")
    key = evidence_item.get("key", "")

    if "🚨" in verdict:
        return _alarm_log_lr(key)

    if "⚠️" in verdict:
        return math.log(3)  # 弱信号

    if "✅" in verdict:
        return math.log(0.5)  # 温和的"正常"证据

    return 0.0


# ═══════════════════════════════════════════════════════════
# Module 2: Bayes Net Evidence Synthesis
# ═══════════════════════════════════════════════════════════

class EvidenceNode:
    """贝叶斯网络节点：一个检测项"""
    def __init__(self, name: str, log_lr: float, category: str,
                 depends_on: List[str] = None):
        self.name = name
        self.log_lr = log_lr  # log P(evidence|fraud) - log P(evidence|honest)
        self.category = category
        self.depends_on = depends_on or []  # 依赖的父节点


def build_evidence_network(evidence_items: List[Dict[str, Any]]) -> dict:
    """
    构建证据贝叶斯网络并计算后验造假概率。

    使用条件独立的近似，将证据按类别分组后聚合。
    """
    if not evidence_items:
        return {
            "posterior_prob": 0.0,
            "bayes_factor": 1.0,
            "verdict": "无检测证据",
        }

    # 将证据项分类
    categories = defaultdict(list)
    for item in evidence_items:
        cat = item.get("category", "other")
        categories[cat].append(item)

    # 先验造假概率（基于已报道的造假率 ~2-5%，取保守的 5%）
    prior_log_odds = math.log(0.05 / 0.95)

    # 计算条件独立的 log LR
    total_log_lr = 0.0
    individual_lrs = {}

    for cat, items in categories.items():
        for item in items:
            key = item.get("key", "unknown")
            lr = evidence_to_likelihood(item)
            individual_lrs[key] = lr
            total_log_lr += lr

    # 后验 log odds
    posterior_log_odds = prior_log_odds + total_log_lr

    # 后验概率
    posterior_prob = 1 / (1 + math.exp(-posterior_log_odds))

    # Bayes Factor
    bayes_factor = math.exp(total_log_lr)

    return {
        "prior_prob": 0.05,
        "prior_log_odds": float(prior_log_odds),
        "total_log_lr": float(total_log_lr),
        "posterior_log_odds": float(posterior_log_odds),
        "posterior_prob": float(posterior_prob),
        "bayes_factor": float(bayes_factor),
        "individual_lrs": {k: float(v) for k, v in individual_lrs.items()},
        "n_evidence_items": len(evidence_items),
        "n_categories": len(categories),
    }


# ═══════════════════════════════════════════════════════════
# Module 3: Shapley Value Decomposition
# ═══════════════════════════════════════════════════════════

def shapley_decomposition(evidence_items: List[Dict[str, Any]]) -> dict:
    """
    计算每个证据项对后验造假概率的 Shapley 值贡献度。

    Shapley 值 = 边际贡献在所有可能排序上的平均。
    由于证据项数量可能较多，使用抽样近似。

    输出：每个证据项的贡献度（百分比）。
    """
    n = len(evidence_items)
    if n <= 1:
        return {"shapley_values": {}, "note": "证据项不足"}

    base = build_evidence_network([])["posterior_prob"]
    full = build_evidence_network(evidence_items)["posterior_prob"]

    # 为每个证据项计算边际贡献
    shapley = {}
    n_permutations = min(100, 2 ** n if n <= 10 else 100)

    for idx, item in enumerate(evidence_items):
        marginal_contributions = []
        for _ in range(n_permutations):
            order = np.random.permutation(n)
            subset_before = [evidence_items[j] for j in order[:np.where(order == idx)[0][0]]]
            subset_with = subset_before + [item]

            val_before = build_evidence_network(subset_before)["posterior_prob"]
            val_with = build_evidence_network(subset_with)["posterior_prob"]
            marginal_contributions.append(val_with - val_before)

        shapley[item.get("key", f"item_{idx}")] = float(np.mean(marginal_contributions))

    # 归一化
    total = sum(abs(v) for v in shapley.values())
    if total > 0:
        shapley_pct = {k: v / total * 100 for k, v in shapley.items()}
    else:
        shapley_pct = {k: 0.0 for k in shapley}

    # 排名
    ranking = sorted(shapley_pct.items(), key=lambda x: x[1], reverse=True)

    return {
        "shapley_values": {k: round(v, 2) for k, v in shapley_pct.items()},
        "top_contributors": ranking[:5],
        "base_prob": base,
        "full_prob": full,
    }


# ═══════════════════════════════════════════════════════════
# Module 4: Confidence Interval (Posterior HDI)
# ═══════════════════════════════════════════════════════════

def posterior_hdi(evidence_items: List[Dict[str, Any]],
                  n_simulations: int = 10000) -> dict:
    """
    通过蒙特卡洛模拟计算后验造假概率的 95% HDI。

    考虑证据的不确定性（每个证据的 LR 有一定范围）。
    """
    if not evidence_items:
        return {"hdi_low": 0.0, "hdi_high": 0.0, "n_sims": 0}

    prior_alpha = 5   # Beta(5, 95) → 先验均值 5%
    prior_beta = 95

    posterior_probs = []
    for _ in range(n_simulations):
        alpha = prior_alpha
        beta = prior_beta

        for item in evidence_items:
            lr = evidence_to_likelihood(item)
            # 添加不确定性：LR 在 [0.5*lr, 2*lr] 范围内均匀分布
            lr_noisy = lr + np.random.normal(0, abs(lr) * 0.3)
            if lr_noisy > 0:
                alpha += lr_noisy / (1 + lr_noisy)
                beta += 1 / (1 + lr_noisy)
            elif lr_noisy < 0:
                alpha -= lr_noisy / (1 - lr_noisy)
                beta -= 1 / (1 - lr_noisy)

        posterior_prob = np.random.beta(max(alpha, 0.1), max(beta, 0.1))
        posterior_probs.append(posterior_prob)

    posterior_probs = np.array(posterior_probs)
    hdi_low = float(np.percentile(posterior_probs, 2.5))
    hdi_high = float(np.percentile(posterior_probs, 97.5))
    median_prob = float(np.median(posterior_probs))

    return {
        "median_prob": median_prob,
        "hdi_95_low": hdi_low,
        "hdi_95_high": hdi_high,
        "mean_prob": float(np.mean(posterior_probs)),
        "n_simulations": n_simulations,
    }


# ═══════════════════════════════════════════════════════════
# Module 5: Synthesis Report
# ═══════════════════════════════════════════════════════════

def synthesize_evidence(stem_findings: List[Dict] = None,
                        humanities_findings: List[Dict] = None) -> dict:
    """
    合成所有证据，生成统一的后验造假概率和报告。

    参数：
    - stem_findings: 理工科引擎的检测结果列表（每项含 key, verdict, category）
    - humanities_findings: 文科引擎的检测结果列表
    """
    all_evidence = []
    if stem_findings:
        all_evidence.extend(stem_findings)
    if humanities_findings:
        all_evidence.extend(humanities_findings)

    if not all_evidence:
        return {"error": "无检测证据"}

    # 计算后验概率
    network_result = build_evidence_network(all_evidence)
    hdi_result = posterior_hdi(all_evidence)
    shapley = shapley_decomposition(all_evidence)

    # 综合判定
    p = network_result["posterior_prob"]
    bf = network_result["bayes_factor"]
    level, label = _fraud_level(p)
    BF_interpretation = _bf_interpretation(bf)

    return {
        "posterior_fraud_probability": float(p),
        "bayes_factor": float(bf),
        "hdi_95": [hdi_result["hdi_95_low"], hdi_result["hdi_95_high"]],
        "verdict_level": level,
        "verdict_label": label,
        "bf_interpretation": BF_interpretation,
        "top_evidence": shapley["top_contributors"][:3],
        "shapley_decomposition": shapley["shapley_values"],
        "n_total_evidence": len(all_evidence),
    }


def _fraud_level(p: float):
    """根据后验造假概率返回等级与标签。"""
    if p > 0.9:
        return "🔥🔥🔥 战斗力探测器爆表", "极可能造假"
    if p > 0.7:
        return "🔥🔥 高度可疑", "高度可疑"
    if p > 0.5:
        return "🔥 中度可疑", "中度可疑"
    if p > 0.3:
        return "⚡ 低度可疑", "低度可疑"
    return "💨 战斗力只有5", "无明显造假迹象"


def _bf_interpretation(bf: float) -> str:
    """根据贝叶斯因子返回解释文本。"""
    if bf > 100:
        return "极强证据（BF > 100）支持造假假设"
    if bf > 10:
        return "强证据（BF > 10）支持造假假设"
    if bf > 3:
        return "中等证据（BF > 3）支持造假假设"
    if bf > 1:
        return "弱证据（BF > 1）"
    return "证据不支持造假假设（BF < 1）"
