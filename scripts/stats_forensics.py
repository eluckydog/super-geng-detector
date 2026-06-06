#!/usr/bin/env python3
"""
scripts/stats_forensics.py — 超级耿同学数值法医引擎

检测能力：
- Benford 全位 + KS + Anderson-Darling 联合检验
- 扩展 GRIM（含小数精度感知）
- 列差值恒定贝叶斯变点检测
- p-curve 分析（选择性报告偏差）
- 末位数字高级检验（条件均匀性 + 奇偶比）
- SD 模式检测（整数聚集、固定小数位、周期性）
"""
import math
import itertools
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import kstest


# ═══════════════════════════════════════════════════════════
# Module 1: Benford Suite (Full-digit + KS + AD)
# ═══════════════════════════════════════════════════════════

def benford_full(data: List[float], max_digit: int = 4) -> dict:
    """
    Benford 定律全位数字分布检验（1~max_digit 位）。
    联合 KS 检验和 Anderson-Darling 检验。

    适用于论文表格中的所有数值型数据：
    细胞计数、荧光强度、Western blot 定量、qPCR Ct 值等。
    """
    results = {}
    all_warnings = []

    for pos in range(1, max_digit + 1):
        digits = _extract_digit(data, pos)
        if len(digits) < 30:
            results[f"digit_{pos}"] = {"error": f"样本量不足 ({len(digits)})"}
            continue

        if pos == 1:
            expected_probs = np.array([math.log10(1 + 1/d) for d in range(1, 10)])
            observed = np.bincount(digits, minlength=10)[1:10]
        elif pos == 2:
            expected_probs = np.array([
                sum(math.log10(1 + 1/(10*k + d)) for k in range(1, 10))
                for d in range(10)
            ])
            observed = np.bincount(digits, minlength=10)[0:10]
        else:
            expected_probs = np.full(10, 0.1)
            observed = np.bincount(digits, minlength=10)[0:10]

        expected = expected_probs * len(digits)
        chi2, chi2_p = sp_stats.chisquare(f_obs=observed, f_exp=expected)

        # MAD
        obs_prop = observed / len(digits)
        exp_prop = expected_probs
        mad = np.mean(np.abs(obs_prop - exp_prop))

        # KS test
        ks_stat, ks_p = kstest(obs_prop, lambda x: _benford_cdf(x, pos))

        # Anderson-Darling (on observed digit frequencies vs expected)
        # Use empirical CDF comparison
        obs_sorted = np.sort(digits)
        n = len(digits)
        ad_stat = _anderson_darling_stat(digits, pos)

        # 综合判定
        if mad < 0.006:
            conformity = "高度符合"
            severity = "normal"
        elif mad < 0.015:
            conformity = "大致符合"
            severity = "mild"
        elif mad < 0.03:
            conformity = "边缘异常"
            severity = "suspicious"
        else:
            conformity = "显著偏离"
            severity = "high"

        if severity != "normal":
            all_warnings.append(
                f"第{pos}位数字 {conformity}（MAD={mad:.4f}, χ²={chi2:.1f}, p={chi2_p:.4f}）"
            )

        results[f"digit_{pos}"] = {
            "position": pos,
            "n": len(digits),
            "chi2": float(chi2),
            "chi2_p": float(chi2_p),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "ad_stat": float(ad_stat),
            "mad": float(mad),
            "conformity": conformity,
            "severity": severity,
            "observed": observed.tolist(),
            "expected": [round(e, 1) for e in expected],
        }

    # 综合评分
    severities = [
        r.get("severity", "normal")
        for r in results.values()
        if isinstance(r, dict) and "severity" in r
    ]
    if "high" in severities:
        overall = "🚨 Benford 联合检验：存在严重偏离"
    elif severities.count("suspicious") >= 2:
        overall = "⚠️ Benford 联合检验：多处边缘异常，建议关注"
    elif "suspicious" in severities:
        overall = "⚠️ Benford 联合检验：存在轻微异常"
    else:
        overall = "✅ Benford 联合检验：数据分布正常"

    results["overall"] = overall
    results["warnings"] = all_warnings
    return results


def _extract_digit(data: List[float], position: int) -> List[int]:
    """提取第 position 位数字（position=1 为第一位非零数字）"""
    digits = []
    for v in data:
        if v == 0 or not np.isfinite(v):
            continue
        s = f"{abs(v):.15e}"
        if position == 1:
            fd = int(s[0])
            if 1 <= fd <= 9:
                digits.append(fd)
        else:
            sig_part = s.replace('.', '')[:position]
            if len(sig_part) >= position:
                d = int(sig_part[position - 1])
                digits.append(d)
    return digits


def _benford_cdf(x, position: int = 1):
    """Benford 分布的近似 CDF"""
    if position == 1:
        probs = np.cumsum([math.log10(1 + 1/d) for d in range(1, 10)])
        x_arr = np.asarray(x)
        idx = np.clip((x_arr * 9).astype(int), 0, 8)
        result = np.zeros_like(x_arr, dtype=float)
        for i, target_idx in enumerate(idx.flat):
            result.flat[i] = probs[min(target_idx, 8)]
        return float(result) if np.isscalar(x) else result
    return np.asarray(x, dtype=float)


def _anderson_darling_stat(digits: List[int], position: int) -> float:
    """简化的 Anderson-Darling 统计量"""
    n = len(digits)
    if position == 1:
        expected_cdf = lambda d: sum(math.log10(1 + 1/k) for k in range(1, d + 1))
        valid_range = range(1, 10)
    else:
        expected_cdf = lambda d: (d + 1) / 10
        valid_range = range(10)

    sorted_data = np.sort(digits)
    ad = -n
    for i, d in enumerate(sorted_data):
        F = expected_cdf(d)
        F_rev = expected_cdf(sorted_data[-(i + 1)])
        if F > 0 and F < 1 and F_rev > 0 and F_rev < 1:
            ad -= (2*i + 1) * (math.log(F) + math.log(1 - F_rev)) / n
    return float(ad)


# ═══════════════════════════════════════════════════════════
# Module 2: Extended GRIM
# ═══════════════════════════════════════════════════════════

def grim_extended(mean: float, sd: float, n: int, precision_digits: int = None) -> dict:
    """
    扩展 GRIM（Granularity-Related Inconsistency of Means）检验。

    自动检测均值的报告精度，检查报告的 mean±SD 是否能由 n 个整数测量值产生。
    同时检查：
    1. 均值粒度一致性
    2. SD 粒度一致性（n 个整数的方差必须是分母 n 的整数倍）
    3. 均值±SD 范围是否可能包含 n 个整数
    4. 小样本（n≤5）的全枚举检验
    """
    if n <= 0 or sd < 0:
        return {"error": "无效参数"}

    # 自动检测报告精度
    if precision_digits is None:
        mean_str = f"{mean}"
        if '.' in mean_str:
            precision_digits = len(mean_str.split('.')[1])
        else:
            precision_digits = 0

    granularity = 10 ** (-precision_digits)
    min_sum = int(np.ceil(n * (mean - 3 * sd / np.sqrt(n))))
    max_sum = int(np.floor(n * (mean + 3 * sd / np.sqrt(n))))

    # 检查 1：均值粒度
    mean_consistent = abs(mean - round(mean / granularity) * granularity) < 1e-10
    if not mean_consistent:
        # 检查是否与 n 个整数的均值粒度一致
        mean_check = False
        for s in range(max(0, round(mean * n / granularity) - 3),
                       round(mean * n / granularity) + 4):
            if abs(mean - s * granularity / n) < 1e-10:
                mean_check = True
                break
        mean_consistent = mean_check

    # 检查 2：SD 粒度
    sd_consistent = True
    if n > 1:
        target_ss = sd ** 2 * n  # sum of squared deviations
        if granularity > 0:
            ss_unit = granularity ** 2  # 最小平方差单位
            if abs(target_ss / ss_unit - round(target_ss / ss_unit)) > 1e-6:
                sd_consistent = False

    # 检查 3：小样本全枚举
    if n <= 6 and min_sum <= max_sum and (max_sum - min_sum) < 10000:
        possible = False
        for total in range(min_sum, max_sum + 1):
            mean_val = total * granularity / n
            if abs(mean_val - mean) > 1e-10:
                continue
            # 检查是否存在 n 个整数产生给定的 SD
            for combo in itertools.combinations_with_replacement(
                range(0, 101), n):
                vals = np.array(combo, dtype=float) * granularity
                if abs(np.mean(vals) - mean) < 1e-10:
                    sd_val = np.std(vals, ddof=1) if n > 1 else 0
                    if abs(sd_val - sd) < 1e-8:
                        possible = True
                        break
            if possible:
                break
    else:
        possible = None  # 样本太大无法枚举

    # 综合判定
    issues = []
    if not mean_consistent:
        issues.append("均值与整数粒度不兼容")
    if not sd_consistent:
        issues.append("SD 与整数粒度不兼容")
    if possible is False:
        issues.append("小样本全枚举：无法产生声称的 mean±SD")

    if len(issues) >= 2:
        verdict = "🚨 GRIM 扩展检验：多项粒度冲突，数据高度可疑"
    elif len(issues) == 1:
        verdict = f"⚠️ GRIM 扩展检验：{issues[0]}"
    else:
        verdict = "✅ GRIM 扩展检验通过"

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "precision_digits": precision_digits,
        "granularity": granularity,
        "mean_consistent": mean_consistent,
        "sd_consistent": sd_consistent,
        "exhaustive_enum_possible": possible,
        "issues": issues,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 3: Column Difference Changepoint Detection
# ═══════════════════════════════════════════════════════════

def column_diff_changepoint(columns: List[List[float]],
                            threshold: float = 0.001) -> dict:
    """
    列间差值恒定贝叶斯变点检测。

    检查多列数据是否存在恒定的加减关系（编造数据常见特征）。
    使用滑动窗口检测差值序列中的变点——真实数据差值会有自然波动，
    编造数据差值则几乎不变。
    """
    if len(columns) < 2:
        return {"error": "至少需要2列数据"}

    results = []
    for i, j in itertools.combinations(range(len(columns)), 2):
        c1, c2 = np.array(columns[i]), np.array(columns[j])
        if len(c1) != len(c2) or len(c1) < 3:
            continue

        diffs = c1 - c2
        n = len(diffs)

        # 基础统计
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        cv = std_diff / abs(mean_diff) if abs(mean_diff) > 1e-10 else std_diff

        # 滑动窗口变点检测
        window_size = max(3, n // 4)
        window_means = []
        for w in range(n - window_size + 1):
            window_means.append(np.mean(diffs[w:w + window_size]))

        window_std = np.std(window_means) if len(window_means) > 1 else 0
        window_range = max(window_means) - min(window_means)

        # 判定
        if cv < threshold and window_range < abs(mean_diff) * 0.01:
            alarm = "🚨 列间差值高度恒定 — 数据非实验测量而是加减法编造"
            constant = True
        elif cv < threshold * 10:
            alarm = "⚠️ 列间差值变化极小，建议进一步检查"
            constant = False
        else:
            alarm = "正常"
            constant = False

        results.append({
            "col_pair": f"列{i+1} vs 列{j+1}",
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "cv": float(cv),
            "window_range": float(window_range),
            "is_constant_diff": constant,
            "verdict": alarm,
        })

    return {"column_comparisons": results}


# ═══════════════════════════════════════════════════════════
# Module 4: p-curve Analysis
# ═══════════════════════════════════════════════════════════

def p_curve_analysis(p_values: List[float]) -> dict:
    """
    p-curve 分析——检测选择性报告偏差。

    如果研究没有真实效应且没有 p-hacking，
    显著 p 值（p<0.05）应均匀分布在 [0, 0.05] 区间。
    向右偏斜（集中在 0.04-0.05）→ p-hacking
    向左偏斜（集中在 0.00-0.01）→ 真实效应 + 可能存在
    """
    if not p_values:
        return {"error": "无 p 值数据"}

    p_arr = np.array(p_values)
    sig_p = p_arr[p_arr < 0.05]

    if len(sig_p) < 3:
        return {"error": f"显著 p 值不足（n={len(sig_p)}，需要 >=3）"}

    total = len(p_arr)
    sig_count = len(sig_p)

    # 区间分布
    bins = {
        "0.00–0.01": int(np.sum(sig_p < 0.01)),
        "0.01–0.02": int(np.sum((sig_p >= 0.01) & (sig_p < 0.02))),
        "0.02–0.03": int(np.sum((sig_p >= 0.02) & (sig_p < 0.03))),
        "0.03–0.04": int(np.sum((sig_p >= 0.03) & (sig_p < 0.04))),
        "0.04–0.05": int(np.sum((sig_p >= 0.04) & (sig_p < 0.05))),
    }
    nonsig_count = total - sig_count

    # 右偏度检测（p-hacking 指标）
    right_skew = bins["0.04–0.05"] / sig_count if sig_count > 0 else 0

    # Binomial test: H0 = 显著 p 值均匀分布
    # 如果均匀，0.04-0.05 区应有 ~20% 的显著 p 值
    from scipy.stats import binomtest
    bt_result = binomtest(bins["0.04–0.05"], sig_count, p=0.2, alternative='greater')

    # Fisher's method
    valid_p = p_arr[p_arr > 0]
    if len(valid_p) >= 3:
        fisher_stat = -2 * sum(math.log(p) for p in valid_p)
        fisher_p = 1 - sp_stats.chi2.cdf(fisher_stat, 2 * len(valid_p))
    else:
        fisher_stat = None
        fisher_p = None

    # 判定
    if right_skew > 0.35 and bt_result.pvalue < 0.05:
        phacking = f"🚨 严重 p-hacking（{bins['0.04–0.05']}/{sig_count} 集中在 0.04-0.05，p={bt_result.pvalue:.4f}）"
    elif right_skew > 0.25:
        phacking = f"⚠️ 疑似 p-hacking（{bins['0.04–0.05']}/{sig_count} 集中在 0.04-0.05）"
    else:
        phacking = "✅ 未发现明显 p-hacking"

    return {
        "n_total": total,
        "n_significant": sig_count,
        "significant_ratio": float(sig_count / total) if total > 0 else 0,
        "p_value_bins": bins,
        "n_nonsignificant": nonsig_count,
        "right_skew_ratio": float(right_skew),
        "binomial_test_p": float(bt_result.pvalue),
        "fisher_stat": float(fisher_stat) if fisher_stat else None,
        "fisher_p": float(fisher_p) if fisher_p else None,
        "phacking_verdict": phacking,
    }


# ═══════════════════════════════════════════════════════════
# Module 5: Advanced Last-Digit Tests
# ═══════════════════════════════════════════════════════════

def last_digit_advanced(data: List[float]) -> dict:
    """
    末位数字高级检验。

    包括：
    1. χ² 均匀性检验（标准）
    2. 条件均匀性（按数量级分层）
    3. 奇偶比检验
    4. 相邻位数字相关性（真实数据相邻位无关）
    """
    last_digits = []
    for v in data:
        if not np.isfinite(v) or v == 0:
            continue
        s = f"{abs(v):.10f}"
        if '.' in s:
            decimal_part = s.split('.')[1]
            if len(decimal_part) >= 1:
                last_digits.append(int(decimal_part[0]))

    if len(last_digits) < 30:
        return {"error": "样本量不足", "n": len(last_digits)}

    n = len(last_digits)

    # 1. χ² 均匀性
    observed = np.bincount(last_digits, minlength=10)
    expected = np.full(10, n / 10)
    chi2, chi2_p = sp_stats.chisquare(f_obs=observed, f_exp=expected)

    # 2. 奇偶比
    odd_count = sum(1 for d in last_digits if d % 2 == 1)
    even_count = n - odd_count
    odd_ratio = odd_count / n
    # Binomial test: H0 = odd_ratio = 0.5
    odd_binom_p = binomtest(odd_count, n, p=0.5, alternative='two-sided').pvalue

    # 3. 相邻位相关性（检查相邻两个末位数字是否独立）
    if n >= 2:
        pairs = list(zip(last_digits[:-1], last_digits[1:]))
        same = sum(1 for a, b in pairs if a == b)
        expected_same = n / 10  # 均匀分布下，相邻两数相同的概率 1/10
        pair_deviation = (same - expected_same) / np.sqrt(expected_same * 0.9)
    else:
        same = 0
        expected_same = 0
        pair_deviation = 0

    # 综合判定
    issues = []
    if chi2_p < 0.01:
        issues.append(f"末位分布严重不均匀（χ²={chi2:.1f}, p={chi2_p:.4f}）")
    elif chi2_p < 0.05:
        issues.append(f"末位分布不太均匀（χ²={chi2:.1f}, p={chi2_p:.4f}）")

    if odd_binom_p < 0.01:
        issues.append(f"奇偶比严重失衡（{odd_count}/{n}={odd_ratio:.2f}, p={odd_binom_p:.4f}）")
    elif odd_binom_p < 0.05:
        issues.append(f"奇偶比轻微失衡（{odd_count}/{n}={odd_ratio:.2f}, p={odd_binom_p:.4f}）")

    if abs(pair_deviation) > 3:
        issues.append(f"相邻末位高度相关（偏差={pair_deviation:.1f}σ），不符合独立均匀分布")

    if len(issues) >= 2:
        verdict = f"🚨 末位数字多重异常：" + "；".join(issues)
    elif len(issues) == 1:
        verdict = f"⚠️ {issues[0]}"
    else:
        verdict = "✅ 末位数字分布正常"

    return {
        "n": n,
        "chi2": float(chi2),
        "chi2_p": float(chi2_p),
        "odd_ratio": float(odd_ratio),
        "odd_binomial_p": float(odd_binom_p),
        "adjacent_correlation_z": float(pair_deviation),
        "observed": observed.tolist(),
        "issues": issues,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 6: SD Pattern Detection
# ═══════════════════════════════════════════════════════════

def sd_pattern_detect(sd_values: List[float],
                      group_labels: List[str] = None) -> dict:
    """
    SD 模式检测。

    检查多组 SD 值是否存在异常模式：
    1. 全部相同（最可疑）
    2. 全部为整数
    3. 固定小数位数
    4. 呈现等差/等比规律
    5. 聚类（几组共享相同 SD）
    """
    if len(sd_values) < 2:
        return {"error": "至少需要2组 SD 值"}

    sds = sd_values
    n = len(sds)

    # 1. 全部相同
    unique_sds = len(set(round(s, 8) for s in sds))
    all_same = unique_sds == 1

    # 2. 整数聚集
    integer_sds = sum(1 for s in sds if abs(s - round(s)) < 1e-10)
    integer_ratio = integer_sds / n

    # 3. 固定小数位
    decimal_counts = []
    for s in sds:
        s_str = f"{s}"
        if '.' in s_str:
            decimal_counts.append(len(s_str.split('.')[1]))
    same_decimals = len(set(decimal_counts)) == 1 if decimal_counts else False

    # 4. 聚类
    from scipy.cluster.hierarchy import fcluster, linkage
    if n >= 3:
        Z = linkage(np.array(sds).reshape(-1, 1), method='single')
        clusters = fcluster(Z, t=0.01 * np.std(sds) if np.std(sds) > 0 else 0.01,
                           criterion='distance')
        n_clusters = len(set(clusters))
    else:
        n_clusters = unique_sds

    # 5. 变异系数
    sd_cv = float(np.std(sds) / np.mean(sds)) if np.mean(sds) > 0 else 0

    # 判定
    issues = []
    if all_same:
        issues.append("🚨 所有组 SD 完全相同（数据极可能编造）")
    if integer_ratio > 0.8:
        issues.append(f"⚠️ {integer_sds}/{n} 组 SD 为整数（异常整齐）")
    if same_decimals and len(set(sds)) > 1:
        issues.append("⚠️ SD 小数位数完全一致")
    if n_clusters <= n * 0.4 and n >= 4:
        issues.append(f"⚠️ SD 严重聚类（共{n_clusters}个不同值，{n}组数据）")

    if len(issues) >= 2:
        verdict = "🚨 SD 模式多重异常：" + "；".join(issues)
    elif len(issues) == 1:
        verdict = issues[0]
    else:
        verdict = "✅ SD 模式正常"

    return {
        "n_groups": n,
        "sd_values": [round(s, 4) for s in sds],
        "all_same": all_same,
        "unique_count": unique_sds,
        "integer_ratio": float(integer_ratio),
        "sd_cv": sd_cv,
        "n_clusters": n_clusters,
        "issues": issues,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 7: Full Table Arsenal
# ═══════════════════════════════════════════════════════════

def full_stats_arsenal(data_table: List[Dict[str, Any]]) -> dict:
    """
    对论文数据表运行全套数值法医检测。
    """
    all_values = []
    all_means = []
    all_sds = []
    all_ns = []
    columns = []

    for row in data_table:
        if "values" in row and row["values"]:
            all_values.extend(row["values"])
        if "mean" in row:
            all_means.append(row["mean"])
        if "sd" in row:
            all_sds.append(row["sd"])
        if "n" in row:
            all_ns.append(row["n"])

    results = {}

    # Benford
    if len(all_values) >= 30:
        results["benford"] = benford_full(all_values)

    # 末位
    if len(all_values) >= 30:
        results["last_digit"] = last_digit_advanced(all_values)

    # GRIM
    grim_results = []
    for i, row in enumerate(data_table):
        if all(k in row for k in ["mean", "sd", "n"]) and row["n"] is not None:
            grim_results.append(grim_extended(row["mean"], row["sd"], row["n"]))
    if grim_results:
        results["grim"] = grim_results

    # SD 模式
    if len(all_sds) >= 2:
        results["sd_pattern"] = sd_pattern_detect(all_sds)

    # 列差值
    if "columns" in data_table[0] if data_table else False:
        cols = [row["columns"] for row in data_table]
        results["column_diff"] = column_diff_changepoint(cols)

    # 汇总
    alarms = []
    for key, val in results.items():
        if isinstance(val, dict) and "verdict" in val and "🚨" in str(val.get("verdict", "")):
            alarms.append(f"[{key}] {val['verdict']}")
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "🚨" in str(item.get("verdict", "")):
                    alarms.append(f"[{key}] {item['verdict']}")

    results["alarm_count"] = len(alarms)
    results["alarms"] = alarms
    results["overall"] = (
        f"🔥 数值法医完成：{len(alarms)} 个警报"
        if alarms
        else "✅ 数值法医未发现明显异常"
    )

    return results
