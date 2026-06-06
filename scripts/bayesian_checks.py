#!/usr/bin/env python3
"""
scripts/bayesian_checks.py — 超级耿同学贝叶斯检验引擎

NumPyro MCMC 驱动的深度检测：
- 分层贝叶斯 SD 一致性模型
- 后验预测分布检验
- Bayes Factor 计算（Savage-Dickey / Bridge Sampling）
- 贝叶斯剂量-效应模型（检测"太完美"的拟合）
- 贝叶斯变点检测（列差值恒定性）
"""
import itertools
import numpy as np
from scipy import stats as sp_stats
from typing import List, Tuple, Optional, Dict, Any

# JAX/NumPyro may not be installed — graceful fallback
_HAS_NUMPYRO = False
try:
    import jax.numpy as jnp
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    _HAS_NUMPYRO = True
except ImportError:
    pass


def _check_numpyro():
    if not _HAS_NUMPYRO:
        return {
            "error": "NumPyro/JAX 未安装。请运行: pip install numpyro jax",
            "fallback": True,
        }
    return None


# ═══════════════════════════════════════════════════════════
# Module 1: Hierarchical Bayesian SD Model
# ═══════════════════════════════════════════════════════════

def hierarchical_sd_model(groups: List[Tuple[float, float, int]],
                          num_samples: int = 2000,
                          num_warmup: int = 1000) -> dict:
    """
    分层贝叶斯模型：估计各组的真实方差/标准差。

    问题：如果论文声称多组"独立实验"的 SD 完全相同，
    这在统计学上几乎不可能。分层模型可以量化这种不可能性。

    模型：
    σ_g ~ LogNormal(μ_σ, τ_σ)    # 各组真实 SD 来自共同分布
    y_gi ~ Normal(μ_g, σ_g)       # 个体观测

    从后验推断：P(所有 σ_g 相等 | data) ≈ 0 还是 ≈ 1？
    """
    err = _check_numpyro()
    if err:
        return err

    if len(groups) < 2:
        return {"error": "至少需要2组数据"}

    # 准备数据
    all_y = []
    group_idx = []
    group_means = []
    for i, (mean, sd, n) in enumerate(groups):
        # 从声称的 mean±SD 模拟原始数据
        np.random.seed(42 + i)
        sim_data = np.random.normal(mean, sd, max(n, 2))
        all_y.extend(sim_data)
        group_idx.extend([i] * len(sim_data))
        group_means.append(mean)

    all_y = jnp.array(all_y)
    group_idx = jnp.array(group_idx, dtype=jnp.int32)
    n_groups = len(groups)

    def model():
        # 总体 SD 分布的超参数
        mu_sigma = numpyro.sample("mu_sigma", dist.Normal(0, 5))
        tau_sigma = numpyro.sample("tau_sigma", dist.HalfNormal(2))

        # 各组真实均值
        with numpyro.plate("groups", n_groups):
            mu_g = numpyro.sample("mu_g", dist.Normal(jnp.array(group_means), 5))
            sigma_g = numpyro.sample(
                "sigma_g",
                dist.LogNormal(mu_sigma, tau_sigma)
            )

        # 观测
        with numpyro.plate("data", len(all_y)):
            numpyro.sample(
                "obs",
                dist.Normal(mu_g[group_idx], sigma_g[group_idx]),
                obs=all_y,
            )

    # MCMC 采样
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(42))

    # 提取后验
    posterior = mcmc.get_samples()
    sigma_post = posterior["sigma_g"]  # shape: (num_samples, n_groups)

    # 计算各组 SD 的变异系数（后验）
    sigma_mean = jnp.mean(sigma_post, axis=0)
    sigma_std = jnp.std(sigma_post, axis=0)
    sigma_cv = sigma_std / sigma_mean

    # 检查"所有组 SD 相等"的后验概率
    # 计算最大的成对差异
    sigma_diff_max = jnp.max(sigma_post, axis=1) - jnp.min(sigma_post, axis=1)
    # 如果所有 SD 相等，最大差异应为 0
    epsilon = 0.01 * jnp.mean(sigma_mean)
    prob_all_equal = jnp.mean(sigma_diff_max < epsilon)

    # Bayes Factor: 模型内比较
    # BF = P(data|H1: SD不同) / P(data|H0: SD相同)
    # 简化为：后验 SD 变异度
    mean_cv = float(jnp.mean(sigma_cv))
    prob_equal = float(prob_all_equal)

    if prob_equal < 0.001:
        verdict = "🚨 分层贝叶斯模型：各组 SD 完全相同的后验概率 ≈ 0 —— 数据高度可疑"
        bf_approx = "BF > 1000"
    elif prob_equal < 0.05:
        verdict = "⚠️ 分层贝叶斯模型：各组 SD 相等的概率极低 ({:.3f})".format(prob_equal)
        bf_approx = f"BF ≈ {1/prob_equal:.0f}"
    else:
        verdict = "✅ SD 差异在合理范围"
        bf_approx = f"BF ≈ {1/prob_equal:.0f}" if prob_equal > 0 else "BF > 1000"

    return {
        "n_groups": n_groups,
        "sigma_posterior_mean": [float(x) for x in sigma_mean],
        "sigma_posterior_sd": [float(x) for x in sigma_std],
        "sigma_cv_posterior": [float(x) for x in sigma_cv],
        "mean_cv": mean_cv,
        "prob_all_sd_equal": prob_equal,
        "bayes_factor_approx": bf_approx,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 2: Posterior Predictive Check
# ═══════════════════════════════════════════════════════════

def posterior_predictive_check(observed_values: List[float],
                               claimed_mean: float,
                               claimed_sd: float,
                               n_simulations: int = 10000) -> dict:
    """
    贝叶斯后验预测检查（增强版）。

    检验论文声称的 mean±SD 是否与观测数据兼容。
    如果数据点落在声称分布极端尾部 → 数据与声称分布不兼容。

    增加：
    - 多尺度检验（整体 + 局部）
    - 极端值 z-score 分析
    - 模拟数据的 order statistic 比较
    """
    obs = np.array(observed_values)
    n = len(obs)

    # 模拟
    np.random.seed(42)
    simulated = np.random.normal(claimed_mean, claimed_sd, (n_simulations, n))

    # 逐点百分位
    anomalies = []
    for i in range(n):
        percentile = sp_stats.percentileofscore(simulated[:, i], obs[i]) / 100
        z_score = (obs[i] - claimed_mean) / claimed_sd
        is_extreme = abs(z_score) > 2.5
        anomalies.append({
            "value": float(obs[i]),
            "z_score": float(z_score),
            "percentile": float(percentile),
            "extreme": bool(is_extreme),
        })

    extreme_count = sum(1 for a in anomalies if a["extreme"])

    # 整体检验：Mahalanobis 距离
    try:
        cov_inv = np.linalg.inv(np.cov(simulated.T))
        obs_centered = obs - claimed_mean
        mahalanobis = obs_centered @ cov_inv @ obs_centered
        maha_percentile = 1 - sp_stats.chi2.cdf(mahalanobis, n)
    except np.linalg.LinAlgError:
        mahalanobis = None
        maha_percentile = None

    # order statistic 检验
    obs_sorted = np.sort(obs)
    sim_sorted = np.sort(simulated, axis=1)
    sim_lo = np.percentile(sim_sorted, 2.5, axis=0)
    sim_hi = np.percentile(sim_sorted, 97.5, axis=0)
    outside_ci = np.sum((obs_sorted < sim_lo) | (obs_sorted > sim_hi))

    # 判定
    if extreme_count / n > 0.3 or outside_ci > n * 0.3:
        verdict = "🚨 大量数据点极端异常，与声称分布严重不兼容"
    elif extreme_count > 0 or outside_ci > 0:
        verdict = f"⚠️ {extreme_count}/{n} 点极端，{outside_ci}/{n} 点超出 95% CI"
    else:
        verdict = "✅ 数据与声称分布兼容"

    return {
        "n": n,
        "claimed_mean": claimed_mean,
        "claimed_sd": claimed_sd,
        "simulations": n_simulations,
        "anomalies": anomalies,
        "extreme_count": extreme_count,
        "extreme_ratio": float(extreme_count / n),
        "outside_ci_count": int(outside_ci),
        "mahalanobis_percentile": float(maha_percentile) if maha_percentile else None,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 3: Bayesian Dose-Response Model
# ═══════════════════════════════════════════════════════════

def bayesian_dose_response(doses: List[float],
                           responses: List[float],
                           response_sds: List[float] = None,
                           num_samples: int = 2000,
                           num_warmup: int = 1000) -> dict:
    """
    贝叶斯剂量-效应模型：检测"太完美"的拟合曲线。

    真实生物/化学实验的剂量-效应曲线会有自然噪声。
    如果曲线过于光滑（所有点精确落在拟合线上）→ 数据可能编造。

    使用贝叶斯非线性回归 + 残差分析。
    """
    err = _check_numpyro()
    if err:
        return err

    if len(doses) < 3:
        return {"error": "至少需要3个剂量点"}

    doses_arr = jnp.array(doses)
    responses_arr = jnp.array(responses)
    if response_sds:
        sd_arr = jnp.array(response_sds)
    else:
        sd_arr = jnp.ones(len(responses)) * jnp.std(responses_arr) * 0.1

    def model():
        # Hill equation: R = Rmax * D^n / (Kd^n + D^n)
        Rmax = numpyro.sample("Rmax", dist.HalfNormal(10 * jnp.max(responses)))
        Kd = numpyro.sample("Kd", dist.LogNormal(jnp.log(jnp.median(doses)), 2))
        n_hill = numpyro.sample("n_hill", dist.Gamma(2, 2))

        # 残差标准差
        sigma_resid = numpyro.sample("sigma_resid", dist.HalfNormal(jnp.std(responses)))

        # 预测
        mu = Rmax * doses_arr ** n_hill / (Kd ** n_hill + doses_arr ** n_hill)

        with numpyro.plate("data", len(doses)):
            numpyro.sample("obs", dist.Normal(mu, sigma_resid), obs=responses_arr)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(42))

    posterior = mcmc.get_samples()

    # 检查拟合残差
    mu_post = jnp.mean(
        posterior["Rmax"][:, None] * doses_arr[None, :] ** posterior["n_hill"][:, None] /
        (posterior["Kd"][:, None] ** posterior["n_hill"][:, None] +
         doses_arr[None, :] ** posterior["n_hill"][:, None]),
        axis=0
    )
    residuals = responses_arr - mu_post
    resid_std = float(jnp.std(residuals))
    r_squared = 1 - jnp.sum(residuals ** 2) / jnp.sum(
        (responses_arr - jnp.mean(responses_arr)) ** 2
    )
    r_squared = float(r_squared)

    # "太完美"检测：R² 过高 + 残差非正态 + 无离群值
    too_perfect = False
    reasons = []
    if r_squared > 0.995:
        too_perfect = True
        reasons.append(f"R²={r_squared:.4f} 过高，真实生物实验难以达到此精度")
    if resid_std < 0.01 * np.mean(responses):
        too_perfect = True
        reasons.append("残差极小，数据过于完美")

    if too_perfect:
        verdict = "🚨 剂量-效应曲线过于完美，疑似编造：" + "；".join(reasons)
    elif r_squared > 0.98:
        verdict = f"⚠️ 剂量-效应拟合极好（R²={r_squared:.4f}），建议结合其他指标判断"
    else:
        verdict = f"✅ 剂量-效应拟合正常（R²={r_squared:.4f}）"

    return {
        "n_doses": len(doses),
        "r_squared": r_squared,
        "residual_std": resid_std,
        "reasons": reasons,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 4: Simplified Bayes Factor (Savage-Dickey)
# ═══════════════════════════════════════════════════════════

def savage_dickey_bf(sd_values: List[float]) -> dict:
    """
    Savage-Dickey 密度比：检验"所有 SD 相等"的 Bayes Factor。

    H0: 所有组的真实 SD 相同
    H1: 各组 SD 不同

    使用 Savage-Dickey 密度比近似：
    BF₁₀ = P(data|H1)/P(data|H0) ≈ prior(H0) / posterior(H0)

    由于先验下"所有 SD 精确相等"的概率为 0，
    改为检验"SD 的变异系数是否在合理范围"。
    """
    sds = np.array(sd_values)
    n = len(sds)

    if n < 2:
        return {"error": "至少需要2个 SD 值"}

    mean_sd = np.mean(sds)
    cv = np.std(sds) / mean_sd if mean_sd > 0 else 0

    # Fisher z-变换比较方差
    # H0: 所有方差相等 vs H1: 至少一个不同
    # 使用 Bartlett 检验
    # 简化为基于样本量 n_i 的模拟
    if n == 2:
        # F 检验
        F = (sds[0] / sds[1]) ** 2
        df1 = 10  # 假设每组 ~10 个样本
        df2 = 10
        p_equal = 2 * min(sp_stats.f.cdf(F, df1, df2),
                          1 - sp_stats.f.cdf(F, df1, df2))
    else:
        # Bartlett 检验（需要各组有足够样本）
        # 使用 CV 作为替代
        # 在假设每组 n=10 的情况下，期望 CV ≈ 1/sqrt(20) ≈ 0.22
        expected_cv = 0.22
        p_equal = 2 * (1 - sp_stats.norm.cdf(abs(cv / expected_cv - 1) * 3))

    bayes_factor = 1 / p_equal if p_equal > 0 else float('inf')

    if bayes_factor > 100:
        verdict = f"🚨 Bayes Factor = {bayes_factor:.0f}:1 支持 SD 不等（即 SD 不可能完全相同）"
    elif bayes_factor > 10:
        verdict = f"⚠️ Bayes Factor = {bayes_factor:.0f}:1，SD 相等的可能性极低"
    elif bayes_factor > 3:
        verdict = f"⚠️ 微弱证据支持 SD 不等（BF={bayes_factor:.1f}）"
    else:
        verdict = "✅ SD 差异在随机波动范围内"

    return {
        "n_groups": n,
        "sd_cv": float(cv),
        "approximate_bf": float(bayes_factor) if bayes_factor != float('inf') else None,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 5: Bayesian Changepoint for Column Diffs
# ═══════════════════════════════════════════════════════════

def bayesian_changepoint_diffs(columns: List[List[float]],
                               num_samples: int = 2000,
                               num_warmup: int = 1000) -> dict:
    """
    贝叶斯变点模型：检测列间差值是否经历结构变化。

    编造数据：差值从第一个到最后一个都是恒定的。
    真实数据：差值应有自然的随机波动，可能在不同数据段有不同的均值。

    模型检测差值序列中是否存在变点。
    没有变点 → 差值恒定 → 可疑。
    """
    err = _check_numpyro()
    if err:
        return err

    if len(columns) < 2:
        return {"error": "至少需要2列数据"}

    results = []
    for i, j in itertools.combinations(range(len(columns)), 2):
        c1, c2 = np.array(columns[i]), np.array(columns[j])
        if len(c1) != len(c2) or len(c1) < 5:
            continue

        diffs = c1 - c2
        n = len(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        # 简化的贝叶斯变点模型
        diffs_jnp = jnp.array(diffs)

        def changepoint_model():
            # 有两个状态：state 0 和 state 1
            # 状态转换点
            tau = numpyro.sample("tau", dist.Uniform(0, n - 1))

            # 两个状态的均值和标准差
            mu1 = numpyro.sample("mu1", dist.Normal(mean_diff, std_diff * 2))
            mu2 = numpyro.sample("mu2", dist.Normal(mean_diff, std_diff * 2))
            sigma = numpyro.sample("sigma", dist.HalfNormal(std_diff * 2))

            # 根据 tau 分配状态
            idx = jnp.arange(n)
            state = jnp.where(idx < tau, 0, 1)
            mu = jnp.where(state == 0, mu1, mu2)

            with numpyro.plate("data", n):
                numpyro.sample("obs", dist.Normal(mu, sigma), obs=diffs_jnp)

        nuts_kernel = NUTS(changepoint_model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples,
                    progress_bar=False)
        mcmc.run(jax.random.PRNGKey(42 + i * 100 + j))

        posterior = mcmc.get_samples()
        mu1_post = float(jnp.mean(posterior["mu1"]))
        mu2_post = float(jnp.mean(posterior["mu2"]))
        mu_diff = abs(mu1_post - mu2_post)

        # 如果两个状态的均值几乎相同 → 没有变点 → 差值恒定
        has_changepoint = mu_diff > std_diff * 0.5

        relative_variation = std_diff / abs(mean_diff) if abs(mean_diff) > 1e-10 else std_diff

        if relative_variation < 0.001 and not has_changepoint:
            verdict = "🚨 差值序列无变点且波动极小 → 数据非测量而是加减法编造"
        elif not has_changepoint and relative_variation < 0.01:
            verdict = "⚠️ 差值高度稳定，可能非自然数据"
        else:
            verdict = "✅ 差值自然波动，符合真实实验数据特征"

        results.append({
            "col_pair": f"列{i+1} vs 列{j+1}",
            "mean_diff": float(mean_diff),
            "std_diff": float(std_diff),
            "relative_variation": float(relative_variation),
            "mu_state1": mu1_post,
            "mu_state2": mu2_post,
            "has_changepoint": bool(has_changepoint),
            "verdict": verdict,
        })

    return {"bayesian_changepoint_analysis": results}
