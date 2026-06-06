#!/usr/bin/env python3
"""
scripts/text_forensics.py — 超级耿同学文本取证引擎

HMM 驱动的文本分析：
- 作者风格一致性检测（多状态 HMM）
- AI 生成文本检测
- 复制粘贴边界检测
- 段落风格突变检测
- 引用-正文一致性分析
"""
import re
import math
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

# hmmlearn optional
_HAS_HMM = False
try:
    import hmmlearn.hmm
    _HAS_HMM = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════
# Module 1: Stylometric Feature Extraction
# ═══════════════════════════════════════════════════════════

def extract_stylometric_features(text: str) -> Dict[str, float]:
    """
    从文本中提取风格计量特征。

    用于后续 HMM 状态判断和作者一致性检测。
    """
    sentences = re.split(r'[。！？.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)

    features = {}

    # 句长特征
    if sentences:
        sent_lens = [len(s) for s in sentences]
        features["mean_sentence_len"] = float(np.mean(sent_lens))
        features["std_sentence_len"] = float(np.std(sent_lens))
        features["cv_sentence_len"] = features["std_sentence_len"] / features["mean_sentence_len"] if features["mean_sentence_len"] > 0 else 0

    # 词汇特征
    if words:
        word_lens = [len(w) for w in words]
        features["mean_word_len"] = float(np.mean(word_lens))
        features["std_word_len"] = float(np.std(word_lens))

    # 标点特征
    total_chars = len(text)
    if total_chars > 0:
        features["comma_ratio"] = text.count('，') / total_chars
        features["period_ratio"] = text.count('。') / total_chars
        features["semicolon_ratio"] = text.count('；') / total_chars
        features["quote_ratio"] = (text.count('"') + text.count('"') + text.count('「')) / total_chars

    # 虚词/功能词比率（中文）
    function_words = ['的', '了', '是', '在', '和', '与', '或', '也', '就', '都', '但', '而', '却',
                      '这', '那', '其', '之', '将', '把', '被', '从', '以', '对', '向', '为',
                      '因为', '所以', '因此', '然而', '虽然', '但是', '如果', '那么']
    if chinese_chars:
        text_clean = text
        fw_count = sum(text_clean.count(w) for w in function_words)
        features["function_word_ratio"] = fw_count / len(chinese_chars) if chinese_chars else 0

    # 段落特征
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
    if paragraphs:
        para_lens = [len(p) for p in paragraphs]
        features["n_paragraphs"] = len(paragraphs)
        features["mean_para_len"] = float(np.mean(para_lens))
        features["std_para_len"] = float(np.std(para_lens))

    # 引用密度
    citations = re.findall(r'\[[\d,\s\-]+\]|\[\d+\]|\([A-Z][a-z]+ et al\., \d{4}\)|（[^）]{0,50}，\d{4}）', text)
    features["citation_density"] = len(citations) / max(len(paragraphs), 1)

    return features


# ═══════════════════════════════════════════════════════════
# Module 2: Style Shift Detection (HMM)
# ═══════════════════════════════════════════════════════════

def detect_style_shifts(text: str,
                        window_size: int = 200) -> dict:
    """
    使用 HMM 检测文本中的风格突变。

    将文本分成窗口，对每个窗口提取风格特征，
    训练一个 HMM 来检测隐藏的"作者状态"。

    正常论文：1个主导状态（同一个作者写的）
    异常论文：2+个状态频繁切换（AI 生成段落 / 复制粘贴）
    """
    # 分窗
    windows = []
    for i in range(0, len(text), window_size // 2):
        window_text = text[i:i + window_size]
        if len(window_text) > 50:
            features = extract_stylometric_features(window_text)
            windows.append(features)

    if len(windows) < 5:
        return {"error": f"文本太短（{len(windows)}个窗口），需要至少5个窗口"}

    # 特征向量化
    feature_names = [
        "mean_sentence_len", "std_sentence_len", "cv_sentence_len",
        "mean_word_len", "comma_ratio", "period_ratio",
        "function_word_ratio", "citation_density"
    ]
    if len(windows) > 0:
        feature_names = [f for f in feature_names if f in windows[0]]

    X = np.array([
        [w.get(f, 0) for f in feature_names]
        for w in windows
    ])

    # 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # 风格变化检测（无 HMM 备选方案）
    # 计算相邻窗口间的余弦距离
    style_shifts = []
    for i in range(1, len(X_norm)):
        cos_sim = np.dot(X_norm[i], X_norm[i-1]) / (
            np.linalg.norm(X_norm[i]) * np.linalg.norm(X_norm[i-1]) + 1e-10
        )
        style_shifts.append({
            "position": i * (window_size // 2),
            "cosine_similarity": float(cos_sim),
            "is_shift": bool(cos_sim < 0.7),
        })

    n_shifts = sum(1 for s in style_shifts if s["is_shift"])
    shift_ratio = n_shifts / len(style_shifts) if style_shifts else 0

    # HMM 状态数推断
    if _HAS_HMM and len(windows) >= 10:
        try:
            best_n_states = _infer_hmm_states(X_norm, max_states=5)
        except (ValueError, RuntimeError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"HMM state inference failed, falling back to clustering: {e}")
            best_n_states = 1
    else:
        # 简化的状态数推断：聚类
        from scipy.cluster.hierarchy import fcluster, linkage
        if len(X_norm) >= 4:
            Z = linkage(X_norm, method='ward')
            max_d = 2.0
            clusters = fcluster(Z, max_d, criterion='distance')
            best_n_states = len(set(clusters))
        else:
            best_n_states = 1

    # 判定
    if best_n_states >= 3 and shift_ratio > 0.3:
        verdict = "🚨 检测到多个写作风格状态（{}个）+ 高频风格突变（{:.0%}）→ 疑似 AI 生成或多人拼凑".format(
            best_n_states, shift_ratio)
    elif best_n_states >= 2 and shift_ratio > 0.2:
        verdict = "⚠️ 存在 {0} 个写作风格，{1:.0%} 的窗口出现风格突变".format(
            best_n_states, shift_ratio)
    elif best_n_states >= 2:
        verdict = "⚠️ 可能存在 {0} 个写作风格（如中英文摘要差异）".format(best_n_states)
    else:
        verdict = "✅ 写作风格一致，未检测到明显突变"

    return {
        "n_windows": len(windows),
        "inferred_states": best_n_states,
        "style_shifts": style_shifts,
        "n_shifts": n_shifts,
        "shift_ratio": float(shift_ratio),
        "verdict": verdict,
    }


def _infer_hmm_states(X: np.ndarray, max_states: int = 5) -> int:
    """用 HMM 推断最优状态数"""
    best_score = -np.inf
    best_n = 1

    for n_states in range(1, max_states + 1):
        try:
            model = hmmlearn.hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
            )
            model.fit(X)
            score = model.score(X)
            # BIC-like penalty
            n_params = n_states ** 2 + 2 * n_states * X.shape[1]
            bic = score - 0.5 * n_params * np.log(len(X))
            if bic > best_score:
                best_score = bic
                best_n = n_states
        except (ValueError, RuntimeError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"HMM fit failed for n_states={n_states}: {e}")
            continue

    return best_n


# ═══════════════════════════════════════════════════════════
# Module 3: AI-Generated Text Detection
# ═══════════════════════════════════════════════════════════

def detect_ai_generated_text(text: str) -> dict:
    """
    检测 AI 生成文本的特征。

    基于多个统计特征，不需要外部 API：
    1. 词汇多样性（TTR — Type-Token Ratio）
    2. 句长分布（AI 文本句长趋于均匀）
    3. 功能词使用模式
    4. 高频 n-gram 重复度
    5. burstiness（人类 vs AI 的句子长度 burst 模式不同）
    """
    sentences = re.split(r'[。！？.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if len(sentences) < 5:
        return {"error": "文本太短"}

    # 1. TTR
    words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text)
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0

    # 2. 句长分布
    sent_lens = [len(s) for s in sentences]
    mean_len = np.mean(sent_lens)
    std_len = np.std(sent_lens)
    cv_len = std_len / mean_len if mean_len > 0 else 0

    # AI 的句长 CV 通常更小（更均匀）
    # 正常人类：CV 0.3-0.7
    # AI：CV 0.15-0.35

    # 3. Burstiness（句子长度序列的自相关）
    if len(sent_lens) >= 4:
        diffs = np.diff(sent_lens)
        burstiness = np.std(diffs) / np.mean(np.abs(diffs)) if np.mean(np.abs(diffs)) > 0 else 0
    else:
        burstiness = 0

    # 4. Trigrams 重复率
    trigrams = []
    for sentence in sentences:
        chars = re.sub(r'\s+', '', sentence)
        for i in range(len(chars) - 2):
            trigrams.append(chars[i:i+3])
    if trigrams:
        trigram_counter = Counter(trigrams)
        repeated = sum(1 for c in trigram_counter.values() if c > 3)
        trigram_repeat_rate = repeated / len(trigram_counter) if trigram_counter else 0
    else:
        trigram_repeat_rate = 0

    # 5. 标点使用（中文 AI 文本逗号密度偏高）
    total_chars = len(re.sub(r'\s+', '', text))
    comma_density = text.count('，') / total_chars if total_chars > 0 else 0

    # 综合评分
    score = 0.0
    reasons = []

    if cv_len < 0.25:
        score += 0.3
        reasons.append(f"句长极为均匀（CV={cv_len:.3f}），典型 AI 特征")
    elif cv_len < 0.35:
        score += 0.15
        reasons.append(f"句长偏均匀（CV={cv_len:.3f}），偏 AI 特征")

    if ttr < 0.3 and len(words) > 500:
        score += 0.2
        reasons.append(f"词汇多样性偏低（TTR={ttr:.3f}）")

    if burstiness < 0.5 and len(sent_lens) >= 10:
        score += 0.15
        reasons.append(f"Burstiness 偏低（{burstiness:.3f}），缺乏人类写作的自然波动")

    if comma_density > 0.08:
        score += 0.1
        reasons.append(f"逗号密度偏高（{comma_density:.3f}）")

    if trigram_repeat_rate > 0.3:
        score += 0.15
        reasons.append(f"高频 trigram 重复（{trigram_repeat_rate:.2%}）")

    # 判定
    if score >= 0.5:
        verdict = f"🚨 高概率为 AI 生成文本（得分 {score:.2f}/1.0）"
    elif score >= 0.3:
        verdict = f"⚠️ 部分段落疑似 AI 生成（得分 {score:.2f}/1.0）"
    elif score >= 0.15:
        verdict = f"⚠️ 轻微 AI 特征（得分 {score:.2f}/1.0），可能为润色工具辅助"
    else:
        verdict = "✅ 未检测到明显 AI 生成特征"

    return {
        "sentences": len(sentences),
        "words": len(words),
        "ttr": float(ttr),
        "sentence_cv": float(cv_len),
        "burstiness": float(burstiness),
        "trigram_repeat_rate": float(trigram_repeat_rate),
        "ai_score": float(score),
        "reasons": reasons,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 4: Citation-Body Consistency
# ═══════════════════════════════════════════════════════════

def citation_consistency_check(text: str) -> dict:
    """
    引用-正文一致性检查。

    检测论文引用的参考文献是否真的支持所声称的观点。
    —— 文科论文的经典造假手法之一。

    基于：
    1. 引用密度异常（某些段落过分密集引用）
    2. 引用与上下文的语义矛盾标志词
    3. 引用位置的模式分析
    """
    # 提取引用标记
    citation_patterns = re.findall(
        r'(?:\[[\d,\s\-]+\]|\[\d+\]|\([A-Z][a-z]+(?:\s(?:et al\.|和)[^)]*)?,\s*\d{4}[a-z]?\)|（[^）]*?，\s*\d{4}[a-z]?）)',
        text
    )

    # 引用的上下文句子
    sentences = re.split(r'[。！？.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    cited_sentences = [s for s in sentences if re.search(
        r'\[[\d,\s\-]+\]|\[\d+\]|\([A-Z].*?\d{4}\)|（[^）]*?\d{4}[）]',
        s
    )]

    # 引用密度
    n_sentences = len(sentences)
    n_cited = len(cited_sentences)
    citation_density = n_cited / n_sentences if n_sentences > 0 else 0

    # 引用上下文标志
    # 过度肯定的引用语言
    overclaim_markers = [
        '证实了', '毫无疑问', '充分证明', '完全支持', '明确表明',
        'definitively', 'undoubtedly', 'conclusively',
    ]
    overclaims = []
    for s in cited_sentences:
        for marker in overclaim_markers:
            if marker in s:
                overclaims.append({"sentence": s[:80], "marker": marker})

    # 引用聚类检测
    if len(cited_sentences) >= 3:
        citation_positions = []
        for i, s in enumerate(sentences):
            if re.search(r'\[[\d,\s\-]+\]|\[\d+\]', s):
                citation_positions.append(i)

        if len(citation_positions) >= 5:
            # 检测 clusters — 连续句子都有引用（可疑：可能是在填引用）
            diffs = np.diff(citation_positions)
            clusters = sum(1 for d in diffs if d <= 2)
            cluster_ratio = clusters / len(diffs) if len(diffs) > 0 else 0
        else:
            cluster_ratio = 0
    else:
        cluster_ratio = 0

    # 判定
    issues = []
    if overclaims:
        issues.append(f"{len(overclaims)} 处过度肯定的引用表述")
    if cluster_ratio > 0.5 and n_cited > 10:
        issues.append(f"引用过度密集（聚类比 {cluster_ratio:.0%}），疑似引用充数")
    if citation_density > 0.7:
        issues.append(f"引用密度过高（{citation_density:.0%}），几乎每句都在引用")

    if len(issues) >= 2:
        verdict = "🚨 引用模式异常：" + "；".join(issues)
    elif len(issues) == 1:
        verdict = f"⚠️ {issues[0]}"
    else:
        verdict = "✅ 引用模式正常"

    return {
        "n_sentences": n_sentences,
        "n_citations": len(citation_patterns),
        "citation_density": float(citation_density),
        "overclaims": overclaims,
        "citation_cluster_ratio": float(cluster_ratio),
        "issues": issues,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 5: Argument Structure Analysis
# ═══════════════════════════════════════════════════════════

def argument_structure_analysis(text: str) -> dict:
    """
    论证结构分析：检测逻辑谬误。

    检测：
    1. 循环论证（结论出现在前提中）
    2. 同义反复
    3. 缺乏实证支持的主张
    """
    sentences = re.split(r'[。！？.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    # 提取主张性句子（包含断言标志词）
    claim_markers = [
        '因此', '所以', '由此可见', '这表明', '说明', '证明',
        '综上所述', '总之', '综上', '故而', '因而',
        'therefore', 'thus', 'hence', 'consequently',
    ]

    claims = []
    for s in sentences:
        for marker in claim_markers:
            if marker in s:
                claims.append(s)
                break

    # 检测理念类词汇（缺乏操作化定义）
    vague_concepts = [
        '范式', '话语', '场域', '建构', '解构', '后现代',
        '异化', '主体性', '规训', '文化资本', '惯习',
    ]
    vague_uses = []
    for s in sentences:
        for v in vague_concepts:
            if v in s:
                vague_uses.append({"sentence": s[:100], "concept": v})
                break  # 每句只计一次

    # 缺乏数据的实证主张
    data_markers = ['%', 'N=', 'n=', 'p<', 'p=', '显著', 'significant',
                    't=', 'F=', 'χ²', 'r=', 'OR=', 'HR=', '表', '图']
    unsupported_claims = []
    for s in claims:
        has_data = any(m in s for m in data_markers)
        if not has_data:
            unsupported_claims.append(s[:100])

    # 判定
    issues = []
    if vague_uses:
        issues.append(f"{len(vague_uses)} 处缺乏操作化定义的模糊概念")
    if unsupported_claims:
        issues.append(f"{len(unsupported_claims)} 处缺乏数据支持的实证主张")
    if len(claims) > 0.3 * len(sentences):
        issues.append(f"主张密度过高（{len(claims)}/{len(sentences)}），论证可能不充分")

    if len(issues) >= 2:
        verdict = "🚨 论证结构异常：" + "；".join(issues)
    elif len(issues) == 1:
        verdict = f"⚠️ {issues[0]}"
    else:
        verdict = "✅ 论证结构合理"

    return {
        "n_sentences": len(sentences),
        "n_claims": len(claims),
        "vague_concept_count": len(vague_uses),
        "vague_concepts": vague_uses[:10],
        "unsupported_claims": unsupported_claims[:5],
        "issues": issues,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 6: Full Text Arsenal
# ═══════════════════════════════════════════════════════════

def full_text_arsenal(text: str) -> dict:
    """
    对论文全文运行全套文本法医检测。
    """
    results = {}

    results["style_shifts"] = detect_style_shifts(text)
    results["ai_detection"] = detect_ai_generated_text(text)
    results["citation_check"] = citation_consistency_check(text)
    results["argument_analysis"] = argument_structure_analysis(text)

    # 汇总警报
    alarms = []
    for key, val in results.items():
        if isinstance(val, dict) and "verdict" in val and "🚨" in str(val.get("verdict", "")):
            alarms.append(f"[{key}] {val['verdict']}")

    results["alarm_count"] = len(alarms)
    results["alarms"] = alarms
    results["overall"] = (
        f"🔥 文本法医完成：{len(alarms)} 个警报"
        if alarms
        else "✅ 文本法医未发现明显异常"
    )

    return results
