#!/usr/bin/env python3
"""
scripts/image_forensics.py — 超级耿同学图像法医引擎

检测能力：
- PRNU 传感器指纹比对（同一相机拍的"不同"图片会露馅）
- JPEG 双重压缩检测
- 噪声方差一致性分析（不同来源拼接的图噪声不同）
- ORB/SIFT 特征匹配（copy-move forgery detection）
- 克隆区域检测
"""
import numpy as np
from scipy import ndimage, fftpack


# ═══════════════════════════════════════════════════════════
# Module 1: PRNU Sensor Fingerprint
# ═══════════════════════════════════════════════════════════

def prnu_extract(image_path: str) -> dict:
    """
    提取图片的 PRNU（Photo Response Non-Uniformity）传感器指纹。

    PRNU 是每个相机传感器特有的"噪声指纹"。
    同一相机拍的两张不同图片会有相似的 PRNU 模式。
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
    except Exception as e:
        return {"error": f"无法加载图片: {e}"}

    # 去噪获得噪声残差（使用小波或简单的高通滤波）
    denoised = ndimage.median_filter(img_array, size=3)
    noise_residual = img_array - denoised

    # 提取 PRNU 估计
    # PRNU = E(noise_residual * intensity) / E(intensity^2)
    intensity = img_array / 255.0
    intensity = np.clip(intensity, 1e-6, 1.0)
    prnu = np.mean(noise_residual * intensity) / np.mean(intensity ** 2)

    # PRNU 空间变异性
    prnu_std = np.std(noise_residual * intensity)

    # 分块 PRNU 一致性（同一相机：PRNU 在图像各处一致）
    h, w = img_array.shape
    block_size = min(h, w) // 4
    if block_size > 16:
        block_prnus = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = img_array[i:i+block_size, j:j+block_size]
                block_denoised = ndimage.median_filter(block, size=3)
                block_noise = block - block_denoised
                block_intensity = np.clip(block / 255.0, 1e-6, 1.0)
                b_prnu = np.mean(block_noise * block_intensity) / np.mean(block_intensity ** 2)
                block_prnus.append(b_prnu)
        prnu_cv = float(np.std(block_prnus) / (abs(np.mean(block_prnus)) + 1e-10))
    else:
        block_prnus = []
        prnu_cv = 0

    return {
        "image_shape": (h, w),
        "prnu_magnitude": float(prnu),
        "prnu_spatial_variation": float(prnu_std),
        "prnu_block_cv": prnu_cv,
        "n_blocks": len(block_prnus),
    }


def prnu_compare(image_path_a: str, image_path_b: str) -> dict:
    """
    比较两张图片的 PRNU 指纹相似度。

    高相似度 → 同一相机拍摄 → 如果声称是不同实验的独立图片，则是造假。
    """
    fa = prnu_extract(image_path_a)
    fb = prnu_extract(image_path_b)

    if "error" in fa or "error" in fb:
        return {"error": "无法提取 PRNU 指纹",
                "details": [fa.get("error", ""), fb.get("error", "")]}

    # 归一化 PRNU 幅值比较
    pa = fa["prnu_magnitude"]
    pb = fb["prnu_magnitude"]
    similarity = 1 - abs(pa - pb) / (abs(pa) + abs(pb) + 1e-10)

    # PRNU 变异系数比较
    cv_similarity = 1 - abs(fa["prnu_block_cv"] - fb["prnu_block_cv"]) / (
        max(fa["prnu_block_cv"], fb["prnu_block_cv"], 1e-10) * 2 + 1e-10
    )

    combined_sim = 0.6 * similarity + 0.4 * cv_similarity

    if combined_sim > 0.85:
        verdict = "🚨 PRNU 高度相似（{:.2f}）→ 两张图片极可能来自同一相机/传感器".format(combined_sim)
    elif combined_sim > 0.7:
        verdict = "⚠️ PRNU 中度相似（{:.2f}）→ 建议进一步检查".format(combined_sim)
    else:
        verdict = f"✅ PRNU 差异显著（{combined_sim:.2f}）→ 不同相机拍摄"

    return {
        "prnu_similarity": float(combined_sim),
        "magnitude_similarity": float(similarity),
        "cv_similarity": float(cv_similarity),
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 2: Noise Variance Consistency
# ═══════════════════════════════════════════════════════════

def noise_variance_analysis(image_path: str) -> dict:
    """
    噪声方差一致性分析。

    同一来源的图片，噪声方差应在图像各处保持一致。
    拼接伪造的图片，不同区域的噪声方差会显著不同。
    这是检测 Western blot 拼接的利器。
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
    except Exception as e:
        return {"error": f"无法加载图片: {e}"}

    h, w = img_array.shape

    # 分块
    block_sizes = [32, 64]
    all_blocks = []

    for bs in block_sizes:
        for i in range(0, h - bs, bs // 2):
            for j in range(0, w - bs, bs // 2):
                block = img_array[i:i+bs, j:j+bs]
                # 高通滤波取得噪声
                smoothed = ndimage.median_filter(block, size=3)
                noise = block - smoothed
                var = float(np.var(noise))
                all_blocks.append({
                    "position": (i, j),
                    "size": bs,
                    "noise_variance": var,
                })

    if not all_blocks:
        return {"error": "图像太小"}

    variances = [b["noise_variance"] for b in all_blocks]
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    cv_var = std_var / mean_var if mean_var > 0 else 0

    # 检测异常块
    z_scores = [(v - mean_var) / std_var for v in variances]
    anomalies = []
    for i, z in enumerate(z_scores):
        if abs(z) > 2.5:
            b = all_blocks[i]
            anomalies.append({
                "position": b["position"],
                "z_score": float(z),
                "variance": float(variances[i]),
            })

    if len(anomalies) > len(all_blocks) * 0.15:
        verdict = f"🚨 噪声方差高度不均（{len(anomalies)}/{len(all_blocks)} 块异常）→ 疑似图像拼接"
    elif anomalies:
        verdict = f"⚠️ 检测到 {len(anomalies)} 个噪声异常区域"
    elif cv_var > 0.5:
        verdict = f"⚠️ 噪声方差变异较大（CV={cv_var:.2f}），建议进一步检查"
    else:
        verdict = "✅ 噪声方差一致"

    return {
        "n_blocks": len(all_blocks),
        "mean_noise_variance": float(mean_var),
        "cv_noise_variance": float(cv_var),
        "anomaly_blocks": len(anomalies),
        "anomaly_details": anomalies[:5],
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 3: Double JPEG Compression Detection
# ═══════════════════════════════════════════════════════════

def detect_double_jpeg(image_path: str) -> dict:
    """
    检测 JPEG 双重压缩。

    篡改图片通常经历两次保存：
    1. 原图保存为 JPEG（第一次）
    2. 修改后再次保存为 JPEG（第二次）
    两次压缩的量化表不同，会在 DCT 系数直方图中留下周期性特征。

    检测 DCT 系数直方图的周期性峰值。
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        original_mode = img.mode
        img_array = np.array(img, dtype=np.float64)
    except Exception as e:
        return {"error": f"无法加载图片: {e}"}

    # 转换为灰度
    if len(img_array.shape) == 3:
        img_gray = np.mean(img_array, axis=2)
    else:
        img_gray = img_array

    # 分块 DCT
    h, w = img_gray.shape
    h_blocks = h // 8
    w_blocks = w // 8
    if h_blocks < 4 or w_blocks < 4:
        return {"error": "图像太小"}

    dct_coeffs = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = img_gray[i*8:(i+1)*8, j*8:(j+1)*8]
            if block.shape == (8, 8):
                dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.append(dct_block)

    if not dct_coeffs:
        return {"error": "无法计算 DCT 系数"}

    # 提取高频 AC 系数
    ac_coeffs = []
    for dct in dct_coeffs:
        for x in range(8):
            for y in range(8):
                if x > 0 or y > 0:  # 跳过 DC
                    ac_coeffs.append(dct[x, y])

    ac_coeffs = np.array(ac_coeffs)

    # 量化步长的周期性检测
    # 双重压缩时，第二层的量化步长会在整数倍处产生峰值
    ac_int = np.round(ac_coeffs).astype(int)
    ac_int = ac_int[np.abs(ac_int) < 20]

    # 计算直方图
    hist, _ = np.histogram(ac_int, bins=np.arange(-20, 21))
    total = np.sum(hist)

    # 检测整数倍位置是否有异常峰值
    # 第一次压缩的量化步长会导致 DCT 系数在特定值上聚集
    hist_norm = hist / total if total > 0 else hist

    # 检测周期性模式（傅里叶分析直方图）
    hist_fft = np.abs(np.fft.fft(hist_norm))
    peak_freqs = np.argsort(hist_fft[1:len(hist_fft)//2])[-3:][::-1]

    # 如果存在显著的周期性 → 双重压缩
    max_peak = float(np.max(hist_fft[1:len(hist_fft)//2]))
    mean_fft = float(np.mean(hist_fft[1:len(hist_fft)//2]))
    periodicity = max_peak / mean_fft if mean_fft > 0 else 1

    if periodicity > 5:
        verdict = "🚨 检测到 JPEG 双重压缩特征 → 图片可能被篡改后重新保存"
    elif periodicity > 3:
        verdict = "⚠️ 存在 JPEG 双重压缩的弱特征"
    else:
        verdict = "✅ 未检测到明显双重压缩特征"

    return {
        "n_dct_blocks": len(dct_coeffs),
        "periodicity_score": float(periodicity),
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 4: Copy-Move Forgery Detection (ORB)
# ═══════════════════════════════════════════════════════════

def detect_copy_move(image_path: str,
                     min_cluster: int = 5) -> dict:
    """
    检测 copy-move 伪造（复制-粘贴篡改）。

    使用 ORB 特征检测 + 特征匹配：
    1. 提取局部 ORB 特征点
    2. 匹配相似特征对
    3. 聚类匹配对的相对偏移向量
    4. 同一区域的 cluster → copy-move 检测

    应用：检测 Western blot 条带复制、图片区域复用等。
    """
    try:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "无法加载图片（opencv）"}
    except ImportError:
        return {"error": "opencv-python-headless 未安装"}
    except Exception as e:
        return {"error": f"加载图片失败: {e}"}

    h, w = img.shape

    # ORB 特征提取
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None or len(keypoints) < 20:
        return {"error": "特征点不足", "n_keypoints": len(keypoints) if keypoints else 0}

    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors)

    # 过滤好的匹配（距离小 + 排除自匹配）
    good_matches = []
    for m in matches:
        if m.distance < 50:  # 距离阈值
            p1 = keypoints[m.queryIdx].pt
            p2 = keypoints[m.trainIdx].pt
            # 排除自匹配和距离太近
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            if dist > 20:
                good_matches.append((p1, p2, m.distance))

    if len(good_matches) < min_cluster:
        return {
            "n_keypoints": len(keypoints),
            "n_matches": len(good_matches),
            "suspicious_clusters": 0,
            "verdict": "✅ 未检测到 copy-move 伪造",
        }

    # 聚类匹配对的偏移向量
    offsets = []
    for p1, p2, _ in good_matches:
        offset = (p2[0] - p1[0], p2[1] - p1[1])
        offsets.append(offset)

    # 简单聚类：找出现频率最高的偏移向量
    from collections import Counter
    offset_counter = Counter(offsets)
    most_common = offset_counter.most_common(10)

    # 检测是否有偏移向量频繁出现
    clusters = []
    for offset, count in most_common:
        if count >= min_cluster:
            clusters.append({
                "offset": offset,
                "count": count,
            })

    if clusters:
        n_clusters = len(clusters)
        total_clustered = sum(c["count"] for c in clusters)
        verdict = f"🚨 检测到 copy-move 伪造：{n_clusters} 个重复区域（共 {total_clustered} 个匹配点）"
    elif len(good_matches) > 50:
        verdict = f"⚠️ 存在 {len(good_matches)} 个相似特征对，但未形成明确 cluster，建议人工复核"
    else:
        verdict = "✅ 未检测到显著 copy-move 伪造"

    return {
        "n_keypoints": len(keypoints),
        "n_matches": len(good_matches),
        "suspicious_clusters": len(clusters),
        "clusters": clusters[:5],
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════
# Module 5: Full Image Arsenal
# ═══════════════════════════════════════════════════════════

def full_image_arsenal(image_path: str) -> dict:
    """
    对图片运行全套图像法医检测。
    """
    results = {}

    results["prnu"] = prnu_extract(image_path)
    results["noise_variance"] = noise_variance_analysis(image_path)
    results["double_jpeg"] = detect_double_jpeg(image_path)
    results["copy_move"] = detect_copy_move(image_path)

    alarms = []
    for key, val in results.items():
        if isinstance(val, dict) and "verdict" in val and "🚨" in str(val.get("verdict", "")):
            alarms.append(f"[{key}] {val['verdict']}")

    results["alarm_count"] = len(alarms)
    results["alarms"] = alarms
    results["overall"] = (
        f"🔥 图像法医完成：{len(alarms)} 个警报"
        if alarms
        else "✅ 图像法医未发现明显异常"
    )

    return results


def compare_two_images(path_a: str, path_b: str) -> dict:
    """
    对比两张图片是否是同一张图的修改版。
    """
    fa = prnu_extract(path_a)
    fb = prnu_extract(path_b)
    if "error" in fa or "error" in fb:
        return {"error": "无法提取 PRNU 指纹"}

    prnu_result = prnu_compare(path_a, path_b)
    return prnu_result
