---
name: super-geng-detector
description: '学术论文多维度打假检测器。双引擎架构（理工科数值/图像法医 + 文科文本/方法论校验），贝叶斯证据链合成输出后验造假概率。当用户要求"打假"、"查重"、"论文检测"、"数据验证"、"方法审核"时使用。'
license: MIT
version: 1.0.0
---

# 超级耿同学（The Super Geng）

多引擎学术论文异常检测框架。不是「有问题/没问题」的二值判断，而是用贝叶斯证据网络给出带不确定性的后验造假概率。

## 架构

```
                    ┌──────────────────────┐
                    │   超级耿同学 v1.0    │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼──────────┐      ┌──────────▼──────────┐
     │  理工科引擎 (STEM)    │      │  文科引擎 (Humanities) │
     ├─────────────────────┤      ├─────────────────────┤
     │ • Benford 全位联合   │      │ • HMM 文本风格溯源    │
     │ • 扩展 GRIM 检验     │      │ • 因果图方法论校验    │
     │ • p-curve 选择性报告 │      │ • 论证结构分析        │
     │ • SD 贝叶斯一致性    │      │ • 引用-正文一致性     │
     │ • PRNU 传感器指纹    │      │ • AI 生成文本识别     │
     │ • JPEG 双重压缩      │      │ • 语义模糊概念检测    │
     │ • Copy-Move 检测     │      │                      │
     └──────────┬──────────┘      └──────────┬──────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  证据链合成引擎        │
                    │  Bayes Net + Shapley  │
                    │  后验概率 + 95% HDI   │
                    │  Bayes Factor          │
                    └────────────────────────┘
```

## 快速开始

```bash
# 自动分类论文类型并检测
python scripts/super_geng.py detect paper.pdf --auto

# 指定模式
python scripts/super_geng.py detect paper.pdf --mode stem
python scripts/super_geng.py detect paper.pdf --mode humanities

# 独立模块
python scripts/super_geng.py stats data.csv       # 数值法医
python scripts/super_geng.py image figure.png     # 图像法医
python scripts/super_geng.py text paper.txt        # 文本法医
```

## 检测维度

### 理工科（7 维）

| 维度 | 目标 | 方法 | 模块 |
|------|------|------|------|
| 数据分布异常 | Benford 偏离 | 全位 KS + AD 联合检验 | `stats_forensics.py` |
| 粒度冲突 | mean±SD 与 n 不兼容 | 扩展 GRIM + 后验预测 | `bayesian_checks.py` |
| SD 共谋 | 多组 SD 完全相同 | 分层贝叶斯方差模型 (MCMC) | `bayesian_checks.py` |
| 选择性报告 | p-hacking | p-curve 右偏 + Fisher 合并 | `stats_forensics.py` |
| 列差值恒定 | 数据人为平移 | 贝叶斯变点检测 | `bayesian_checks.py` |
| 图像复用 | 同图不同实验 | PRNU 指纹 + ORB 特征匹配 | `image_forensics.py` |
| 图像拼接 | 多源合成 | 噪声方差一致性 + JPEG 双重压缩 | `image_forensics.py` |

### 文科（5 维）

| 维度 | 目标 | 方法 | 模块 |
|------|------|------|------|
| 风格突变 | 多人拼凑 / 润色痕迹 | HMM 状态推断 + 滑动窗口特征 | `text_forensics.py` |
| 因果谬误 | 方法论不支持结论 | DAG 构建 + d-separation | `causal_inference.py` |
| 引用虚设 | 参考文献不支撑观点 | 引用密度异常 + 过度肯定检测 | `text_forensics.py` |
| 论证空转 | 循环论证 | 主张-证据覆盖率 + 模糊概念 | `text_forensics.py` |
| AI 生成 | LLM 代写 | TTR + 句长 CV + Burstiness | `text_forensics.py` |

## 证据链合成

各维度输出不直接相加，而是通过贝叶斯网络建模条件依赖：

1. **各证据 → log LR**：在条件独立假设下求和
2. **后验造假概率**：P(fraud | evidence)，先验默认 5%
3. **Bayes Factor**：BF₁₀ > 10 强证据，> 100 极强
4. **Shapley 值**：每条证据对最终结论的边际贡献
5. **蒙特卡洛 HDI**：95% 最高密度区间，量化不确定性

输出示例：
```
后验造假概率：86.8% [95% HDI: 71.2% ~ 94.1%]
Bayes Factor：125.3（极强证据）
关键证据：跨图数据复制（Fig1E vs Fig5，18/18 组逐字节相同）
```

## 安装

```bash
pip install -r requirements.txt
```

核心依赖（必需）：`numpy`, `scipy`, `pillow`, `opencv-python-headless`, `hmmlearn`, `PyPDF2`

增强依赖（可选）：`numpyro`, `jax`（贝叶斯 MCMC，缺省时自动降级为解析近似）

## 模块清单

| 文件 | 功能 | 核心技术 |
|------|------|----------|
| `super_geng.py` | CLI 主入口 | 论文分类 + 引擎调度 + 报告生成 |
| `stats_forensics.py` | 数值法医 | Benford 全位、GRIM 扩展、p-curve、末位检验、SD 模式 |
| `bayesian_checks.py` | 贝叶斯检验 | NumPyro MCMC、分层模型、后验预测、BF (Savage-Dickey) |
| `image_forensics.py` | 图像法医 | PRNU、噪声方差、JPEG 双重压缩、ORB Copy-Move |
| `text_forensics.py` | 文本法医 | HMM 风格状态、AI 生成检测、元话语分析、克隆检测 |
| `causal_inference.py` | 因果推断 | DAG、d-separation、后门准则、样本量校验 |
| `evidence_engine.py` | 证据合成 | Bayes Net、BMA、Shapley 分解、Monte Carlo HDI |

## 设计原则

1. **LLM 负责推理**：论文分类、数据提取、上下文综合判断
2. **工具负责计算**：像素比对、MCMC 采样、统计检验、DAG 分析
3. **贝叶斯统一证据**：输出带不确定性的后验概率，不做二值断言
4. **可复现**：固定随机种子，输出原始中间结果

## 已知局限

1. PRNU 检测需要高分辨率原图，JPEG 压缩会削弱传感器指纹
2. 贝叶斯模型对先验分布敏感（默认造假率 5%，可按领域调整）
3. HMM 文本检测需要 ≥2000 字
4. PDF 扫描件需先 OCR
5. **本工具输出统计证据，不构成学术不端的最终认定**——最终判断需由专业机构做出

## 测试用例

`test_cases/pone_0313446/` 包含一份已撤稿 PLOS ONE 论文的完整检测记录：
- 论文 PDF + 补充数据 Excel
- 跨图数据复制检测结果（Fig1E/Fig5、Fig2B/Fig6 逐字节相同）
- 后验造假概率 86.8%，BF=125

运行：`cd test_cases/pone_0313446 && python run_detection.py`

## 许可

MIT License
