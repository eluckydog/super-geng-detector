# 超级耿同学（The Super Geng）

学术论文多维度异常检测框架。

不是简单给出「有问题/没问题」的二值判断，而是用贝叶斯证据网络对每条证据建模，输出**带不确定性的后验造假概率**。

```
           ┌────────────────────────────┐
           │     超级耿同学 v1.0        │
           └──────────┬─────────────────┘
                      │
     ┌────────────────┴─────────────────┐
     │                                  │
  理工科引擎                        文科引擎
  ┌──────────────┐                ┌──────────────┐
  │ Benford 联合  │                │ HMM 文本溯源  │
  │ 扩展 GRIM     │                │ 因果图校验    │
  │ p-curve       │                │ 论证结构分析  │
  │ PRNU 指纹     │                │ 引用一致性    │
  │ Copy-Move     │                │ AI 生成检测   │
  └──────┬───────┘                └──────┬───────┘
         └──────────────┬───────────────┘
                        │
              ┌─────────▼─────────┐
              │  证据链合成引擎    │
              │  Bayes Net         │
              │  Shapley 贡献度   │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │  后验造假概率      │
              │  + 95% HDI        │
              │  + Bayes Factor    │
              └───────────────────┘
```

---

## 安装

```bash
git clone https://github.com/eluckydog/super-geng-fraud-detector.git
cd super-geng-fraud-detector
pip install -r requirements.txt
```

可选增强（贝叶斯 MCMC）：

```bash
pip install numpyro jax
```

---

## 快速开始

```bash
# 自动分类论文类型并检测
python scripts/super_geng.py detect paper.pdf --auto

# 指定模式
python scripts/super_geng.py detect paper.pdf --mode stem
python scripts/super_geng.py detect paper.pdf --mode humanities

# 独立模块
python scripts/super_geng.py stats data.csv       # 数值法医
python scripts/super_geng.py image figure.png    # 图像法医
python scripts/super_geng.py image a.png --compare b.png  # 两图 PRNU 比对
python scripts/super_geng.py text paper.txt       # 文本法医
```

---

## 检测维度

### 理工科（7 维）

| 维度 | 检测目标 | 核心方法 |
|------|----------|----------|
| 数据分布 | Benford 偏离 | 全位 KS + Anderson-Darling 联合检验 |
| 粒度冲突 | mean±SD 与 n 不兼容 | 扩展 GRIM + MCMC 后验预测 |
| SD 共谋 | 多组 SD 完全相同 | 分层贝叶斯方差模型 |
| 选择性报告 | p-hacking | p-curve 右偏 + Fisher 合并 |
| 列差值恒定 | 数据人为平移 | 贝叶斯变点检测 |
| 图像复用 | 同图不同实验 | PRNU 传感器指纹 + ORB 特征匹配 |
| 图像拼接 | 多源合成 | 噪声方差一致性 + JPEG 双重压缩 |

### 文科（5 维）

| 维度 | 检测目标 | 核心方法 |
|------|----------|----------|
| 风格突变 | 多人拼凑 / 润色痕迹 | HMM 状态推断 + 滑动窗口特征 |
| 因果谬误 | 方法论不支持结论 | DAG 构建 + d-separation + 后门准则 |
| 引用虚设 | 参考文献不支撑观点 | 引用密度异常 + 过度肯定检测 |
| 论证空转 | 循环论证 | 主张-证据覆盖率 + 模糊概念检测 |
| AI 生成 | LLM 代写 | TTR + 句长 CV + Burstiness |

---

## 证据链合成

各维度输出通过贝叶斯网络聚合，而非简单相加：

1. **各证据 → log LR**：在条件独立假设下求和
2. **后验造假概率**：P(fraud | evidence)，默认先验 5%
3. **Bayes Factor**：BF₁₀ > 10 为强证据，> 100 为极强证据
4. **Shapley 值**：量化每条证据对最终结论的边际贡献
5. **蒙特卡洛 HDI**：95% 最高密度区间，表达不确定性

输出示例（已撤稿论文 PLOS ONE e0313446 实测）：

```
后验造假概率：86.8%  [95% HDI: 71.2% ~ 94.1%]
Bayes Factor：125.3（极强证据）
关键证据：
  - 跨图数据复制（Fig1E vs Fig5，18/18 组逐字节相同）
  - 写作风格突变（HMM 状态数 4，异常转移概率 0.73）
```

---

## 模块结构

```
scripts/
├── super_geng.py          # CLI 主入口
├── stats_forensics.py     # 数值法医（Benford / GRIM / p-curve / 末位）
├── bayesian_checks.py     # 贝叶斯检验（MCMC / 后验预测 / BF）
├── image_forensics.py     # 图像法医（PRNU / 噪声 / JPEG / Copy-Move）
├── text_forensics.py      # 文本法医（HMM / AI 检测 / 风格分析）
├── causal_inference.py    # 因果推断（DAG / d-separation / 后门准则）
└── evidence_engine.py     # 证据合成（Bayes Net / BMA / Shapley）
```

---

## 设计原则

| 原则 | 说明 |
|------|------|
| LLM 负责推理 | 论文分类、数据提取、上下文综合 |
| 工具负责计算 | 像素比对、MCMC 采样、统计检验、DAG 分析 |
| 贝叶斯统一证据 | 输出带不确定性的后验概率，不做二值断言 |
| 可复现 | 固定随机种子，输出原始中间结果 |

---

## 已知局限

1. PRNU 检测需要高分辨率未压缩原图（JPEG 压缩会削弱传感器指纹信号）
2. 贝叶斯模型对先验分布敏感（默认造假率 5%，可按领域调整）
3. HMM 文本检测需要 ≥ 2000 字
4. PDF 扫描件需先 OCR 为可提取文本
5. **本工具输出统计证据，不构成学术不端的最终认定**——最终判断需由专业机构做出

---

## 测试用例

`test_cases/pone_0313446/` 包含一份已撤稿 PLOS ONE 论文的完整检测记录：

- 论文 PDF + 补充数据（Excel）
- 跨图数据复制检测结果（Fig1E/Fig5、Fig2B/Fig6 逐字节相同）
- 后验造假概率 86.8%，BF = 125

运行：

```bash
cd test_cases/pone_0313446
python run_detection.py
```

---

## 项目由来

本项目受到 [wooly99/geng-academic-fraud-detector](https://github.com/wooly99/geng-academic-fraud-detector) 的启发。

但在实现路径上，我们基于**对 OpenClaw Skill 工作机制的重新理解**，完全重写了所有检测模块：

- wooly99 的原版以 prompt 模板为核心，依赖 LLM 的语义理解完成检测；
- 本项目的定位是**专业用于初步检查论文的工具**——每个检测维度均有对应的计算模块（NumPyro MCMC、HMM、Benford 联合检验、PRNU 传感器指纹等），LLM 仅负责推理调度与证据综合。

各模块均为从零实现，不依赖任何现有模板：
- `stats_forensics.py`：Benford 全位联合检验（KS + AD），优于单一 Benford 检验
- `bayesian_checks.py`：NumPyro MCMC 驱动的分层贝叶斯 SD 一致性模型
- `text_forensics.py`：HMM 滑动窗口风格状态推断，支持中英文
- `evidence_engine.py`：贝叶斯网络 + Shapley 值证据合成

> 这不是一个「调用 LLM 打假」的项目，而是一个**用计算工具代替人眼做初步筛查**的框架。

---

## 许可

MIT License
