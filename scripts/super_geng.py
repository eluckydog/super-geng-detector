#!/usr/bin/env python3
"""
super_geng.py — 超级耿同学学术打假检测器 主入口

用法：
    python super_geng.py detect paper.pdf --auto          # 自动检测论文类型
    python super_geng.py detect paper.pdf --mode stem     # 理工科模式
    python super_geng.py detect paper.pdf --mode humanities  # 文科模式
    python super_geng.py stats data.csv                   # 统计法医快速检测
    python super_geng.py image figure.png                 # 图像法医检测
    python super_geng.py text paper.txt                   # 文本法医检测

"战斗力只有5的渣渣论文，在我面前不堪一击。" —— 超级耿同学
"""
import sys
import os
import json
import argparse
from pathlib import Path

# 添加 scripts 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from stats_forensics import full_stats_arsenal
from text_forensics import full_text_arsenal
from image_forensics import full_image_arsenal
from causal_inference import validate_methodology
from evidence_engine import synthesize_evidence


# ═══════════════════════════════════════════════════════════
# 耿同学语录引擎
# ═══════════════════════════════════════════════════════════

GENG_QUOTES = {
    "benford_alarm": [
        "这数据分布，连基纽特战队都不会信。",
        "Benford 定律都过不了，你当这是那美克星的统计学？",
    ],
    "grim_alarm": [
        "标准差全是整数？你们实验室的移液器是不是赛亚人训练过的？",
        "均值粒度冲突——这不是数据，这是龟派气功震出来的随机数。",
    ],
    "sd_alarm": [
        "所有组 SD 完全相同？你当这是饺子用超能力催眠了方差吗。",
        "SD 值的整齐程度比弗利萨军的列队还统一。",
    ],
    "p_hacking_alarm": [
        "p 值全挤在 0.04-0.05，你这 p-hacking 技术连撒旦先生都骗不过。",
        "这么多刚好显著的 p 值？你是拿龙珠许愿了吧。",
    ],
    "image_alarm": [
        "同一张图旋转 180 度就是新实验？你以为你是天津饭的太阳拳？",
        "这张 blot 的拼接技术比撒旦先生的世界冠军还水。",
    ],
    "noise_alarm": [
        "不同区域的噪声差异堪比贝吉塔和琪琪的战力差。",
        "噪声模式不一致——这不是一张图，这是弗利萨拼出来的破烂。",
    ],
    "text_alarm": [
        "写作风格突变比超赛变身还频繁，你是请了几个人帮你写？",
        "这文本平滑得不像人写的，界王都没有你这么均匀的句长。",
    ],
    "causal_alarm": [
        "你想证明因果却连后门路径都没阻断——这是战斗前连气都没运好。",
        "中介当混淆，混淆当中介——你的因果推断是跟基纽学的吧。",
    ],
    "default_alarm": [
        "这篇论文的造假水平，战斗力探测器只显示 5。",
        "我一个退学的博士都能超三，你们这些杰青连气都不会运。",
        "这不是学术造假，这是弗利萨级别的学术破坏。",
        "界王拳 x20 的统计检验都救不了你的数据。",
    ],
    "clean": [
        "目前没发现明显问题——但别得意，我还在盯着你的 supplementary。",
        "战斗力暂时正常，但这可能是因为你还没进化到能让我检测出问题的水平。",
    ],
    "conclusion_high": [
        "战斗力探测器爆表了！这篇论文造假程度堪比超赛神。",
        "别以为地球人看不懂统计——本超级耿同学的探测器不会撒谎。",
    ],
    "conclusion_low": [
        "战斗力不佳，但至少比拉帝兹强。继续努力，地球人。",
        "还行，但别得意，下回我还能升级到超四。",
    ],
}

import random

def spout_quote(category: str, alarms: int = 0) -> str:
    """吐一句耿同学语录"""
    if category in GENG_QUOTES:
        return random.choice(GENG_QUOTES[category])
    if alarms > 3:
        return random.choice(GENG_QUOTES["default_alarm"])
    if alarms == 0:
        return random.choice(GENG_QUOTES["clean"])
    return random.choice(GENG_QUOTES["default_alarm"])


# ═══════════════════════════════════════════════════════════
# 论文分类器
# ═══════════════════════════════════════════════════════════

def classify_paper(text: str) -> str:
    """
    自动分类论文类型：stem 或 humanities。

    基于特征词比例。
    """
    stem_keywords = [
        'p<', 'p=', 'p <', 'P<', 'P=', 'N=', 'n=',
        '±', 'mean', 'SD', 'SEM', 't-test', 'ANOVA',
        'western blot', 'PCR', 'qPCR', 'ELISA', 'HPLC',
        'cell', 'μM', 'mM', 'nM', 'mg/ml', 'μg/ml',
        'Fig.', 'Figure', 'Table', 'supplementary',
        r'\chi^2', 'Student', 'Mann-Whitney', 'Kruskal',
        'regression', 'correlation', 'coefficient',
    ]
    humanities_keywords = [
        '话语', '范式', '解构', '后现代', '场域', '主体性',
        '意识形态', '文本分析', '深度访谈', '参与观察',
        '民族志', '叙事', '文化资本', '惯习', '规训',
        'discourse', 'narrative', 'hermeneutic', 'phenomenology',
        'grounded theory', 'qualitative', 'ethnography',
        'thus', 'therefore', 'consequently', 'hence',
    ]

    stem_count = sum(1 for kw in stem_keywords if kw.lower() in text.lower())
    humanities_count = sum(1 for kw in humanities_keywords if kw.lower() in text.lower())

    if stem_count >= 5 and stem_count >= humanities_count * 2:
        return "stem"
    elif humanities_count >= 3 and humanities_count >= stem_count * 2:
        return "humanities"

    # 数据密度判断
    numbers = sum(1 for c in text if c.isdigit())
    total = len(text)
    number_ratio = numbers / total if total > 0 else 0
    citations = text.count('[')
    citation_density = citations / max(total, 1) * 1000

    if number_ratio > 0.05:
        return "stem"
    elif citation_density > 5:
        return "humanities"

    return "stem"  # 默认理工科


# ═══════════════════════════════════════════════════════════
# 引擎驱动
# ═══════════════════════════════════════════════════════════

def run_stem_engine(text: str, data_extraction_fn=None) -> dict:
    """
    运行理工科引擎。

    参数:
    - text: 论文文本（从 PDF 提取）
    - data_extraction_fn: 可选，从文本中提取数值数据的函数
    """
    findings = []
    quotes = []

    # 第一式：Benford（需要数据）
    # 如果有数据提取函数，提取表格数据
    if data_extraction_fn:
        try:
            data = data_extraction_fn(text)
            if data:
                benford_result = {
                    "key": "benford_suite",
                    "category": "stem_numeric",
                }
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Benford data extraction failed: {e}")

    # 第九式：因果推断
    causal = validate_methodology(text)
    findings.append({
        "key": "causal_methodology",
        "category": "stem_method",
        **causal,
    })

    return {
        "findings": findings,
        "quotes": quotes,
    }


def run_humanities_engine(text: str) -> dict:
    """
    运行文科引擎。
    """
    findings = []
    quotes = []

    # HMM 文本法医
    text_result = full_text_arsenal(text)
    for key, val in text_result.items():
        if isinstance(val, dict) and "verdict" in val and key not in ("alarm_count", "alarms", "overall"):
            findings.append({
                "key": f"text_{key}",
                "category": "humanities_text",
                "verdict": val["verdict"],
            })
            if "🚨" in str(val.get("verdict", "")):
                quotes.append({
                    "type": "text_alarm",
                    "quote": spout_quote("text_alarm"),
                })

    # 因果推断
    causal = validate_methodology(text)
    findings.append({
        "key": "causal_methodology",
        "category": "humanities_method",
        "verdict": causal["verdict"],
    })
    if "🚨" in causal["verdict"]:
        quotes.append({
            "type": "causal_alarm",
            "quote": spout_quote("causal_alarm"),
        })

    return {
        "findings": findings,
        "quotes": quotes,
    }


# ═══════════════════════════════════════════════════════════
# 报告生成器
# ═══════════════════════════════════════════════════════════

def generate_report(paper_path: str, mode: str, stem_result: dict = None,
                    humanities_result: dict = None) -> str:
    """生成打假报告"""
    paper_name = os.path.basename(paper_path)

    all_findings = []
    if stem_result:
        all_findings.extend(stem_result.get("findings", []))
    if humanities_result:
        all_findings.extend(humanities_result.get("findings", []))

    synthesis = synthesize_evidence(all_findings)

    lines = []
    lines.append("=" * 60)
    lines.append("  🔥 超级耿同学学术打假检测报告")
    lines.append("=" * 60)
    lines.append(f"  论文: {paper_name}")
    lines.append(f"  检测模式: {'理工科引擎' if mode == 'stem' else '文科引擎' if mode == 'humanities' else '自动'}")
    lines.append("")

    # 综合评定
    lines.append("## 📊 综合评定")
    p = synthesis["posterior_fraud_probability"]
    bf = synthesis["bayes_factor"]
    hdi = synthesis["hdi_95"]
    lines.append(f"  后验造假概率: {p:.1%}")
    lines.append(f"  Bayes Factor: {bf:.1f} （{synthesis['bf_interpretation']}）")
    lines.append(f"  95% HDI: [{hdi[0]:.1%}, {hdi[1]:.1%}]")
    lines.append(f"  战斗力等级: {synthesis['verdict_level']}")
    lines.append("")

    # 证据项
    if all_findings:
        lines.append("## 🔍 检测发现")
        lines.append("")
        for finding in all_findings:
            key = finding.get("key", "unknown")
            verdict = finding.get("verdict", "无结论")
            lines.append(f"  [{key}] {verdict}")

    # Top 贡献
    if synthesis["top_evidence"]:
        lines.append("")
        lines.append("## 💪 最强证据（Shapley 值排名）")
        for name, pct in synthesis["top_evidence"]:
            lines.append(f"  • {name}: {pct:.1f}% 贡献度")

    # 辣评
    alarms = sum(1 for f in all_findings if "🚨" in str(f.get("verdict", "")))
    lines.append("")
    lines.append("## 💬 超级耿同学辣评")
    if alarms > 3:
        lines.append(f"  {spout_quote('conclusion_high')}")
    elif alarms > 0:
        lines.append(f"  {spout_quote('default_alarm')}")
    else:
        lines.append(f"  {spout_quote('conclusion_low')}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("  ⚠️ 本报告提供统计证据和概率判断，不提供「100% 确定」的结论。")
    lines.append("     学术不端的最终认定需要专业机构调查。")
    lines.append("     致敬耿洪伟。致敬鸟山明。")
    lines.append("=" * 60)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🔥 超级耿同学学术打假检测器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python super_geng.py detect paper.pdf --auto
  python super_geng.py detect paper.pdf --mode stem
  python super_geng.py stats data.csv
  python super_geng.py image figure.png
  python super_geng.py text paper.txt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # detect 命令
    detect_parser = subparsers.add_parser("detect", help="检测论文")
    detect_parser.add_argument("paper", help="论文 PDF 文件路径")
    detect_parser.add_argument("--mode", choices=["auto", "stem", "humanities"],
                              default="auto", help="检测模式")
    detect_parser.add_argument("--output", "-o", help="输出报告文件路径")
    detect_parser.add_argument("--json", action="store_true", help="JSON 格式输出")

    # stats 命令
    stats_parser = subparsers.add_parser("stats", help="数值法医快速检测")
    stats_parser.add_argument("data_file", help="CSV 数据文件")

    # image 命令
    image_parser = subparsers.add_parser("image", help="图像法医检测")
    image_parser.add_argument("image_file", help="图片文件路径")
    image_parser.add_argument("--compare", help="对比图片路径（PRNU 比对）")

    # text 命令
    text_parser = subparsers.add_parser("text", help="文本法医检测")
    text_parser.add_argument("text_file", help="文本文件路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "detect":
        # 读取论文
        paper_path = args.paper
        if not os.path.exists(paper_path):
            print(f"❌ 文件不存在: {paper_path}")
            sys.exit(1)

        print(f"📄 读取论文: {paper_path}")

        # 尝试读取 PDF
        text = ""
        try:
            import subprocess
            # 用 pdftotext 提取文本
            result = subprocess.run(
                ["pdftotext", paper_path, "-"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                text = result.stdout
            else:
                # 尝试 PyPDF2
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(paper_path)
                    text = "\n".join(
                        page.extract_text() or ""
                        for page in reader.pages
                    )
                except ImportError:
                    pass
        except Exception as e:
            print(f"[WARN] PDF text extraction failed: {e}")

        if not text or len(text) < 100:
            print("⚠️ 无法提取文本，请确保已安装 pdftotext 或 PyPDF2")
            print("  将跳过文本分析，仅对可提取数据运行检测")
            text = ""

        # 分类
        if args.mode == "auto":
            mode = classify_paper(text) if text else "stem"
            print(f"🔍 自动分类: {mode}")
        else:
            mode = args.mode

        # 运行引擎
        stem_result = None
        humanities_result = None

        if mode == "stem":
            print("⚙️ 运行理工科引擎...")
            stem_result = run_stem_engine(text)
        elif mode == "humanities":
            print("⚙️ 运行文科引擎...")
            humanities_result = run_humanities_engine(text)

        # 生成报告
        report = generate_report(paper_path, mode, stem_result, humanities_result)

        if args.json:
            syn = synthesize_evidence(
                (stem_result or {}).get("findings", []) +
                (humanities_result or {}).get("findings", [])
            )
            print(json.dumps(syn, indent=2, ensure_ascii=False))
        else:
            print(report)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n📝 报告已保存至: {args.output}")

    elif args.command == "stats":
        import csv
        if not os.path.exists(args.data_file):
            print(f"❌ 文件不存在: {args.data_file}")
            sys.exit(1)

        with open(args.data_file, encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            print("❌ 空文件")
            return

        # 提取数值列
        import numpy as np
        data_columns = []
        header = rows[0]
        n_cols = len(header)

        for col in range(n_cols):
            values = []
            for row in rows[1:]:
                if col < len(row):
                    try:
                        values.append(float(row[col]))
                    except (ValueError, TypeError):
                        pass
            if len(values) >= 5:
                data_columns.append(values)

        if not data_columns:
            print("❌ 未找到数值列")
            return

        print(f"📊 找到 {len(data_columns)} 列数值数据")
        from stats_forensics import full_stats_arsenal

        # 构造数据表
        table = [{"values": col} for col in data_columns]
        result = full_stats_arsenal(table)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "image":
        from image_forensics import full_image_arsenal, compare_two_images

        if args.compare:
            result = compare_two_images(args.image_file, args.compare)
        else:
            result = full_image_arsenal(args.image_file)

        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "text":
        if not os.path.exists(args.text_file):
            print(f"❌ 文件不存在: {args.text_file}")
            sys.exit(1)

        with open(args.text_file, encoding='utf-8') as f:
            text = f.read()

        from text_forensics import full_text_arsenal
        result = full_text_arsenal(text)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
