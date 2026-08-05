#!/usr/bin/env python3
"""
scripts/causal_inference.py — 超级耿同学因果推断引擎

检测论文的方法论是否真正支持其因果声明：
- DAG 构建：从方法论描述自动构建因果图
- d-separation：检查声称的因果路径是否被混淆
- 后门准则：是否存在未控制的混淆变量
- 中介 vs 混淆：区分中介变量和混淆变量
- 因果方向检测：是否存在反向因果的可能
"""
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# Module 1: Causal Graph Building
# ═══════════════════════════════════════════════════════════

class CausalGraph:
    """简单的因果图：节点 + 有向边"""

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Set[Tuple[str, str]] = set()  # (from, to)
        self.bidirected: Set[Tuple[str, str]] = set()  # 未观测混淆

    def add_node(self, name: str):
        self.nodes.add(name)

    def add_edge(self, from_node: str, to_node: str):
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges.add((from_node, to_node))

    def add_confounder(self, node_a: str, node_b: str):
        """添加未观测混淆（双向边）"""
        pair = tuple(sorted([node_a, node_b]))
        self.bidirected.add(pair)

    def parents(self, node: str) -> Set[str]:
        return {f for f, t in self.edges if t == node}

    def children(self, node: str) -> Set[str]:
        return {t for f, t in self.edges if f == node}

    def ancestors(self, node: str) -> Set[str]:
        result = set()
        frontier = self.parents(node)
        while frontier:
            p = frontier.pop()
            if p not in result:
                result.add(p)
                frontier.update(self.parents(p))
        return result

    def is_d_separated(self, X: str, Y: str, Z: Set[str] = None) -> bool:
        """
        检查 X 和 Y 是否在给定 Z 时 d-分离。

        简化实现：检查所有无向路径是否被 Z 阻断。
        """
        if Z is None:
            Z = set()

        # 获取从 X 到 Y 的所有无向路径
        paths = self._find_all_paths(X, Y)

        for path in paths:
            if not self._is_path_blocked(path, Z):
                return False
        return True

    def _find_all_paths(self, source: str, target: str) -> List[List[str]]:
        """找到从 source 到 target 的所有无向路径"""
        # 构建无向邻接表
        adj = defaultdict(set)
        for f, t in self.edges:
            adj[f].add(t)
            adj[t].add(f)  # 无向化
        for a, b in self.bidirected:
            adj[a].add(b)
            adj[b].add(a)

        paths = []
        def dfs(current, target, visited, path):
            if current == target:
                paths.append(path[:])
                return
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, target, visited, path)
                    path.pop()
                    visited.remove(neighbor)

        dfs(source, target, {source}, [source])
        return paths

    def _node_triple_type(self, prev: str, node: str, nxt: str) -> str:
        """判断三元组 (prev, node, nxt) 的因果结构类型。"""
        forward = (prev, node) in self.edges and (node, nxt) in self.edges
        backward = (nxt, node) in self.edges and (node, prev) in self.edges
        fork = (prev, node) in self.edges and (nxt, node) in self.edges
        if forward or backward:
            return "chain"
        if fork:
            return "fork"
        if (prev, node) in self.edges and (nxt, node) in self.edges:
            return "collider"
        return "none"

    def _is_path_blocked(self, path: List[str], Z: Set[str]) -> bool:
        """检查一条路径是否被 Z 阻断"""
        for i in range(1, len(path) - 1):
            node = path[i]
            triple = self._node_triple_type(path[i - 1], node, path[i + 1])

            # Chain / Fork：被条件化时阻断
            if triple in ("chain", "fork") and node in Z:
                return True

            # Collider：未条件化且后代未被条件化时阻断
            if triple == "collider" and node not in Z:
                if not (Z & self._descendants_of(node)):
                    return True

        return False

    def _descendants_of(self, node: str) -> Set[str]:
        """获取 node 的所有后代"""
        result = set()
        frontier = {node}
        while frontier:
            current = frontier.pop()
            for child in self.children(current):
                if child not in result:
                    result.add(child)
                    frontier.add(child)
        return result

    def backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """找到 treatment → outcome 的所有后门路径"""
        all_paths = self._find_all_paths(treatment, outcome)
        backdoors = []
        for path in all_paths:
            if len(path) >= 3:
                # 后门路径：从 treatment 出发的第一条边是 incoming
                first_edge = (path[1], path[0])
                if first_edge in self.edges or tuple(sorted(first_edge)) in self.bidirected:
                    backdoors.append(path)
        return backdoors

    def get_minimal_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        """后门准则：找到阻断所有后门路径的最小变量集"""
        backdoors = self.backdoor_paths(treatment, outcome)
        if not backdoors:
            return set()

        # 简化：取所有后门路径上（除 treatment/outcome 外）的节点
        candidates = set()
        for path in backdoors:
            for node in path[1:-1]:
                candidates.add(node)

        # 排除 treatment 的后代（避免引入碰撞偏倚）
        descendants = self._descendants_of(treatment)
        candidates -= descendants

        return candidates


# ═══════════════════════════════════════════════════════════
# Module 2: Method-to-DAG Extractor
# ═══════════════════════════════════════════════════════════

def extract_causal_claims(methods_text: str, results_text: str = "") -> dict:
    """
    从方法论描述中提取因果声明并构建 DAG。

    解析论文中声明的：
    - 自变量（treatment/exposure）
    - 因变量（outcome）
    - 控制变量
    - 调节变量/中介变量

    并检查声明的因果路径是否逻辑自洽。
    """
    claims = {
        "treatments": [],
        "outcomes": [],
        "controls": [],
        "mediators": [],
        "moderators": [],
    }

    # 识别因果标志词
    causal_patterns = {
        "treatments": [
            r'(?:自变量|处理[组因]?|暴露|干预|treatment|exposure|independent\s+variable)[：:\s]*([^，。；\n]+)',
            r'(?:我们)?(?:将|通过|采用).*?(?:处理|干预|操纵|改变)([^，。；\n]+)',
            r'(?:比较|对比).*?(.+?)(?:组|群|条件下)',
        ],
        "outcomes": [
            r'(?:因变量|结果|结局|outcome|dependent\s+variable)[：:\s]*([^，。；\n]+)',
            r'(?:测量|检测|评估|观察)([^，。；\n]+?)(?:的)?(?:变化|差异|影响|水平)',
        ],
        "controls": [
            r'(?:控制|控制变量|covariate|混杂)[：:\s]*([^，。；\n]+)',
            r'(?:控制了?|校正了?|调整了?)([^，。；\n]+)',
        ],
        "mediators": [
            r'(?:中介|mediator|路径|机制).*?(?:通过|经由)([^，。；\n]+)',
            r'(?:中介效应|中介分析|mediation)',
        ],
        "moderators": [
            r'(?:调节|moderator|交互|interaction).*?([^，。；\n]+)',
            r'(?:调节效应|交互效应)',
        ],
    }

    for category, patterns in causal_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, methods_text)
            for m in matches:
                m_clean = m.strip().rstrip('的，。；')
                if len(m_clean) > 2 and len(m_clean) < 100:
                    claims[category].append(m_clean)

    # 去重
    for k in claims:
        claims[k] = list(dict.fromkeys(claims[k]))[:10]

    return claims


def build_causal_graph(claims: dict) -> CausalGraph:
    """
    从提取的因果声明构建因果图。
    """
    g = CausalGraph()

    # 简化变量名
    treatments = [f"T{i+1}: {t[:20]}" for i, t in enumerate(claims["treatments"][:3])]
    outcomes = [f"Y{i+1}: {o[:20]}" for i, o in enumerate(claims["outcomes"][:3])]
    controls = [f"C{i+1}: {c[:20]}" for i, c in enumerate(claims["controls"][:5])]
    mediators = [f"M{i+1}: {m[:20]}" for i, m in enumerate(claims["mediators"][:3])]

    # 默认结构：仅构建"主效应"边（treatment -> outcome）与中介路径，
    # 这些是合法的因果假设，不应被判为后门。
    _build_main_effect_edges(g, treatments, outcomes, mediators)

    # 关键修复：控制变量不再默认全连接到所有 treatment/outcome（原实现
    # 的笛卡尔积会导致每条 treatment->outcome 都存在经 control 的后门路径，
    # 造成必然误报）。仅当控制变量的原始声明文本中实际提及了某结果名时，
    # 才连 control -> 该 outcome 边；treatment 同理。这样后门检测反映的是
    # 文本真实声明的混淆结构，而非凭空构造。
    control_text = " ".join(claims.get("controls", []))
    _build_sparse_control_edges(g, controls, treatments, outcomes, control_text)

    return g


def _build_main_effect_edges(g, treatments, outcomes, mediators):
    """构建 treatment->outcome 主效应边与 treatment->mediator->outcome 路径。"""
    for t in treatments:
        for y in outcomes:
            g.add_edge(t, y)
    for t in treatments:
        for m in mediators:
            g.add_edge(t, m)
    for m in mediators:
        for y in outcomes:
            g.add_edge(m, y)


def _build_sparse_control_edges(g, controls, treatments, outcomes, control_text):
    """基于声明文本稀疏地连接控制变量边（避免默认全连接的必然误报）。"""
    for c in controls:
        c_name = c.lower()
        for y in outcomes:
            if c_name[:8] in control_text and _mentions(y, control_text):
                g.add_edge(c, y)
        for t in treatments:
            if c_name[:8] in control_text and _mentions(t, control_text):
                g.add_edge(c, t)


def _mentions(node_label: str, text: str) -> bool:
    """判断节点标签是否出现在文本中（简单子串匹配，作为稀疏连接依据）"""
    token = node_label.lower()
    return token[:8] in text.lower()


# ═══════════════════════════════════════════════════════════
# Module 3: Methodology Validation
# ═══════════════════════════════════════════════════════════

def validate_methodology(methods_text: str) -> dict:
    """
    验证论文方法论是否支持其因果声明。

    检测：
    1. 是否存在未控制的混淆变量（后门路径未阻断）
    2. 是否混淆了中介和混淆变量
    3. 样本量是否不足以支持声称的因果分析
    4. 是否存在反向因果的可能
    5. 时间顺序是否合理
    """
    claims = extract_causal_claims(methods_text)
    g = build_causal_graph(claims)

    issues = []

    # 1. 后门路径未阻断
    backdoor_issues = _check_backdoor(g)
    issues.extend(backdoor_issues)

    # 2. 样本量不足
    n_total = _check_sample_size(methods_text, claims, issues)

    # 3. 中介 vs 混淆混淆
    _check_mediator_confound(claims, issues)

    # 4. 反向因果检测
    _check_reverse_causal(methods_text, issues)

    # 综合判定
    if len(issues) >= 3:
        verdict = f"🚨 方法论严重缺陷：{'；'.join(issues[:3])}"
    elif len(issues) >= 1:
        verdict = f"⚠️ 方法论问题：{'；'.join(issues[:2])}"
    else:
        verdict = "✅ 方法论基本合理"

    return {
        "claims_summary": {
            "treatments": claims["treatments"][:3],
            "outcomes": claims["outcomes"][:3],
            "controls": claims["controls"][:5],
            "mediators": claims["mediators"][:3],
        },
        "graph_nodes": len(g.nodes),
        "graph_edges": len(g.edges),
        "sample_size": n_total,
        "backdoor_issues": len(backdoor_issues),
        "issues": issues,
        "verdict": verdict,
    }


def _check_backdoor(g) -> list:
    """检测后门路径未被控制变量阻断的情况。"""
    issues = []
    treatments = [n for n in g.nodes if n.startswith("T")]
    outcomes = [n for n in g.nodes if n.startswith("Y")]
    controls_in_graph = {n for n in g.nodes if n.startswith("C")}

    for t in treatments:
        for y in outcomes:
            if t == y:
                continue
            backdoors = g.backdoor_paths(t, y)
            if backdoors:
                adj_set = g.get_minimal_adjustment_set(t, y)
                missing_controls = adj_set - controls_in_graph
                if missing_controls:
                    issues.append(
                        f"后门路径未阻断：{t}→{y} 存在混淆，"
                        f"需额外控制 {missing_controls}"
                    )
    return issues


def _check_sample_size(methods_text: str, claims: dict, issues: list) -> int:
    """检测样本量是否足以支撑因果分析，返回最大样本量（无则 None）。"""
    sample_patterns = [
        r'[Nn]\s*[=＝]\s*(\d+)',
        r'(\d+)\s*(?:例|个|名|只|份)',
        r'sample\s*size.*?(\d+)',
    ]
    sample_sizes = []
    for pattern in sample_patterns:
        matches = re.findall(pattern, methods_text)
        sample_sizes.extend([int(m) for m in matches if int(m) > 1])

    if not sample_sizes:
        return None

    n_total = max(sample_sizes)
    n_vars = len(claims["treatments"]) + len(claims["controls"]) + 1
    if n_vars > 1 and n_total / n_vars < 10:
        issues.append(
            f"样本量不足：{n_total} 个样本 vs {n_vars} 个变量，"
            f"每变量不足10个观测"
        )
    return n_total


def _check_mediator_confound(claims: dict, issues: list):
    """检查中介变量是否可能是被误判的混淆变量。"""
    confound_keywords = ['年龄', '性别', '收入', '教育', '基线', 'baseline',
                        'age', 'sex', 'gender', 'income', 'education']
    for med in claims["mediators"][:3]:
        for kw in confound_keywords:
            if kw in med:
                issues.append(
                    f"可能的混淆误作中介：'{med}' 含 '{kw}'，"
                    f"可能是混淆变量而非中介变量"
                )
                break


def _check_reverse_causal(methods_text: str, issues: list):
    """检测仅报告相关却声称因果（无纵向/干预设计）的情况。"""
    if not re.search(r'(?:相关|关联|correlation|association)', methods_text):
        return
    has_causal = re.search(
        r'(?:因果|causal|cause|效应|effect)', methods_text)
    has_design = re.search(
        r'(?:纵[向贯]|longitudinal|前瞻|prospective|干预|intervention)',
        methods_text)
    if not (has_causal or has_design):
        issues.append(
            "仅报告相关/关联但声称因果，且无纵向设计或干预设计"
        )


# ═══════════════════════════════════════════════════════════
# Module 4: Quick Causal Validation
# ═══════════════════════════════════════════════════════════

def quick_causal_check(claim: str, method: str) -> dict:
    """
    快速因果检验：给定一个因果主张和方法描述，
    检查方法是否支持该主张。
    """
    full_text = claim + "\n" + method
    result = validate_methodology(full_text)
    result["claim"] = claim[:200]
    result["method"] = method[:200]
    return result
