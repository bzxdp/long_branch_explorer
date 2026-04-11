#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree

EPS = 1e-12
MIN_MASKING_CLADE_SIZE = 2
MIN_CROSS_GENE_OBS = 3
DEFAULT_INTERNAL_MIN_PATH_BRANCHES = 3  # including the tested branch itself
WARN_FEW_TREES = 50
WARN_VERY_FEW_TREES = 10
CROSS_GENE_DISABLE_TREES = 3
NA_STR = "NA"
TREE_EXTS = {".nwk", ".tree", ".tre", ".newick", ".txt", ".treefile"}

FIGTREE_COLOR_HEX = {
    "quartet_drop": "#FF0000",
    "stem_drop": "#800080",
    "internal_branch_drop": "#FF69B4",
    "terminal_drop": "#FFA500",
    "global_terminal_rescue": "#00008B",
    "global_stem_rescue": "#ADD8E6",
    "global_internal_rescue": "#20B2AA",
    "default": "#000000",
}


@dataclass
class StemMembershipRow:
    tree_id: str
    taxon: str
    stem_group: str
    obs_count: int
    stem_branch: float
    median_branch: float
    ratio: float
    tested: bool
    outlier: bool
    rescued_by_global_stem: bool
    skipped_due_to_ancestor_outlier: str


@dataclass
class InternalBranchRow:
    tree_id: str
    edge_id: str
    node_a: str
    node_b: str
    branch_length: float
    matched_known_stem_group: str
    tested_by_known_stem: bool
    tested_by_global_internal: bool
    ever_tested_internal: bool
    untested_internal_warning: str
    global_internal_median_branch: float
    global_internal_ratio: float
    side_a_tip_count: int
    side_b_tip_count: int
    side_a_median_split_to_tip: float
    side_b_median_split_to_tip: float
    deleted_side: str
    candidate_taxa: List[str]
    downstream_internal_branch_count: int
    downstream_long_branch_count: int
    downstream_long_fraction: float
    candidate_outlier: bool
    rescued_by_global_internal: bool
    drop_applied: bool
    pruning_blocked_reason: str
    global_internal_q25_branch: float
    global_internal_q50_branch: float
    global_internal_q75_branch: float
    global_internal_rescue_quantile: str
    global_internal_rescue_cutoff: float


@dataclass
class TaxonResult:
    tree_id: str
    taxon: str
    anchor_label: str
    branch_length: float
    effective_length: float
    local_median: float
    local_ratio: float
    cross_gene_method: str
    cross_gene_obs: int
    cross_gene_median_branch: float
    cross_gene_ratio: float
    stem_group_count: int
    stem_outlier_group_count: int
    stem_rescued_group_count: int
    internal_drop_edge_count: int
    internal_rescued_edge_count: int
    global_terminal_q25_branch: float
    global_terminal_q50_branch: float
    global_terminal_q75_branch: float
    global_rescue_quantile: str
    global_rescue_cutoff: float
    global_stem_q25_branch: float
    global_stem_q50_branch: float
    global_stem_q75_branch: float
    global_stem_rescue_quantile: str
    global_stem_rescue_cutoff: float
    global_internal_q25_branch: float
    global_internal_q50_branch: float
    global_internal_q75_branch: float
    global_internal_rescue_quantile: str
    global_internal_rescue_cutoff: float
    comparator_taxa: List[str]
    decision: str
    reason: str
    quartet_drop: bool = False
    cross_gene_terminal_drop: bool = False
    stem_drop: bool = False
    internal_branch_drop: bool = False
    global_terminal_rescue_applied: bool = False
    global_stem_rescue_membership: bool = False
    global_internal_rescue_membership: bool = False
    applied_rules: List[str] = field(default_factory=list)
    display_category: str = "default"
    multiple_rules_applied: bool = False


@dataclass
class TreeContext:
    tree: Tree
    tree_id: str
    parent_map: Dict[Clade, Optional[Clade]]
    node_ids: Dict[Clade, int]
    quartet_mask_groups: Dict[str, Set[str]] = field(default_factory=dict)
    taxon_to_quartet_mask_group: Dict[str, str] = field(default_factory=dict)
    stem_groups: Dict[str, Set[str]] = field(default_factory=dict)
    taxon_to_stem_groups: Dict[str, List[str]] = field(default_factory=dict)
    current_clade_stems: Dict[str, float] = field(default_factory=dict)
    stem_eval: Dict[str, "StemEvalRecord"] = field(default_factory=dict)
    mandatory_stem_drop_groups: Set[str] = field(default_factory=set)
    rescued_stem_groups: Set[str] = field(default_factory=set)
    cross_gene_medians: Dict[str, float] = field(default_factory=dict)
    cross_gene_counts: Dict[str, int] = field(default_factory=dict)
    clade_stem_medians: Dict[str, float] = field(default_factory=dict)
    clade_stem_counts: Dict[str, int] = field(default_factory=dict)
    cross_gene_ratio_threshold: float = 3.0
    clade_stem_ratio_threshold: float = 3.0
    cross_gene_enabled: bool = True
    hybrid_terminal: bool = False
    hybrid_stem: bool = False
    global_q25_terminal_branch: float = float("nan")
    global_q50_terminal_branch: float = float("nan")
    global_q75_terminal_branch: float = float("nan")
    global_rescue_quantile: Optional[float] = None
    global_rescue_cutoff: float = float("nan")
    global_q25_stem_branch: float = float("nan")
    global_q50_stem_branch: float = float("nan")
    global_q75_stem_branch: float = float("nan")
    global_stem_rescue_quantile: Optional[float] = None
    global_stem_rescue_cutoff: float = float("nan")
    global_q25_internal_branch: float = float("nan")
    global_q50_internal_branch: float = float("nan")
    global_q75_internal_branch: float = float("nan")
    global_internal_rescue_quantile: Optional[float] = None
    global_internal_rescue_cutoff: float = float("nan")
    taxon_to_internal_drop_edges: Dict[str, List[str]] = field(default_factory=dict)
    taxon_to_internal_rescued_edges: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class QuartetChoice:
    anchor: Clade
    comparators: List[Clade]


@dataclass
class MaskDetectionResult:
    mask_groups: Dict[str, Set[str]]
    stem_lengths: Dict[str, float]
    monophyletic_clades: List[Tuple[str, int]]


@dataclass
class CrossGeneStats:
    taxon_medians: Dict[str, float]
    taxon_counts: Dict[str, int]
    global_q25_terminal_branch: float
    global_q50_terminal_branch: float
    global_q75_terminal_branch: float
    clade_stem_medians: Dict[str, float]
    clade_stem_counts: Dict[str, int]
    global_q25_stem_branch: float
    global_q50_stem_branch: float
    global_q75_stem_branch: float
    global_q25_internal_branch: float
    global_q50_internal_branch: float
    global_q75_internal_branch: float
    all_terminal_lengths: List[float]
    all_retained_stems: List[float]
    all_internal_lengths: List[float]


@dataclass
class StemEvalRecord:
    taxa: Set[str]
    stem_branch: float
    obs_count: int
    median_branch: float
    ratio: float
    tested: bool
    outlier: bool
    rescued_by_global_stem: bool = False
    skipped_due_to_ancestor_outlier: Optional[str] = None


@dataclass
class InternalEdgeRecord:
    edge_id: str
    edge_len: float
    matched_known_stem_group: Optional[str]
    node_a: str
    node_b: str
    tip_names_a: List[str]
    tip_names_b: List[str]
    median_a: float
    median_b: float


def median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.median(vals) if vals else None


def quantile(values: Iterable[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return None
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    pos = (len(vals) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return vals[low]
    weight = pos - low
    return vals[low] * (1.0 - weight) + vals[high] * weight


def format_optional_quantile(q: Optional[float]) -> str:
    return NA_STR if q is None else f"{q:.2f}"


def quantile_reason_suffix(q: float) -> str:
    return f"q{int(round(q * 100))}"


def format_scalar(value) -> str:
    if value is None:
        return NA_STR
    if isinstance(value, float) and math.isnan(value):
        return NA_STR
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def quantile_arg(value: str) -> float:
    q = float(value)
    if not (0.0 <= q <= 1.0):
        raise argparse.ArgumentTypeError("must be a number between 0 and 1 inclusive")
    return q


def load_tree(path: Path) -> Tree:
    return Phylo.read(str(path), "newick")


def iter_tree_files(trees_dir: Path) -> List[Path]:
    return sorted(p for p in trees_dir.iterdir() if p.is_file() and p.suffix.lower() in TREE_EXTS)


def build_parent_map(tree: Tree) -> Dict[Clade, Optional[Clade]]:
    parent_map: Dict[Clade, Optional[Clade]] = {tree.root: None}
    for parent in tree.find_clades(order="level"):
        for child in parent.clades:
            parent_map[child] = parent
    return parent_map


def assign_node_ids(tree: Tree) -> Dict[Clade, int]:
    return {clade: idx for idx, clade in enumerate(tree.find_clades(order="level"))}


def branch_length_to_parent(clade: Clade) -> float:
    return float(clade.branch_length) if clade.branch_length is not None else 0.0


def node_label(ctx: TreeContext, node: Optional[Clade]) -> str:
    if node is None:
        return "no_anchor"
    return (node.name or f"unnamed_tip_{ctx.node_ids[node]}") if node.is_terminal() else f"node_{ctx.node_ids[node]}"


def undirected_neighbors(parent_map: Dict[Clade, Optional[Clade]], node: Clade) -> List[Clade]:
    neighbors = list(node.clades)
    parent = parent_map[node]
    if parent is not None:
        neighbors.append(parent)
    return neighbors


def subtree_tips_away_from(start: Clade, blocked: Clade, parent_map: Dict[Clade, Optional[Clade]]) -> List[Clade]:
    tips: List[Clade] = []
    queue = deque([start])
    visited = {blocked}
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node.is_terminal():
            tips.append(node)
            continue
        for neigh in undirected_neighbors(parent_map, node):
            if neigh not in visited:
                queue.append(neigh)
    return tips


def parse_named_clades(path: Path) -> Dict[str, Set[str]]:
    clades: Dict[str, Set[str]] = {}
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Invalid line {line_number} in {path}: missing ':'. "
                    f"Expected format like 'CladeA: taxon1 taxon2 taxon3'. "
                    f"Offending line: {raw_line.rstrip()}"
                )
            name, taxa = line.split(":", 1)
            if "," in taxa:
                raise ValueError(
                    f"Comma detected in taxa list for '{name.strip()}' on line {line_number}. "
                    "Use SPACE-separated taxa only."
                )
            taxa_set = {x.strip() for x in taxa.split() if x.strip()}
            if not taxa_set:
                raise ValueError(
                    f"No taxa found for clade '{name.strip()}' on line {line_number} in {path}."
                )
            clades[name.strip()] = taxa_set
    return clades


def build_networkx_tree(tree: Tree) -> Tuple[nx.Graph, int]:
    G = nx.Graph()
    clade_to_id: Dict[Clade, int] = {}
    for idx, clade in enumerate(tree.find_clades(order="level")):
        clade_to_id[clade] = idx
        G.add_node(idx, clade=clade, is_tip=clade.is_terminal(), name=clade.name if clade.is_terminal() else None)
    for parent in tree.find_clades(order="level"):
        pid = clade_to_id[parent]
        for child in parent.clades:
            cid = clade_to_id[child]
            G.add_edge(pid, cid, length=branch_length_to_parent(child))
    return G, clade_to_id[tree.root]


def contract_degree_two_root(G: nx.Graph, root_id: int) -> nx.Graph:
    H = G.copy()
    if root_id not in H:
        return H
    neighbors = list(H.neighbors(root_id))
    if len(neighbors) != 2:
        return H
    a, b = neighbors
    new_len = float(H[root_id][a].get("length", 0.0)) + float(H[root_id][b].get("length", 0.0))
    H.remove_node(root_id)
    if H.has_edge(a, b):
        H[a][b]["length"] = min(float(H[a][b].get("length", 0.0)), new_len)
    else:
        H.add_edge(a, b, length=new_len)
    return H


def build_unrooted_graph(tree: Tree) -> Tuple[nx.Graph, int]:
    G_raw, root_id = build_networkx_tree(tree)
    return contract_degree_two_root(G_raw, root_id), root_id


def internal_edges_from_graph(G: nx.Graph) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    for u, v, data in G.edges(data=True):
        if G.nodes[u].get("is_tip") or G.nodes[v].get("is_tip"):
            continue
        edges.append((u, v, float(data.get("length", 0.0))))
    edges.sort(key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))
    return edges


def tip_names_in_component(G: nx.Graph, component: Set[int]) -> Set[str]:
    return {G.nodes[node]["name"] for node in component if G.nodes[node].get("name")}


def detect_mask_groups_from_graph(tree: Tree, clade_defs: Dict[str, Set[str]], min_size: int = MIN_MASKING_CLADE_SIZE) -> MaskDetectionResult:
    mask_groups: Dict[str, Set[str]] = {}
    stem_lengths: Dict[str, float] = {}
    monophyletic_clades: List[Tuple[str, int]] = []
    if not clade_defs:
        return MaskDetectionResult(mask_groups, stem_lengths, monophyletic_clades)
    G_raw, root_id = build_networkx_tree(tree)
    G = contract_degree_two_root(G_raw, root_id)
    all_named_tips = {data["name"] for _, data in G.nodes(data=True) if data.get("name")}
    for clade_name, taxa in clade_defs.items():
        present_taxa = taxa & all_named_tips
        if len(present_taxa) < min_size:
            continue
        best_fragment: Set[str] = set()
        best_stem_length = float("nan")
        for u, v, data in list(G.edges(data=True)):
            H = G.copy()
            H.remove_edge(u, v)
            components = list(nx.connected_components(H))
            if len(components) != 2:
                continue
            comp1, comp2 = components
            side1 = tip_names_in_component(H, comp1)
            side2 = tip_names_in_component(H, comp2)
            edge_len = float(data.get("length", 0.0))
            if len(side1) >= min_size and side1 <= present_taxa and len(side1) > len(best_fragment):
                best_fragment = side1
                best_stem_length = edge_len
            if len(side2) >= min_size and side2 <= present_taxa and len(side2) > len(best_fragment):
                best_fragment = side2
                best_stem_length = edge_len
        if len(best_fragment) >= min_size:
            mask_groups[clade_name] = set(best_fragment)
            stem_lengths[clade_name] = best_stem_length
            monophyletic_clades.append((clade_name, len(best_fragment)))
    return MaskDetectionResult(mask_groups, stem_lengths, monophyletic_clades)


def collapse_identical_stem_fragments(
    raw_groups: Dict[str, Set[str]],
    raw_stems: Dict[str, float],
    clade_defs: Dict[str, Set[str]],
) -> Tuple[Dict[str, Set[str]], Dict[str, float], Dict[str, List[str]]]:
    by_fragment: Dict[frozenset[str], List[str]] = defaultdict(list)
    for name, taxa in raw_groups.items():
        by_fragment[frozenset(taxa)].append(name)

    collapsed_groups: Dict[str, Set[str]] = {}
    collapsed_stems: Dict[str, float] = {}
    fragment_aliases: Dict[str, List[str]] = {}
    for frag, names in by_fragment.items():
        chosen = sorted(names, key=lambda n: (len(clade_defs.get(n, set())), n))[0]
        collapsed_groups[chosen] = set(frag)
        collapsed_stems[chosen] = raw_stems[chosen]
        fragment_aliases[chosen] = sorted(names)

    return collapsed_groups, collapsed_stems, fragment_aliases


def build_taxon_to_group(groups: Dict[str, Set[str]]) -> Dict[str, str]:
    taxon_to_group: Dict[str, str] = {}
    for group_name, taxa in groups.items():
        for taxon in taxa:
            if taxon in taxon_to_group and taxon_to_group[taxon] != group_name:
                raise ValueError(
                    f"Taxon {taxon!r} appears in more than one retained quartet-mask group: "
                    f"{taxon_to_group[taxon]!r} and {group_name!r}"
                )
            taxon_to_group[taxon] = group_name
    return taxon_to_group


def build_taxon_to_groups(groups: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    taxon_to_groups: Dict[str, List[str]] = defaultdict(list)
    for group_name in sorted(groups):
        for taxon in sorted(groups[group_name]):
            taxon_to_groups[taxon].append(group_name)
    return dict(taxon_to_groups)


def select_global_quantile_cutoff(values: List[float], q: Optional[float]) -> float:
    if q is None or not values:
        return float("nan")
    cutoff = quantile(values, q)
    return cutoff if cutoff is not None else float("nan")


def cross_gene_method_name(enabled: bool = True, hybrid_terminal: bool = False, hybrid_stem: bool = False) -> str:
    if not enabled:
        return "disabled_small_dataset"
    if hybrid_terminal or hybrid_stem:
        parts = []
        if hybrid_terminal:
            parts.append("hybrid_terminal")
        if hybrid_stem:
            parts.append("hybrid_stem")
        return "+".join(parts)
    return "median_ratio"


def hybrid_trigger_cutoff(local_median: float, global_median: float, ratio_threshold: float) -> float:
    return ((ratio_threshold - 1.0) * local_median) + global_median


def should_drop_terminal_by_cross_gene(
    branch_length: float,
    local_median_value: float,
    obs_count: int,
    ratio_threshold: float,
    global_terminal_median: float,
    hybrid_terminal: bool,
) -> Tuple[bool, float, float]:
    if obs_count < MIN_CROSS_GENE_OBS:
        return False, float("nan"), float("nan")
    ratio = float("nan")
    hybrid_cutoff = float("nan")
    if local_median_value is not None and not math.isnan(local_median_value) and local_median_value > EPS:
        ratio = branch_length / local_median_value
    if hybrid_terminal:
        if (
            local_median_value is None
            or math.isnan(local_median_value)
            or local_median_value <= EPS
            or global_terminal_median is None
            or math.isnan(global_terminal_median)
        ):
            return False, ratio, hybrid_cutoff
        hybrid_cutoff = hybrid_trigger_cutoff(local_median_value, global_terminal_median, ratio_threshold)
        return (branch_length > hybrid_cutoff), ratio, hybrid_cutoff
    return (not math.isnan(ratio) and ratio >= ratio_threshold), ratio, hybrid_cutoff


def evaluate_stem_groups_for_tree(
    stem_groups: Dict[str, Set[str]],
    current_clade_stems: Dict[str, float],
    cross_gene_stats: CrossGeneStats,
    clade_stem_ratio_threshold: float,
    global_stem_rescue_quantile: Optional[float],
    hybrid_stem: bool,
) -> Tuple[Dict[str, StemEvalRecord], Set[str], Set[str]]:
    stem_eval: Dict[str, StemEvalRecord] = {}
    mandatory_drop_groups: Set[str] = set()
    rescued_groups: Set[str] = set()

    global_cutoff = select_global_quantile_cutoff(
        cross_gene_stats.all_retained_stems,
        global_stem_rescue_quantile,
    )
    global_stem_median = median(cross_gene_stats.all_retained_stems)

    ordered_groups = sorted(stem_groups.keys(), key=lambda g: (-len(stem_groups[g]), g))

    for group_name in ordered_groups:
        taxa = stem_groups[group_name]
        stem_branch = current_clade_stems.get(group_name, float("nan"))
        obs_count = cross_gene_stats.clade_stem_counts.get(group_name, 0)
        median_branch = cross_gene_stats.clade_stem_medians.get(group_name, float("nan"))

        tested = obs_count >= MIN_CROSS_GENE_OBS
        skipped_due_to = None

        for parent_group in sorted(mandatory_drop_groups):
            if taxa <= stem_groups[parent_group]:
                tested = False
                skipped_due_to = parent_group
                break

        ratio = float("nan")
        outlier = False
        rescued_by_global_stem = False

        if tested and skipped_due_to is None:
            if not math.isnan(median_branch) and median_branch > EPS:
                ratio = stem_branch / median_branch

            if hybrid_stem:
                if global_stem_median is not None and not math.isnan(global_stem_median) and not math.isnan(median_branch):
                    outlier = stem_branch > hybrid_trigger_cutoff(median_branch, global_stem_median, clade_stem_ratio_threshold)
            else:
                outlier = not math.isnan(ratio) and ratio >= clade_stem_ratio_threshold

            if outlier and global_stem_rescue_quantile is not None:
                if not math.isnan(global_cutoff) and stem_branch < global_cutoff:
                    rescued_by_global_stem = True
                    rescued_groups.add(group_name)

            if outlier and not rescued_by_global_stem:
                mandatory_drop_groups.add(group_name)

        stem_eval[group_name] = StemEvalRecord(
            taxa=taxa,
            stem_branch=stem_branch,
            obs_count=obs_count,
            median_branch=median_branch,
            ratio=ratio,
            tested=tested,
            outlier=outlier,
            rescued_by_global_stem=rescued_by_global_stem,
            skipped_due_to_ancestor_outlier=skipped_due_to,
        )

    return stem_eval, mandatory_drop_groups, rescued_groups


def build_taxon_to_mandatory_stem_outlier_groups(
    taxon_to_stem_groups: Dict[str, List[str]],
    mandatory_stem_drop_groups: Set[str],
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    for taxon, group_names in taxon_to_stem_groups.items():
        for group_name in group_names:
            if group_name in mandatory_stem_drop_groups:
                result[taxon].append(group_name)
    return {taxon: sorted(groups) for taxon, groups in result.items()}


def build_taxon_to_rescued_stem_groups(
    stem_groups: Dict[str, Set[str]],
    rescued_stem_groups: Set[str],
    mandatory_stem_drop_groups: Set[str],
) -> Dict[str, List[str]]:
    """Assign stem rescue only to the rescued clade itself, excluding taxa that belong
    to more specific dropped nested clades. Aliases are unchanged because they share
    the same representative fragment before this stage."""
    result: Dict[str, List[str]] = defaultdict(list)
    for rescued_group in sorted(rescued_stem_groups):
        rescued_taxa = set(stem_groups.get(rescued_group, set()))
        if not rescued_taxa:
            continue
        nested_dropped_taxa: Set[str] = set()
        for dropped_group in mandatory_stem_drop_groups:
            dropped_taxa = stem_groups.get(dropped_group, set())
            if dropped_group == rescued_group:
                continue
            if dropped_taxa and dropped_taxa < rescued_taxa:
                nested_dropped_taxa.update(dropped_taxa)
        surviving_taxa = rescued_taxa - nested_dropped_taxa
        for taxon in sorted(surviving_taxa):
            result[taxon].append(rescued_group)
    return {taxon: sorted(groups) for taxon, groups in result.items()}


def collect_branch_and_clade_stats(
    tree_files: List[Path],
    stem_test_defs: Optional[Dict[str, Set[str]]],
    progress: bool = False,
) -> CrossGeneStats:
    lengths_by_taxon: Dict[str, List[float]] = defaultdict(list)
    all_terminal_lengths: List[float] = []
    clade_stems_by_group: Dict[str, List[float]] = defaultdict(list)
    all_retained_stems: List[float] = []
    all_internal_lengths: List[float] = []
    total = len(tree_files)

    for i, tree_file in enumerate(tree_files, start=1):
        if progress:
            print(f"Collecting cross-gene statistics: {i}/{total} trees ({tree_file.stem})", flush=True)
        tree = load_tree(tree_file)

        for tip in tree.get_terminals():
            if not tip.name:
                continue
            bl = branch_length_to_parent(tip)
            lengths_by_taxon[tip.name].append(bl)
            all_terminal_lengths.append(bl)

        stem_groups: Dict[str, Set[str]] = {}
        if stem_test_defs:
            stem_raw = detect_mask_groups_from_graph(
                tree=tree,
                clade_defs=stem_test_defs,
                min_size=MIN_MASKING_CLADE_SIZE,
            )
            stem_groups, collapsed_stems, _ = collapse_identical_stem_fragments(
                raw_groups=stem_raw.mask_groups,
                raw_stems=stem_raw.stem_lengths,
                clade_defs=stem_test_defs,
            )
            for group_name, stem_len in collapsed_stems.items():
                clade_stems_by_group[group_name].append(stem_len)
                all_retained_stems.append(stem_len)

        for rec in build_internal_edge_records(tree, tree_file.stem, stem_groups):
            if rec.matched_known_stem_group is None:
                all_internal_lengths.append(rec.edge_len)

    taxon_medians: Dict[str, float] = {}
    taxon_counts: Dict[str, int] = {}
    for taxon, vals in lengths_by_taxon.items():
        taxon_counts[taxon] = len(vals)
        med = median(vals)
        if med is not None:
            taxon_medians[taxon] = med

    clade_stem_medians: Dict[str, float] = {}
    clade_stem_counts: Dict[str, int] = {}
    for group_name, vals in clade_stems_by_group.items():
        clade_stem_counts[group_name] = len(vals)
        med = median(vals)
        if med is not None:
            clade_stem_medians[group_name] = med

    q25_terminal = quantile(all_terminal_lengths, 0.25)
    q50_terminal = quantile(all_terminal_lengths, 0.50)
    q75_terminal = quantile(all_terminal_lengths, 0.75)
    q25_stem = quantile(all_retained_stems, 0.25)
    q50_stem = quantile(all_retained_stems, 0.50)
    q75_stem = quantile(all_retained_stems, 0.75)
    q25_internal = quantile(all_internal_lengths, 0.25)
    q50_internal = quantile(all_internal_lengths, 0.50)
    q75_internal = quantile(all_internal_lengths, 0.75)

    return CrossGeneStats(
        taxon_medians=taxon_medians,
        taxon_counts=taxon_counts,
        global_q25_terminal_branch=q25_terminal if q25_terminal is not None else float("nan"),
        global_q50_terminal_branch=q50_terminal if q50_terminal is not None else float("nan"),
        global_q75_terminal_branch=q75_terminal if q75_terminal is not None else float("nan"),
        clade_stem_medians=clade_stem_medians,
        clade_stem_counts=clade_stem_counts,
        global_q25_stem_branch=q25_stem if q25_stem is not None else float("nan"),
        global_q50_stem_branch=q50_stem if q50_stem is not None else float("nan"),
        global_q75_stem_branch=q75_stem if q75_stem is not None else float("nan"),
        global_q25_internal_branch=q25_internal if q25_internal is not None else float("nan"),
        global_q50_internal_branch=q50_internal if q50_internal is not None else float("nan"),
        global_q75_internal_branch=q75_internal if q75_internal is not None else float("nan"),
        all_terminal_lengths=all_terminal_lengths,
        all_retained_stems=all_retained_stems,
        all_internal_lengths=all_internal_lengths,
    )


def comparator_allowed(ctx: TreeContext, focal_name: str, comparator_name: str) -> bool:
    focal_group = ctx.taxon_to_quartet_mask_group.get(focal_name)
    comp_group = ctx.taxon_to_quartet_mask_group.get(comparator_name)
    if focal_group is None:
        return comp_group is None
    return comp_group is None or comp_group == focal_group


def focal_neighbor_for_anchor(ctx: TreeContext, tip: Clade, anchor: Clade) -> Optional[Clade]:
    for neigh in undirected_neighbors(ctx.parent_map, anchor):
        if tip in subtree_tips_away_from(neigh, anchor, ctx.parent_map):
            return neigh
    return None


def candidate_anchors_for_tip(ctx: TreeContext, tip: Clade) -> List[Clade]:
    start = ctx.parent_map.get(tip)
    if start is None:
        return []
    visited: Set[Clade] = {tip}
    queue = deque([start])
    ordered: List[Clade] = []
    while queue:
        level_nodes = list(queue)
        queue.clear()
        level_nodes.sort(key=lambda n: ctx.node_ids[n])
        for node in level_nodes:
            if node in visited:
                continue
            visited.add(node)
            if not node.is_terminal():
                ordered.append(node)
            neighs = undirected_neighbors(ctx.parent_map, node)
            neighs.sort(key=lambda n: ctx.node_ids[n])
            for neigh in neighs:
                if neigh not in visited:
                    queue.append(neigh)
    return ordered


def build_allowed_groups_at_anchor(ctx: TreeContext, tip: Clade, anchor: Clade) -> List[List[Clade]]:
    focal_direction = focal_neighbor_for_anchor(ctx, tip, anchor)
    if focal_direction is None:
        return []
    groups: List[List[Clade]] = []
    for neigh in undirected_neighbors(ctx.parent_map, anchor):
        if neigh is focal_direction:
            continue
        tips = [
            t
            for t in subtree_tips_away_from(neigh, anchor, ctx.parent_map)
            if t.name and comparator_allowed(ctx, tip.name, t.name)
        ]
        if not tips:
            continue
        tips.sort(key=lambda t: (float(ctx.tree.distance(anchor, t)), t.name))
        groups.append(tips)
    return groups


def choose_quartet_from_groups(ctx: TreeContext, anchor: Clade, groups: List[List[Clade]]) -> Optional[List[Clade]]:
    if len(groups) < 2:
        return None
    if sum(len(g) for g in groups) < 3:
        return None
    chosen: List[Clade] = [g[0] for g in groups]
    if len(chosen) > 3:
        chosen.sort(key=lambda t: (float(ctx.tree.distance(anchor, t)), t.name))
        chosen = chosen[:3]
    remaining = [t for g in groups for t in g[1:]]
    remaining.sort(key=lambda t: (float(ctx.tree.distance(anchor, t)), t.name))
    while len(chosen) < 3 and remaining:
        chosen.append(remaining.pop(0))
    return chosen if len(chosen) == 3 else None


def find_mask_aware_quartet_for_tip(ctx: TreeContext, tip: Clade) -> Tuple[Optional[QuartetChoice], str]:
    for anchor in candidate_anchors_for_tip(ctx, tip):
        groups = build_allowed_groups_at_anchor(ctx, tip, anchor)
        comparators = choose_quartet_from_groups(ctx, anchor, groups)
        if comparators is not None:
            return QuartetChoice(anchor=anchor, comparators=comparators), "valid_quartet"
    return None, "insufficient_valid_quartet_after_masking"


def node_name_for_graph(G: nx.Graph, node_id: int) -> str:
    if G.nodes[node_id].get("is_tip"):
        return G.nodes[node_id].get("name") or f"tip_{node_id}"
    return f"node_{node_id}"


def component_tip_names_and_distances(
    H: nx.Graph,
    component: Set[int],
    start_node: int,
) -> Tuple[List[str], List[float]]:
    sub = H.subgraph(component).copy()
    lengths = nx.single_source_dijkstra_path_length(sub, source=start_node, weight="length")
    tip_names: List[str] = []
    tip_distances: List[float] = []
    for node in sorted(component):
        if sub.nodes[node].get("is_tip") and sub.nodes[node].get("name"):
            tip_names.append(sub.nodes[node]["name"])
            tip_distances.append(float(lengths.get(node, float("nan"))))
    return tip_names, [d for d in tip_distances if not math.isnan(d)]


def build_internal_edge_records(
    tree: Tree,
    tree_id: str,
    stem_groups: Dict[str, Set[str]],
) -> List[InternalEdgeRecord]:
    G, _ = build_unrooted_graph(tree)
    fragment_to_known_stem: Dict[frozenset[str], str] = {
        frozenset(taxa): group_name for group_name, taxa in stem_groups.items()
    }
    records: List[InternalEdgeRecord] = []

    for idx, (u, v, edge_len) in enumerate(internal_edges_from_graph(G), start=1):
        edge_id = f"internal_edge_{idx}"
        H = G.copy()
        H.remove_edge(u, v)
        components = list(nx.connected_components(H))
        if len(components) != 2:
            continue

        comp_a, comp_b = components
        if u not in comp_a:
            comp_a, comp_b = comp_b, comp_a

        tip_names_a, tip_distances_a = component_tip_names_and_distances(H, comp_a, u)
        tip_names_b, tip_distances_b = component_tip_names_and_distances(H, comp_b, v)

        side_a_set = frozenset(tip_names_a)
        side_b_set = frozenset(tip_names_b)
        matched_known_stem_group = fragment_to_known_stem.get(side_a_set) or fragment_to_known_stem.get(side_b_set)

        median_a = median(tip_distances_a)
        median_b = median(tip_distances_b)

        records.append(
            InternalEdgeRecord(
                edge_id=edge_id,
                edge_len=edge_len,
                matched_known_stem_group=matched_known_stem_group,
                node_a=node_name_for_graph(G, u),
                node_b=node_name_for_graph(G, v),
                tip_names_a=sorted(tip_names_a),
                tip_names_b=sorted(tip_names_b),
                median_a=median_a if median_a is not None else float("nan"),
                median_b=median_b if median_b is not None else float("nan"),
            )
        )

    return records


def internal_edges_in_taxon_subset(
    edge_records: List[InternalEdgeRecord],
    taxon_subset: Set[str],
) -> List[InternalEdgeRecord]:
    subset_edges: List[InternalEdgeRecord] = []
    for rec in edge_records:
        if rec.matched_known_stem_group is not None:
            continue
        side_a = set(rec.tip_names_a)
        side_b = set(rec.tip_names_b)
        if side_a <= taxon_subset or side_b <= taxon_subset:
            subset_edges.append(rec)
    return subset_edges


def detect_internal_branch_events_for_tree(
    tree: Tree,
    tree_id: str,
    stem_groups: Dict[str, Set[str]],
    cross_gene_stats: CrossGeneStats,
    internal_branch_ratio_threshold: float,
    internal_side_ratio_threshold: float,
    internal_path_consistency_threshold: float,
    min_path_branches_total: int,
    global_internal_rescue_quantile: Optional[float],
) -> Tuple[
    List[InternalBranchRow],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Set[str],
    Set[str],
    List[InternalBranchRow],
]:
    rows: List[InternalBranchRow] = []
    taxon_to_drop_edges: Dict[str, List[str]] = defaultdict(list)
    taxon_to_rescued_edges: Dict[str, List[str]] = defaultdict(list)
    dropped_taxa: Set[str] = set()
    rescued_edge_ids: Set[str] = set()
    untested_internal_rows: List[InternalBranchRow] = []

    global_internal_median = median(cross_gene_stats.all_internal_lengths)
    global_internal_cutoff = select_global_quantile_cutoff(
        cross_gene_stats.all_internal_lengths,
        global_internal_rescue_quantile,
    )
    edge_records = build_internal_edge_records(tree, tree_id, stem_groups)

    for rec in edge_records:
        tested_by_known_stem = rec.matched_known_stem_group is not None
        tested_by_global_internal = rec.matched_known_stem_group is None
        ever_tested_internal = tested_by_known_stem or tested_by_global_internal
        untested_internal_warning = NA_STR if ever_tested_internal else "NEVER_TESTED_BY_STEM_OR_GLOBAL_INTERNAL"

        candidate_outlier = False
        rescued_by_global_internal = False
        drop_applied = False
        global_internal_ratio = float("nan")
        deleted_side = NA_STR
        candidate_taxa: List[str] = []
        downstream_internal_branch_count = 0
        downstream_long_branch_count = 0
        downstream_long_fraction = float("nan")
        pruning_blocked_reason = NA_STR

        if tested_by_known_stem:
            pruning_blocked_reason = "matched_known_stem_group"
        elif tested_by_global_internal:
            if global_internal_median is None or global_internal_median <= EPS:
                pruning_blocked_reason = "no_global_internal_background"
            else:
                global_internal_ratio = rec.edge_len / global_internal_median
                candidate_outlier = global_internal_ratio > internal_branch_ratio_threshold

                if not candidate_outlier:
                    pruning_blocked_reason = "not_internal_outlier"
                elif (
                    global_internal_rescue_quantile is not None
                    and not math.isnan(global_internal_cutoff)
                    and rec.edge_len < global_internal_cutoff
                ):
                    rescued_by_global_internal = True
                    rescued_edge_ids.add(rec.edge_id)
                    pruning_blocked_reason = "rescued_by_global_internal"
                else:
                    median_a = rec.median_a
                    median_b = rec.median_b

                    side_ratio = float("nan")
                    if (
                        not math.isnan(median_a)
                        and not math.isnan(median_b)
                        and min(median_a, median_b) > EPS
                    ):
                        side_ratio = max(median_a, median_b) / min(median_a, median_b)

                    if math.isnan(side_ratio) or side_ratio <= internal_side_ratio_threshold:
                        pruning_blocked_reason = "side_ratio_below_threshold"
                    else:
                        if median_a > median_b:
                            deleted_side = "A"
                            candidate_taxa = rec.tip_names_a
                        elif median_b > median_a:
                            deleted_side = "B"
                            candidate_taxa = rec.tip_names_b
                        else:
                            pruning_blocked_reason = "median_tie_no_decision"

                    if candidate_taxa:
                        # Conservative safeguard: internal pruning is only allowed when the
                        # candidate deleted side is the smaller side of the split.
                        candidate_size = len(candidate_taxa)
                        other_size = len(rec.tip_names_b) if deleted_side == "A" else len(rec.tip_names_a)
                        if candidate_size >= other_size:
                            pruning_blocked_reason = "candidate_side_not_smaller"
                            candidate_taxa = []

                    if candidate_taxa:
                        downstream = [
                            r for r in internal_edges_in_taxon_subset(edge_records, set(candidate_taxa))
                            if r.edge_id != rec.edge_id
                        ]
                        n_downstream = len(downstream)

                        required_downstream = max(0, min_path_branches_total - 1)
                        if n_downstream < required_downstream:
                            pruning_blocked_reason = f"downstream_internal_branches_lt_{min_path_branches_total}_including_tested"
                        else:
                            n_long_downstream = 0
                            for downstream_rec in downstream:
                                ratio = downstream_rec.edge_len / global_internal_median
                                if ratio > internal_branch_ratio_threshold:
                                    n_long_downstream += 1

                            n_total = 1 + n_downstream
                            n_long = 1 + n_long_downstream
                            fraction = n_long / n_total

                            downstream_internal_branch_count = n_total
                            downstream_long_branch_count = n_long
                            downstream_long_fraction = fraction

                            if fraction >= internal_path_consistency_threshold:
                                drop_applied = True
                                pruning_blocked_reason = NA_STR
                                for taxon in candidate_taxa:
                                    taxon_to_drop_edges[taxon].append(rec.edge_id)
                                dropped_taxa.update(candidate_taxa)
                            else:
                                pruning_blocked_reason = "downstream_fraction_below_threshold"

        if rescued_by_global_internal and candidate_taxa:
            for taxon in candidate_taxa:
                taxon_to_rescued_edges[taxon].append(rec.edge_id)

        row = InternalBranchRow(
            tree_id=tree_id,
            edge_id=rec.edge_id,
            node_a=rec.node_a,
            node_b=rec.node_b,
            branch_length=rec.edge_len,
            matched_known_stem_group=rec.matched_known_stem_group or NA_STR,
            tested_by_known_stem=tested_by_known_stem,
            tested_by_global_internal=tested_by_global_internal,
            ever_tested_internal=ever_tested_internal,
            untested_internal_warning=untested_internal_warning,
            global_internal_median_branch=global_internal_median if global_internal_median is not None else float("nan"),
            global_internal_ratio=global_internal_ratio,
            side_a_tip_count=len(rec.tip_names_a),
            side_b_tip_count=len(rec.tip_names_b),
            side_a_median_split_to_tip=rec.median_a,
            side_b_median_split_to_tip=rec.median_b,
            deleted_side=deleted_side,
            candidate_taxa=candidate_taxa,
            downstream_internal_branch_count=downstream_internal_branch_count,
            downstream_long_branch_count=downstream_long_branch_count,
            downstream_long_fraction=downstream_long_fraction,
            candidate_outlier=candidate_outlier,
            rescued_by_global_internal=rescued_by_global_internal,
            drop_applied=drop_applied,
            pruning_blocked_reason=pruning_blocked_reason,
            global_internal_q25_branch=cross_gene_stats.global_q25_internal_branch,
            global_internal_q50_branch=cross_gene_stats.global_q50_internal_branch,
            global_internal_q75_branch=cross_gene_stats.global_q75_internal_branch,
            global_internal_rescue_quantile=format_optional_quantile(global_internal_rescue_quantile),
            global_internal_rescue_cutoff=global_internal_cutoff,
        )
        rows.append(row)
        if not ever_tested_internal:
            untested_internal_rows.append(row)

    return (
        rows,
        {taxon: sorted(v) for taxon, v in taxon_to_drop_edges.items()},
        {taxon: sorted(v) for taxon, v in taxon_to_rescued_edges.items()},
        dropped_taxa,
        rescued_edge_ids,
        untested_internal_rows,
    )


def compute_display_category(
    stem_drop: bool,
    internal_branch_drop: bool,
    global_stem_rescue_membership: bool,
    global_internal_rescue_membership: bool,
    global_terminal_rescue_applied: bool,
    quartet_drop: bool,
    cross_gene_terminal_drop: bool,
) -> str:
    # Terminal rescue must override quartet/global-terminal display when the final
    # decision is KEEP.
    if global_terminal_rescue_applied:
        return "global_terminal_rescue"

    # True drop categories take precedence over rescue memberships.
    if internal_branch_drop:
        return "internal_branch_drop"
    if stem_drop:
        return "stem_drop"

    # Rescue memberships for kept taxa.
    if global_stem_rescue_membership:
        return "global_stem_rescue"
    if global_internal_rescue_membership:
        return "global_internal_rescue"

    # Remaining terminal-only flags.
    if cross_gene_terminal_drop:
        return "terminal_drop"
    if quartet_drop:
        return "quartet_drop"

    return "default"


def score_taxa(ctx: TreeContext, ratio_threshold: float) -> List[TaxonResult]:
    results: List[TaxonResult] = []
    mandatory_by_taxon = build_taxon_to_mandatory_stem_outlier_groups(
        ctx.taxon_to_stem_groups,
        ctx.mandatory_stem_drop_groups,
    )
    rescued_stem_by_taxon = build_taxon_to_rescued_stem_groups(
        ctx.stem_groups,
        ctx.rescued_stem_groups,
        ctx.mandatory_stem_drop_groups,
    )

    global_terminal_median = median(ctx.cross_gene_medians.values())

    for tip in ctx.tree.get_terminals():
        if not tip.name:
            continue

        reasons: List[str] = []
        applied_rules: List[str] = []
        branch_length = branch_length_to_parent(tip)

        stem_groups = ctx.taxon_to_stem_groups.get(tip.name, [])
        clade_stem_outlier_groups = mandatory_by_taxon.get(tip.name, [])
        clade_stem_rescued_groups = rescued_stem_by_taxon.get(tip.name, [])
        internal_drop_edges = ctx.taxon_to_internal_drop_edges.get(tip.name, [])
        internal_rescued_edges = ctx.taxon_to_internal_rescued_edges.get(tip.name, [])

        stem_drop = len(clade_stem_outlier_groups) > 0
        global_stem_rescue_membership = len(clade_stem_rescued_groups) > 0
        internal_branch_drop = len(internal_drop_edges) > 0
        global_internal_rescue_membership = len(internal_rescued_edges) > 0

        if stem_drop:
            applied_rules.append("cross_gene_clade_stem_outlier")
            for group_name in clade_stem_outlier_groups:
                reasons.append(f"cross_gene_clade_stem_outlier:{group_name}")

        if internal_branch_drop:
            applied_rules.append("global_internal_branch_outlier")
            for edge_id in internal_drop_edges:
                reasons.append(f"global_internal_branch_outlier:{edge_id}")

        cross_gene_obs = ctx.cross_gene_counts.get(tip.name, 0)
        cross_gene_med = ctx.cross_gene_medians.get(tip.name, float("nan"))

        if ctx.cross_gene_enabled:
            cross_gene_drop, cross_gene_ratio, cross_gene_hybrid_cutoff = should_drop_terminal_by_cross_gene(
                branch_length,
                cross_gene_med,
                cross_gene_obs,
                ctx.cross_gene_ratio_threshold,
                global_terminal_median if global_terminal_median is not None else float("nan"),
                ctx.hybrid_terminal,
            )
        else:
            cross_gene_drop = False
            cross_gene_ratio = float("nan")
            cross_gene_hybrid_cutoff = float("nan")

        if cross_gene_drop:
            if ctx.hybrid_terminal and not math.isnan(cross_gene_hybrid_cutoff):
                reasons.append(f"cross_gene_terminal_hybrid_outlier:cutoff={cross_gene_hybrid_cutoff:.6g}")
                applied_rules.append("cross_gene_terminal_hybrid_outlier")
            else:
                reasons.append("cross_gene_terminal_branch_outlier")
                applied_rules.append("cross_gene_terminal_branch_outlier")

        choice, choice_reason = find_mask_aware_quartet_for_tip(ctx, tip)
        anchor_label = "no_valid_quartet"
        effective_length = float("nan")
        local_median = float("nan")
        local_ratio = float("nan")
        comparator_taxa: List[str] = []
        quartet_drop = False

        if choice is not None:
            anchor = choice.anchor
            comparators = choice.comparators
            anchor_label = node_label(ctx, anchor)
            comparator_taxa = [c.name for c in comparators]
            comp_vals = [float(ctx.tree.distance(anchor, comp)) for comp in comparators]
            effective_length = float(ctx.tree.distance(anchor, tip))
            local_ref_med = median(comp_vals)
            local_median = local_ref_med if local_ref_med is not None else float("nan")
            if local_median > EPS:
                local_ratio = effective_length / local_median
                if local_ratio >= ratio_threshold:
                    quartet_drop = True
                    reasons.append("quartet_local_ratio_outlier")
                    applied_rules.append("quartet_local_ratio_outlier")

        non_branch_candidate_drop = quartet_drop or cross_gene_drop
        global_terminal_rescue_applied = False

        if stem_drop or internal_branch_drop:
            decision = "DROP"
            reason = ";".join(reasons)
        elif choice is None:
            if non_branch_candidate_drop:
                if (
                    ctx.global_rescue_quantile is not None
                    and not math.isnan(ctx.global_rescue_cutoff)
                    and branch_length < ctx.global_rescue_cutoff
                ):
                    reasons.append(f"below_global_{quantile_reason_suffix(ctx.global_rescue_quantile)}_override")
                    applied_rules.append("global_terminal_rescue")
                    global_terminal_rescue_applied = True
                    decision = "KEEP"
                else:
                    decision = "DROP"
                reason = ";".join(reasons)
            else:
                decision = "FLAG"
                reason = choice_reason
        else:
            if non_branch_candidate_drop:
                if (
                    ctx.global_rescue_quantile is not None
                    and not math.isnan(ctx.global_rescue_cutoff)
                    and branch_length < ctx.global_rescue_cutoff
                ):
                    reasons.append(f"below_global_{quantile_reason_suffix(ctx.global_rescue_quantile)}_override")
                    applied_rules.append("global_terminal_rescue")
                    global_terminal_rescue_applied = True
                    decision = "KEEP"
                else:
                    decision = "DROP"
                reason = ";".join(reasons)
            else:
                decision = "KEEP"
                reason = "not_locally_abnormal"

        if global_stem_rescue_membership:
            applied_rules.append("global_stem_rescue_membership")
        if global_internal_rescue_membership:
            applied_rules.append("global_internal_rescue_membership")

        display_category = compute_display_category(
            stem_drop,
            internal_branch_drop,
            global_stem_rescue_membership,
            global_internal_rescue_membership,
            global_terminal_rescue_applied,
            quartet_drop,
            cross_gene_drop,
        )

        results.append(
            TaxonResult(
                tree_id=ctx.tree_id,
                taxon=tip.name,
                anchor_label=anchor_label,
                branch_length=branch_length,
                effective_length=effective_length,
                local_median=local_median,
                local_ratio=local_ratio,
                cross_gene_method=cross_gene_method_name(ctx.cross_gene_enabled, ctx.hybrid_terminal, ctx.hybrid_stem),
                cross_gene_obs=cross_gene_obs,
                cross_gene_median_branch=cross_gene_med,
                cross_gene_ratio=cross_gene_ratio,
                stem_group_count=len(stem_groups),
                stem_outlier_group_count=len(clade_stem_outlier_groups),
                stem_rescued_group_count=len(clade_stem_rescued_groups),
                internal_drop_edge_count=len(internal_drop_edges),
                internal_rescued_edge_count=len(internal_rescued_edges),
                global_terminal_q25_branch=ctx.global_q25_terminal_branch,
                global_terminal_q50_branch=ctx.global_q50_terminal_branch,
                global_terminal_q75_branch=ctx.global_q75_terminal_branch,
                global_rescue_quantile=format_optional_quantile(ctx.global_rescue_quantile),
                global_rescue_cutoff=ctx.global_rescue_cutoff,
                global_stem_q25_branch=ctx.global_q25_stem_branch,
                global_stem_q50_branch=ctx.global_q50_stem_branch,
                global_stem_q75_branch=ctx.global_q75_stem_branch,
                global_stem_rescue_quantile=format_optional_quantile(ctx.global_stem_rescue_quantile),
                global_stem_rescue_cutoff=ctx.global_stem_rescue_cutoff,
                global_internal_q25_branch=ctx.global_q25_internal_branch,
                global_internal_q50_branch=ctx.global_q50_internal_branch,
                global_internal_q75_branch=ctx.global_q75_internal_branch,
                global_internal_rescue_quantile=format_optional_quantile(ctx.global_internal_rescue_quantile),
                global_internal_rescue_cutoff=ctx.global_internal_rescue_cutoff,
                comparator_taxa=comparator_taxa,
                decision=decision,
                reason=reason,
                quartet_drop=quartet_drop,
                cross_gene_terminal_drop=cross_gene_drop,
                stem_drop=stem_drop,
                internal_branch_drop=internal_branch_drop,
                global_terminal_rescue_applied=global_terminal_rescue_applied,
                global_stem_rescue_membership=global_stem_rescue_membership,
                global_internal_rescue_membership=global_internal_rescue_membership,
                applied_rules=sorted(set(applied_rules)),
                display_category=display_category,
                multiple_rules_applied=(len(set(applied_rules)) > 1),
            )
        )
    return results


def build_stem_membership_rows(
    tree_id: str,
    taxon_to_stem_groups: Dict[str, List[str]],
    stem_eval: Dict[str, StemEvalRecord],
) -> List[StemMembershipRow]:
    rows: List[StemMembershipRow] = []
    for taxon in sorted(taxon_to_stem_groups):
        for group_name in taxon_to_stem_groups[taxon]:
            rec = stem_eval[group_name]
            rows.append(
                StemMembershipRow(
                    tree_id,
                    taxon,
                    group_name,
                    rec.obs_count,
                    rec.stem_branch,
                    rec.median_branch,
                    rec.ratio,
                    rec.tested,
                    rec.outlier,
                    rec.rescued_by_global_stem,
                    rec.skipped_due_to_ancestor_outlier or NA_STR,
                )
            )
    return rows


def write_drop_file(tree_path: Path, results: List[TaxonResult]) -> None:
    with tree_path.with_suffix(".out").open("w") as handle:
        for row in results:
            if row.decision == "DROP":
                handle.write(row.taxon + "\n")


def write_taxon_results(path: Path, results: List[TaxonResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tree_id",
            "taxon",
            "anchor_label",
            "branch_length",
            "effective_length",
            "local_median",
            "local_ratio",
            "cross_gene_method",
            "cross_gene_obs",
            "cross_gene_median_branch",
            "cross_gene_ratio",
            "stem_group_count",
            "stem_outlier_group_count",
            "stem_rescued_group_count",
            "internal_drop_edge_count",
            "internal_rescued_edge_count",
            "global_terminal_q25_branch",
            "global_terminal_q50_branch",
            "global_terminal_q75_branch",
            "global_rescue_quantile",
            "global_rescue_cutoff",
            "global_stem_q25_branch",
            "global_stem_q50_branch",
            "global_stem_q75_branch",
            "global_stem_rescue_quantile",
            "global_stem_rescue_cutoff",
            "global_internal_q25_branch",
            "global_internal_q50_branch",
            "global_internal_q75_branch",
            "global_internal_rescue_quantile",
            "global_internal_rescue_cutoff",
            "comparator_taxa",
            "quartet_drop",
            "cross_gene_terminal_drop",
            "stem_drop",
            "internal_branch_drop",
            "global_terminal_rescue_applied",
            "global_stem_rescue_membership",
            "global_internal_rescue_membership",
            "display_category",
            "multiple_rules_applied",
            "applied_rules",
            "decision",
            "reason",
        ])
        for row in results:
            writer.writerow([
                format_scalar(row.tree_id),
                format_scalar(row.taxon),
                format_scalar(row.anchor_label),
                format_scalar(row.branch_length),
                format_scalar(row.effective_length),
                format_scalar(row.local_median),
                format_scalar(row.local_ratio),
                format_scalar(row.cross_gene_method),
                format_scalar(row.cross_gene_obs),
                format_scalar(row.cross_gene_median_branch),
                format_scalar(row.cross_gene_ratio),
                format_scalar(row.stem_group_count),
                format_scalar(row.stem_outlier_group_count),
                format_scalar(row.stem_rescued_group_count),
                format_scalar(row.internal_drop_edge_count),
                format_scalar(row.internal_rescued_edge_count),
                format_scalar(row.global_terminal_q25_branch),
                format_scalar(row.global_terminal_q50_branch),
                format_scalar(row.global_terminal_q75_branch),
                format_scalar(row.global_rescue_quantile),
                format_scalar(row.global_rescue_cutoff),
                format_scalar(row.global_stem_q25_branch),
                format_scalar(row.global_stem_q50_branch),
                format_scalar(row.global_stem_q75_branch),
                format_scalar(row.global_stem_rescue_quantile),
                format_scalar(row.global_stem_rescue_cutoff),
                format_scalar(row.global_internal_q25_branch),
                format_scalar(row.global_internal_q50_branch),
                format_scalar(row.global_internal_q75_branch),
                format_scalar(row.global_internal_rescue_quantile),
                format_scalar(row.global_internal_rescue_cutoff),
                format_scalar(";".join(row.comparator_taxa) if row.comparator_taxa else NA_STR),
                format_scalar(row.quartet_drop),
                format_scalar(row.cross_gene_terminal_drop),
                format_scalar(row.stem_drop),
                format_scalar(row.internal_branch_drop),
                format_scalar(row.global_terminal_rescue_applied),
                format_scalar(row.global_stem_rescue_membership),
                format_scalar(row.global_internal_rescue_membership),
                format_scalar(row.display_category),
                format_scalar(row.multiple_rules_applied),
                format_scalar(";".join(row.applied_rules) if row.applied_rules else NA_STR),
                format_scalar(row.decision),
                format_scalar(row.reason),
            ])


def write_stem_membership_results(path: Path, rows: List[StemMembershipRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tree_id",
            "taxon",
            "stem_group",
            "obs_count",
            "stem_branch",
            "median_branch",
            "ratio",
            "tested",
            "outlier",
            "rescued_by_global_stem",
            "skipped_due_to_ancestor_outlier",
        ])
        for row in rows:
            writer.writerow([
                format_scalar(row.tree_id),
                format_scalar(row.taxon),
                format_scalar(row.stem_group),
                format_scalar(row.obs_count),
                format_scalar(row.stem_branch),
                format_scalar(row.median_branch),
                format_scalar(row.ratio),
                format_scalar(row.tested),
                format_scalar(row.outlier),
                format_scalar(row.rescued_by_global_stem),
                format_scalar(row.skipped_due_to_ancestor_outlier),
            ])


def write_internal_branch_results(path: Path, rows: List[InternalBranchRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tree_id",
            "edge_id",
            "node_a",
            "node_b",
            "branch_length",
            "matched_known_stem_group",
            "tested_by_known_stem",
            "tested_by_global_internal",
            "ever_tested_internal",
            "untested_internal_warning",
            "global_internal_median_branch",
            "global_internal_ratio",
            "side_a_tip_count",
            "side_b_tip_count",
            "side_a_median_split_to_tip",
            "side_b_median_split_to_tip",
            "deleted_side",
            "candidate_taxa",
            "downstream_internal_branch_count",
            "downstream_long_branch_count",
            "downstream_long_fraction",
            "candidate_outlier",
            "rescued_by_global_internal",
            "drop_applied",
            "pruning_blocked_reason",
            "global_internal_q25_branch",
            "global_internal_q50_branch",
            "global_internal_q75_branch",
            "global_internal_rescue_quantile",
            "global_internal_rescue_cutoff",
        ])
        for row in rows:
            writer.writerow([
                format_scalar(row.tree_id),
                format_scalar(row.edge_id),
                format_scalar(row.node_a),
                format_scalar(row.node_b),
                format_scalar(row.branch_length),
                format_scalar(row.matched_known_stem_group),
                format_scalar(row.tested_by_known_stem),
                format_scalar(row.tested_by_global_internal),
                format_scalar(row.ever_tested_internal),
                format_scalar(row.untested_internal_warning),
                format_scalar(row.global_internal_median_branch),
                format_scalar(row.global_internal_ratio),
                format_scalar(row.side_a_tip_count),
                format_scalar(row.side_b_tip_count),
                format_scalar(row.side_a_median_split_to_tip),
                format_scalar(row.side_b_median_split_to_tip),
                format_scalar(row.deleted_side),
                format_scalar(";".join(row.candidate_taxa) if row.candidate_taxa else NA_STR),
                format_scalar(row.downstream_internal_branch_count),
                format_scalar(row.downstream_long_branch_count),
                format_scalar(row.downstream_long_fraction),
                format_scalar(row.candidate_outlier),
                format_scalar(row.rescued_by_global_internal),
                format_scalar(row.drop_applied),
                format_scalar(row.pruning_blocked_reason),
                format_scalar(row.global_internal_q25_branch),
                format_scalar(row.global_internal_q50_branch),
                format_scalar(row.global_internal_q75_branch),
                format_scalar(row.global_internal_rescue_quantile),
                format_scalar(row.global_internal_rescue_cutoff),
            ])


def write_untested_internal_branches(
    base_dir: Path,
    rows: List[InternalBranchRow],
    single_tree_stem: Optional[str],
) -> Path:
    if single_tree_stem is None:
        path = base_dir / "untested_internal_branches.txt"
    else:
        path = base_dir / f"{single_tree_stem}.untested_internal_branches.txt"
    with path.open("w") as handle:
        for row in rows:
            handle.write(f"{row.tree_id}\t{row.edge_id}\t{row.node_a}\t{row.node_b}\n")
    return path


def figtree_color_for_result(row: TaxonResult) -> str:
    return FIGTREE_COLOR_HEX.get(row.display_category, FIGTREE_COLOR_HEX["default"])


def _tip_annotation_label(name: str, row: Optional[TaxonResult]) -> str:
    color = figtree_color_for_result(row) if row else FIGTREE_COLOR_HEX["default"]
    multiple = "yes" if row and row.multiple_rules_applied else "no"
    applied_rules = ";".join(row.applied_rules) if row and row.applied_rules else NA_STR
    display_category = row.display_category if row else "default"
    decision = row.decision if row else NA_STR
    reason = (row.reason if row else NA_STR).replace('"', "'")
    return (
        f"'{name}'[&!color={color},display_category=\"{display_category}\","
        f"applied_rules=\"{applied_rules}\",multiple_rules=\"{multiple}\","
        f"decision=\"{decision}\",reason=\"{reason}\"]"
    )


def write_coloured_tree(tree_path: Path, tree: Tree, results: List[TaxonResult]) -> None:
    by_taxon = {r.taxon: r for r in results}
    outfile = tree_path.with_name(f"{tree_path.stem}.coloured.nex")
    with outfile.open("w") as handle:
        handle.write("#NEXUS\n")
        handle.write("begin taxa;\n")
        terminals = [tip for tip in tree.get_terminals() if tip.name]
        handle.write(f"  dimensions ntax={len(terminals)};\n")
        handle.write("  taxlabels\n")
        for tip in terminals:
            handle.write(f"    {_tip_annotation_label(tip.name, by_taxon.get(tip.name))}\n")
        handle.write("  ;\nend;\n\nbegin trees;\n  tree TREE1 = ")

        def clade_to_newick(clade: Clade) -> str:
            if clade.is_terminal():
                label = _tip_annotation_label(clade.name or "unnamed", by_taxon.get(clade.name or ""))
                return f"{label}:{clade.branch_length}" if clade.branch_length is not None else label
            children = ",".join(clade_to_newick(child) for child in clade.clades)
            label = clade.name or ""
            if clade.branch_length is not None:
                return f"({children}){label}:{clade.branch_length}"
            return f"({children}){label}"

        handle.write(clade_to_newick(tree.root) + ";\nend;\n")


def print_cross_gene_warnings(tree_count: int) -> None:
    if tree_count <= CROSS_GENE_DISABLE_TREES:
        print(
            f"WARNING: only {tree_count} gene tree(s) detected. Cross-gene terminal, stem, and internal "
            "branch tests are disabled because the dataset is too small for meaningful cross-gene estimates. "
            "Only quartet logic and global terminal-length rescue will be used."
        )
    elif tree_count < WARN_VERY_FEW_TREES:
        print(
            f"WARNING: only {tree_count} gene tree(s) detected. Cross-gene statistics are very unreliable "
            "and may be misleading."
        )
    elif tree_count < WARN_FEW_TREES:
        print(
            f"Warning: cross-gene statistics are based on fewer than {WARN_FEW_TREES} gene trees "
            f"({tree_count}). Results may be unstable."
        )


def print_insufficient_cross_gene_taxa(counts: Dict[str, int]) -> None:
    insufficient = sorted(
        [(taxon, n) for taxon, n in counts.items() if n < MIN_CROSS_GENE_OBS],
        key=lambda x: (x[1], x[0]),
    )
    if not insufficient:
        return
    print(f"\nCross-gene terminal-branch rule not evaluable for {len(insufficient)} taxa (fewer than {MIN_CROSS_GENE_OBS} trees):")
    for taxon, n in insufficient:
        print(f"  {taxon} ({n} tree{'s' if n != 1 else ''})")


def print_insufficient_clade_groups(counts: Dict[str, int], requested_groups: Iterable[str]) -> None:
    insufficient = [
        (group_name, counts.get(group_name, 0))
        for group_name in sorted(set(requested_groups))
        if counts.get(group_name, 0) < MIN_CROSS_GENE_OBS
    ]
    if not insufficient:
        return
    print(f"\nCross-gene clade-stem rule not evaluable for {len(insufficient)} clade(s) (fewer than {MIN_CROSS_GENE_OBS} trees with a retained fragment):")
    for group_name, n in insufficient:
        print(f"  {group_name} ({n} tree{'s' if n != 1 else ''})")


def print_final_drop_statistics(all_results: List[TaxonResult]) -> None:
    total_rows = len(all_results)
    total_drops = sum(1 for r in all_results if r.decision == "DROP")
    print("\nFinal drop statistics:")
    if total_rows == 0:
        print("  No taxa were scored.")
        return
    print(f"  Total scored taxon instances: {total_rows}")
    print(f"  Total dropped taxon instances: {total_drops} ({100.0 * total_drops / total_rows:.2f}%)")
    by_tree: Dict[str, List[TaxonResult]] = defaultdict(list)
    for row in all_results:
        by_tree[row.tree_id].append(row)
    print("  Per gene/tree:")
    for tree_id in sorted(by_tree):
        rows = by_tree[tree_id]
        n_total = len(rows)
        n_drop = sum(1 for r in rows if r.decision == "DROP")
        print(f"    {tree_id}: {n_drop}/{n_total} dropped ({100.0 * n_drop / n_total:.2f}%)")


def print_filtering_summary(
    quartet_terminal_removed: Set[Tuple[str, str]],
    global_terminal_removed: Set[Tuple[str, str]],
    stem_removed_clades: Set[Tuple[str, str]],
    internal_removed_clades: Set[Tuple[str, str]],
    internal_subtrees_removed: int,
) -> None:
    terminal_unique = quartet_terminal_removed | global_terminal_removed
    clade_unique = stem_removed_clades | internal_removed_clades

    print("\n=== FILTERING SUMMARY ===")
    print("\nTerminal branches removed:")
    print(f"  - by quartet rule: {len(quartet_terminal_removed)}")
    print(f"  - by global length rule: {len(global_terminal_removed)}")
    print(f"  - total (unique): {len(terminal_unique)}  [unique taxa removed; not the sum of above]")

    print("\nClades removed:")
    print(f"  - by stem rule: {len(stem_removed_clades)}")
    print(f"  - by internal rule: {len(internal_removed_clades)}")
    print(f"  - total (unique): {len(clade_unique)}  [unique clades removed; not the sum of above]")

    print("\nInternal branch removals:")
    print(f"  - subtrees removed: {internal_subtrees_removed}")


def process_tree(
    tree_path: Path,
    ratio_threshold: float,
    cross_gene_ratio_threshold: float,
    clade_stem_ratio_threshold: float,
    internal_branch_ratio_threshold: float,
    internal_side_ratio_threshold: float,
    internal_path_consistency_threshold: float,
    quartet_long_branch_defs: Optional[Dict[str, Set[str]]],
    stem_test_defs: Optional[Dict[str, Set[str]]],
    cross_gene_stats: CrossGeneStats,
    global_rescue_quantile: Optional[float],
    global_stem_rescue_quantile: Optional[float],
    global_internal_rescue_quantile: Optional[float],
    cross_gene_enabled: bool,
    internal_min_path_branches: int,
    hybrid_terminal: bool,
    hybrid_stem: bool,
) -> Tuple[
    Tree,
    List[TaxonResult],
    List[StemMembershipRow],
    List[InternalBranchRow],
    List[Tuple[str, int]],
    Dict[str, StemEvalRecord],
    Set[str],
    Set[str],
    Set[str],
    Set[str],
    List[InternalBranchRow],
    Dict[str, Set[str]],
    Set[str],
    Dict[str, List[str]],
]:
    tree = load_tree(tree_path)
    parent_map = build_parent_map(tree)
    node_ids = assign_node_ids(tree)

    quartet_mask_groups: Dict[str, Set[str]] = {}
    quartet_mono_clades: List[Tuple[str, int]] = []
    if quartet_long_branch_defs:
        quartet_mask_info = detect_mask_groups_from_graph(
            tree=tree,
            clade_defs=quartet_long_branch_defs,
            min_size=MIN_MASKING_CLADE_SIZE,
        )
        quartet_mask_groups = quartet_mask_info.mask_groups
        quartet_mono_clades = quartet_mask_info.monophyletic_clades

    stem_groups: Dict[str, Set[str]] = {}
    current_clade_stems: Dict[str, float] = {}
    fragment_aliases: Dict[str, List[str]] = {}
    if stem_test_defs:
        stem_raw = detect_mask_groups_from_graph(
            tree=tree,
            clade_defs=stem_test_defs,
            min_size=MIN_MASKING_CLADE_SIZE,
        )
        stem_groups, current_clade_stems, fragment_aliases = collapse_identical_stem_fragments(
            raw_groups=stem_raw.mask_groups,
            raw_stems=stem_raw.stem_lengths,
            clade_defs=stem_test_defs,
        )

    taxon_to_quartet_mask_group = build_taxon_to_group(quartet_mask_groups)
    taxon_to_stem_groups = build_taxon_to_groups(stem_groups)

    if cross_gene_enabled:
        stem_eval, mandatory_stem_drop_groups, rescued_stem_groups = evaluate_stem_groups_for_tree(
            stem_groups,
            current_clade_stems,
            cross_gene_stats,
            clade_stem_ratio_threshold,
            global_stem_rescue_quantile,
            hybrid_stem,
        )
        (
            internal_rows,
            taxon_to_internal_drop_edges,
            taxon_to_internal_rescued_edges,
            dropped_by_internal,
            rescued_internal_edges,
            untested_internal_rows,
        ) = detect_internal_branch_events_for_tree(
            tree=tree,
            tree_id=tree_path.stem,
            stem_groups=stem_groups,
            cross_gene_stats=cross_gene_stats,
            internal_branch_ratio_threshold=internal_branch_ratio_threshold,
            internal_side_ratio_threshold=internal_side_ratio_threshold,
            internal_path_consistency_threshold=internal_path_consistency_threshold,
            min_path_branches_total=internal_min_path_branches,
            global_internal_rescue_quantile=global_internal_rescue_quantile,
        )
    else:
        stem_eval = {}
        mandatory_stem_drop_groups = set()
        rescued_stem_groups = set()
        internal_rows = []
        taxon_to_internal_drop_edges = {}
        taxon_to_internal_rescued_edges = {}
        dropped_by_internal = set()
        rescued_internal_edges = set()
        untested_internal_rows = []

    internal_removed_clades = {
        group_name
        for group_name, taxa in stem_groups.items()
        if taxa and taxa <= dropped_by_internal
    }

    ctx = TreeContext(
        tree=tree,
        tree_id=tree_path.stem,
        parent_map=parent_map,
        node_ids=node_ids,
        quartet_mask_groups=quartet_mask_groups,
        taxon_to_quartet_mask_group=taxon_to_quartet_mask_group,
        stem_groups=stem_groups,
        taxon_to_stem_groups=taxon_to_stem_groups,
        current_clade_stems=current_clade_stems,
        stem_eval=stem_eval,
        mandatory_stem_drop_groups=mandatory_stem_drop_groups,
        rescued_stem_groups=rescued_stem_groups,
        cross_gene_medians=cross_gene_stats.taxon_medians,
        cross_gene_counts=cross_gene_stats.taxon_counts,
        clade_stem_medians=cross_gene_stats.clade_stem_medians,
        clade_stem_counts=cross_gene_stats.clade_stem_counts,
        cross_gene_ratio_threshold=cross_gene_ratio_threshold,
        clade_stem_ratio_threshold=clade_stem_ratio_threshold,
        cross_gene_enabled=cross_gene_enabled,
        hybrid_terminal=hybrid_terminal,
        hybrid_stem=hybrid_stem,
        global_q25_terminal_branch=cross_gene_stats.global_q25_terminal_branch,
        global_q50_terminal_branch=cross_gene_stats.global_q50_terminal_branch,
        global_q75_terminal_branch=cross_gene_stats.global_q75_terminal_branch,
        global_rescue_quantile=global_rescue_quantile,
        global_rescue_cutoff=select_global_quantile_cutoff(
            cross_gene_stats.all_terminal_lengths,
            global_rescue_quantile,
        ),
        global_q25_stem_branch=cross_gene_stats.global_q25_stem_branch,
        global_q50_stem_branch=cross_gene_stats.global_q50_stem_branch,
        global_q75_stem_branch=cross_gene_stats.global_q75_stem_branch,
        global_stem_rescue_quantile=global_stem_rescue_quantile,
        global_stem_rescue_cutoff=select_global_quantile_cutoff(
            cross_gene_stats.all_retained_stems,
            global_stem_rescue_quantile,
        ),
        global_q25_internal_branch=cross_gene_stats.global_q25_internal_branch,
        global_q50_internal_branch=cross_gene_stats.global_q50_internal_branch,
        global_q75_internal_branch=cross_gene_stats.global_q75_internal_branch,
        global_internal_rescue_quantile=global_internal_rescue_quantile,
        global_internal_rescue_cutoff=select_global_quantile_cutoff(
            cross_gene_stats.all_internal_lengths,
            global_internal_rescue_quantile,
        ),
        taxon_to_internal_drop_edges=taxon_to_internal_drop_edges,
        taxon_to_internal_rescued_edges=taxon_to_internal_rescued_edges,
    )

    rows = score_taxa(ctx, ratio_threshold)
    stem_membership_rows = build_stem_membership_rows(tree_path.stem, taxon_to_stem_groups, stem_eval) if stem_eval else []

    return (
        tree,
        rows,
        stem_membership_rows,
        internal_rows,
        quartet_mono_clades,
        stem_eval,
        mandatory_stem_drop_groups,
        rescued_stem_groups,
        dropped_by_internal,
        rescued_internal_edges,
        untested_internal_rows,
        stem_groups,
        internal_removed_clades,
        fragment_aliases,
    )


def expand_clade_aliases(
    representative_clades: Set[str],
    fragment_aliases: Dict[str, List[str]],
) -> Set[str]:
    expanded: Set[str] = set()
    for clade_name in representative_clades:
        for alias in fragment_aliases.get(clade_name, [clade_name]):
            expanded.add(alias)
    return expanded


def build_family_drop_tokens(
    representative_status: Dict[str, str],
    fragment_aliases: Dict[str, List[str]],
) -> List[str]:
    tokens: List[str] = []
    for rep in sorted(representative_status):
        tokens.append(f"{rep}_{representative_status[rep]}")
        for alias in fragment_aliases.get(rep, []):
            if alias == rep:
                continue
            tokens.append(f"{alias}_ALIAS")
    return tokens



def write_dropped_clades_per_family(path: Path, family_to_drop_tokens: Dict[str, List[str]]) -> None:
    with path.open("w") as handle:
        for family in sorted(family_to_drop_tokens):
            tokens = family_to_drop_tokens[family]
            if tokens:
                handle.write(f"{family} {' '.join(tokens)}\n")
            else:
                handle.write(f"{family}\n")


def write_clade_deletion_histogram(
    representative_counter: Counter,
    alias_counter: Counter,
    outpath: Path,
    title: str,
    xlabel: str,
) -> None:
    labels = sorted(set(representative_counter) | set(alias_counter))
    if not labels:
        return
    rep_values = [representative_counter.get(label, 0) for label in labels]
    alias_values = [alias_counter.get(label, 0) for label in labels]

    plt.figure(figsize=(max(8, 0.5 * len(labels)), 5))
    x = list(range(len(labels)))
    plt.bar(x, rep_values, label="Representative clade")
    plt.bar(x, alias_values, bottom=rep_values, label="Alias clade")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Deletion count")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def write_deletion_frequency_histogram(counter: Counter, outpath: Path, title: str, xlabel: str) -> None:
    if not counter:
        return
    labels = list(sorted(counter))
    values = [counter[l] for l in labels]
    plt.figure(figsize=(max(8, 0.5 * len(labels)), 5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Deletion count")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quartet-based long-branch filter with optional masking, cross-gene terminal tests, "
            "known clade-stem tests, conservative non-stem internal-branch tests, global rescue, "
            "coloured trees, and deletion diagnostics."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tree", type=Path, help="Single Newick tree file.")
    group.add_argument("--trees", type=Path, help="Directory of Newick tree files.")
    parser.add_argument("--tip_ratio_threshold", type=float, default=3.0)
    parser.add_argument("--cross_gene_ratio_threshold", type=float, default=3.0)
    parser.add_argument("--clade_stem_ratio_threshold", type=float, default=3.0)
    parser.add_argument("--internal_branch_ratio_threshold", type=float, default=3.0)
    parser.add_argument("--internal_side_ratio_threshold", type=float, default=1.2)
    parser.add_argument("--internal_min_path_branches", type=int, default=DEFAULT_INTERNAL_MIN_PATH_BRANCHES, help="Minimum total number of internal branches on the tested path, including the tested branch itself. Default: 3")
    parser.add_argument(
        "--internal_path_consistency_threshold",
        type=quantile_arg,
        default=0.5,
        help=(
            "For non-stem internal pruning, require at least this fraction (0 to 1) of long branches "
            "along the tested path (including the tested branch) for pruning to occur. Default: 0.5"
        ),
    )
    parser.add_argument("--hybrid_terminal", action="store_true", help="Use hybrid trigger for cross-gene terminal detection.")
    parser.add_argument("--hybrid_stem", action="store_true", help="Use hybrid trigger for stem detection.")
    parser.add_argument("--long_branch_clades", type=Path)
    parser.add_argument("--stem_to_test", type=Path)
    parser.add_argument("--global_rescue", type=quantile_arg)
    parser.add_argument("--global_stem_rescue", type=quantile_arg)
    parser.add_argument("--global_internal_rescue", type=quantile_arg)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.internal_min_path_branches < 1:
        raise SystemExit("--internal_min_path_branches must be >= 1")

    quartet_long_branch_defs = parse_named_clades(args.long_branch_clades) if args.long_branch_clades else None
    stem_test_defs = parse_named_clades(args.stem_to_test) if args.stem_to_test else None

    if args.tree is not None:
        tree_files = [args.tree]
        output_dir = args.tree.parent
        single_stem = args.tree.stem
        taxon_results_path = output_dir / f"{args.tree.stem}.csv"
        stem_results_path = output_dir / f"{args.tree.stem}.stem_memberships.csv"
        internal_results_path = output_dir / f"{args.tree.stem}.internal_branches.csv"
        dropped_clades_path = output_dir / f"{args.tree.stem}.dropped_clades_per_family.txt"
        clade_del_hist_path = output_dir / f"{args.tree.stem}.clade_deletion_frequencies.png"
        taxon_loss_hist_path = output_dir / f"{args.tree.stem}.taxon_loss_frequencies.png"
    else:
        tree_files = iter_tree_files(args.trees)
        if not tree_files:
            raise SystemExit(f"No tree files found in {args.trees}")
        output_dir = args.trees
        single_stem = None
        taxon_results_path = output_dir / "taxon_results.csv"
        stem_results_path = output_dir / "taxon_stem_memberships.csv"
        internal_results_path = output_dir / "internal_branch_results.csv"
        dropped_clades_path = output_dir / "dropped_clades_per_family.txt"
        clade_del_hist_path = output_dir / "clade_deletion_frequencies.png"
        taxon_loss_hist_path = output_dir / "taxon_loss_frequencies.png"

    tree_count = len(tree_files)
    cross_gene_enabled = tree_count > CROSS_GENE_DISABLE_TREES

    print_cross_gene_warnings(tree_count)
    cross_gene_stats = collect_branch_and_clade_stats(
        tree_files=tree_files,
        stem_test_defs=stem_test_defs,
        progress=args.progress,
    )

    if cross_gene_enabled:
        print_insufficient_cross_gene_taxa(cross_gene_stats.taxon_counts)
        if stem_test_defs:
            print_insufficient_clade_groups(cross_gene_stats.clade_stem_counts, stem_test_defs.keys())
        terminal_mode = "hybrid" if args.hybrid_terminal else "median_ratio"
        stem_mode = "hybrid" if args.hybrid_stem else "median_ratio"
        print(f"\nCross-gene terminal method: {terminal_mode}")
        print(f"Cross-gene stem method: {stem_mode}")
        print("Whole-tree internal branch method: non_stem_global_internal_ratio + side asymmetry + path consistency safeguard")
        print("Quartet terminal method: local ratio")
        print(f"Minimum total internal branches required on tested path (including focal branch): {args.internal_min_path_branches}")
        if math.isnan(cross_gene_stats.global_q50_internal_branch):
            print("WARNING: no non-stem internal branches were available to build the global internal-branch background.")
    else:
        print("\nCross-gene terminal/stem/internal methods: disabled_small_dataset")
        print("Whole-tree internal branch method: disabled_small_dataset")
        print("Global terminal rescue remains enabled if --global_rescue is provided.")

    all_results: List[TaxonResult] = []
    all_stem_rows: List[StemMembershipRow] = []
    all_internal_rows: List[InternalBranchRow] = []
    all_untested_internal_rows: List[InternalBranchRow] = []

    family_to_drop_tokens: Dict[str, List[str]] = defaultdict(list)
    clade_rep_delete_counter: Counter = Counter()
    clade_alias_delete_counter: Counter = Counter()
    taxon_loss_counter: Counter = Counter()

    quartet_terminal_removed: Set[Tuple[str, str]] = set()
    global_terminal_removed: Set[Tuple[str, str]] = set()
    stem_removed_clade_instances: Set[Tuple[str, str]] = set()
    internal_removed_clade_instances: Set[Tuple[str, str]] = set()
    internal_subtrees_removed_count = 0

    for i, tree_file in enumerate(tree_files, start=1):
        if args.progress:
            print(f"\nProcessing trees: {i}/{len(tree_files)} ({tree_file.stem})", flush=True)

        (
            tree,
            rows,
            stem_membership_rows,
            internal_rows,
            quartet_mono_clades,
            stem_eval,
            mandatory_stem_drop_groups,
            rescued_stem_groups,
            dropped_by_internal,
            rescued_internal_edges,
            untested_internal_rows,
            stem_groups,
            internal_removed_clades,
            fragment_aliases,
        ) = process_tree(
            tree_path=tree_file,
            ratio_threshold=args.tip_ratio_threshold,
            cross_gene_ratio_threshold=args.cross_gene_ratio_threshold,
            clade_stem_ratio_threshold=args.clade_stem_ratio_threshold,
            internal_branch_ratio_threshold=args.internal_branch_ratio_threshold,
            internal_side_ratio_threshold=args.internal_side_ratio_threshold,
            internal_path_consistency_threshold=args.internal_path_consistency_threshold,
            quartet_long_branch_defs=quartet_long_branch_defs,
            stem_test_defs=stem_test_defs,
            cross_gene_stats=cross_gene_stats,
            global_rescue_quantile=args.global_rescue,
            global_stem_rescue_quantile=args.global_stem_rescue,
            global_internal_rescue_quantile=args.global_internal_rescue,
            cross_gene_enabled=cross_gene_enabled,
            internal_min_path_branches=args.internal_min_path_branches,
            hybrid_terminal=args.hybrid_terminal,
            hybrid_stem=args.hybrid_stem,
        )

        all_results.extend(rows)
        all_stem_rows.extend(stem_membership_rows)
        all_internal_rows.extend(internal_rows)
        all_untested_internal_rows.extend(untested_internal_rows)

        write_drop_file(tree_file, rows)
        write_coloured_tree(tree_file, tree, rows)

        drops = [r for r in rows if r.decision == "DROP"]
        dropped_taxa_this_tree = sorted({r.taxon for r in drops})
        for taxon in dropped_taxa_this_tree:
            taxon_loss_counter[taxon] += 1

        for row in drops:
            key = (tree_file.stem, row.taxon)
            if row.quartet_drop:
                quartet_terminal_removed.add(key)
            if row.cross_gene_terminal_drop:
                global_terminal_removed.add(key)

        stem_rep_clades_this_tree = set(mandatory_stem_drop_groups)
        internal_rep_clades_this_tree = set(internal_removed_clades)

        stem_clades_this_tree = expand_clade_aliases(stem_rep_clades_this_tree, fragment_aliases)
        internal_clades_this_tree = expand_clade_aliases(internal_rep_clades_this_tree, fragment_aliases)

        for clade_name in stem_clades_this_tree:
            stem_removed_clade_instances.add((tree_file.stem, clade_name))
        for clade_name in internal_clades_this_tree:
            internal_removed_clade_instances.add((tree_file.stem, clade_name))

        representative_status: Dict[str, str] = {}
        unique_rep_clades_this_tree = stem_rep_clades_this_tree | internal_rep_clades_this_tree
        for clade_name in sorted(unique_rep_clades_this_tree):
            if clade_name in stem_rep_clades_this_tree and clade_name in internal_rep_clades_this_tree:
                status = "BOTHWAYDROP"
            elif clade_name in stem_rep_clades_this_tree:
                status = "CR"
            else:
                status = "IBR"
            representative_status[clade_name] = status
            clade_rep_delete_counter[clade_name] += 1

        for rep in sorted(unique_rep_clades_this_tree):
            for alias in fragment_aliases.get(rep, []):
                if alias != rep:
                    clade_alias_delete_counter[alias] += 1

        family_to_drop_tokens[tree_file.stem] = build_family_drop_tokens(representative_status, fragment_aliases)

        internal_drops = [r for r in internal_rows if r.drop_applied]
        internal_subtrees_removed_count += len(internal_drops)

        print(f"\nTree: {tree_file.stem}")
        if quartet_long_branch_defs:
            print("  Monophyletic long-branch clades:" if quartet_mono_clades else "  Monophyletic long-branch clades: none")
            for cname, size in quartet_mono_clades:
                print(f"    {cname} ({size} taxa)")

        if cross_gene_enabled and stem_test_defs:
            if stem_eval:
                print("  Stem-test clades:")
                for cname in sorted(stem_eval):
                    rec = stem_eval[cname]
                    size = len(rec.taxa)
                    if rec.skipped_due_to_ancestor_outlier is not None:
                        print(f"    {cname} ({size} taxa, skipped_nested_under_outlier={rec.skipped_due_to_ancestor_outlier})")
                    else:
                        metric = f"stem_ratio={rec.ratio:.3f}" if not math.isnan(rec.ratio) else ""
                        status = ", OUTLIER" if rec.outlier else ""
                        rescue_status = (
                            f", rescued_by_global_stem_q{int(round(args.global_stem_rescue * 100))}"
                            if rec.rescued_by_global_stem and args.global_stem_rescue is not None
                            else ""
                        )
                        comma = ", " if metric else ""
                        print(f"    {cname} ({size} taxa{comma}{metric}{status}{rescue_status})")
            else:
                print("  Stem-test clades: none")

        internal_known = [r for r in internal_rows if r.matched_known_stem_group != NA_STR]
        internal_tested = [r for r in internal_rows if r.tested_by_global_internal]
        internal_outliers = [r for r in internal_rows if r.candidate_outlier]
        internal_blocked = [r for r in internal_rows if r.pruning_blocked_reason not in {NA_STR, "not_internal_outlier"}]

        flags = [r for r in rows if r.decision == "FLAG"]
        rescued = [r for r in rows if r.decision == "KEEP" and "below_global_q" in r.reason]

        print(f"  Internal branches matched to known stems: {', '.join(f'{r.edge_id}->{r.matched_known_stem_group}' for r in internal_known) if internal_known else 'none'}")
        print(f"  Internal branches tested globally: {', '.join(r.edge_id for r in internal_tested) if internal_tested else 'none'}")
        print(f"  Internal-branch outliers: {', '.join(f'{r.edge_id} (ratio={r.global_internal_ratio:.3f})' for r in internal_outliers) if internal_outliers else 'none'}")
        print(f"  Internal branches blocked from pruning: {', '.join(f'{r.edge_id} ({r.pruning_blocked_reason})' for r in internal_blocked) if internal_blocked else 'none'}")
        if args.global_internal_rescue is not None:
            print(f"  Internal branches rescued by global internal rescue: {', '.join(sorted(rescued_internal_edges)) if rescued_internal_edges else 'none'}")
        print(f"  Internal branches applied for pruning: {', '.join(r.edge_id for r in internal_drops) if internal_drops else 'none'}")
        print(f"  Completely untested internal branches: {len(untested_internal_rows)}")
        print(f"  Taxa dropped by internal-branch criterion: {', '.join(sorted(dropped_by_internal)) if dropped_by_internal else 'none'}")
        print(f"  Dropped taxa: {', '.join(f'{r.taxon} ({r.reason})' for r in drops) if drops else 'none'}")
        print(f"  Flagged taxa: {', '.join(f'{r.taxon} ({r.reason})' for r in flags) if flags else 'none'}")
        print(f"  Retained by global rescue: {', '.join(f'{r.taxon} ({r.reason})' for r in rescued) if rescued else 'none'}")
        print(f"  Dropped clades by stem criterion: {', '.join(sorted(stem_rep_clades_this_tree)) if stem_rep_clades_this_tree else 'none'}")
        print(f"  Fully removed clades by internal criterion: {', '.join(sorted(internal_rep_clades_this_tree)) if internal_rep_clades_this_tree else 'none'}")
        if args.global_stem_rescue is not None:
            print(f"  Retained clades by global stem rescue: {', '.join(sorted(rescued_stem_groups)) if rescued_stem_groups else 'none'}")

    write_taxon_results(taxon_results_path, all_results)
    write_stem_membership_results(stem_results_path, all_stem_rows)
    write_internal_branch_results(internal_results_path, all_internal_rows)
    untested_internal_path = write_untested_internal_branches(output_dir, all_untested_internal_rows, single_stem)
    write_dropped_clades_per_family(dropped_clades_path, family_to_drop_tokens)
    write_clade_deletion_histogram(clade_rep_delete_counter, clade_alias_delete_counter, clade_del_hist_path, "Clade deletion frequency (representative + alias reporting)", "Clade")
    write_deletion_frequency_histogram(taxon_loss_counter, taxon_loss_hist_path, "Taxon loss frequency (unique per tree across all rules)", "Taxon")

    print_final_drop_statistics(all_results)
    print_filtering_summary(
        quartet_terminal_removed=quartet_terminal_removed,
        global_terminal_removed=global_terminal_removed,
        stem_removed_clades=stem_removed_clade_instances,
        internal_removed_clades=internal_removed_clade_instances,
        internal_subtrees_removed=internal_subtrees_removed_count,
    )

    print(f"\nProcessed {len(tree_files)} tree(s)")
    print(f"Wrote: {taxon_results_path}")
    print(f"Wrote: {stem_results_path}")
    print(f"Wrote: {internal_results_path}")
    print(f"Wrote: {untested_internal_path}")
    print(f"Wrote: {dropped_clades_path}")
    if clade_rep_delete_counter or clade_alias_delete_counter:
        print(f"Wrote: {clade_del_hist_path}")
    if taxon_loss_counter:
        print(f"Wrote: {taxon_loss_hist_path}")


if __name__ == "__main__":
    main()
