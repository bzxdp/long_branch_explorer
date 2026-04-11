#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree

EPS = 1e-12
MIN_MASKING_CLADE_SIZE = 2
TREE_EXTS = {'.nwk', '.tree', '.tre', '.newick', '.txt', '.treefile'}
NA_STR = 'NA'
PERCENTILES_FOR_VIOLIN_LINES = [0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99]
DEFAULT_K_GRID = [4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2.0]
DEFAULT_P_GRID = [0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90]
OPTIMAL_TRIGGER_COLOR = '#d62728'
PERCENTILE_REF_COLOR = 'grey'
VIOLIN_FACE_COLOR = 'blue'
ALT_K_LOW_COLOR = '#ff7f00'
ALT_K_HIGH_COLOR = '#ffff00'


@dataclass
class MaskDetectionResult:
    mask_groups: Dict[str, Set[str]]
    stem_lengths: Dict[str, float]
    monophyletic_clades: List[Tuple[str, int]]


@dataclass
class StemDistributionRow:
    tree_id: str
    stem_group: str
    stem_branch: float


@dataclass
class TerminalDistributionRow:
    tree_id: str
    taxon: str
    branch_length: float


@dataclass
class InternalDistributionRow:
    tree_id: str
    edge_id: str
    branch_length: float


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


@dataclass
class CandidateResult:
    k_value: float
    percentile_cutoff: float
    cutoff_value: float
    n_flagged_rows: int
    n_flagged_units: int
    total_bl_dropped: float
    total_excess_dropped: float
    raw_gain: Optional[float]
    excess_gain: Optional[float]
    flagged_units: Tuple[str, ...]
    flagged_row_keys: Tuple[Tuple[str, str, float], ...]


@dataclass
class EnvelopePoint:
    n_flagged_events: int
    candidate: CandidateResult
    gain_value: float


@dataclass
class SummaryStats:
    median: Optional[float]
    mean: Optional[float]
    q75: Optional[float]
    q80: Optional[float]
    q85: Optional[float]
    q90: Optional[float]
    q93: Optional[float]
    q95: Optional[float]
    q96: Optional[float]
    q97: Optional[float]
    q98: Optional[float]
    q99: Optional[float]


@dataclass
class AutomationArtifacts:
    output_prefix: str
    files_written: List[Path]
    console_lines: List[str]


def median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.median(vals) if vals else None


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.mean(vals) if vals else None


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


def format_scalar(value) -> str:
    if value is None:
        return NA_STR
    if isinstance(value, float) and math.isnan(value):
        return NA_STR
    return str(value)


def parse_comma_floats(value: str) -> List[float]:
    try:
        parsed = [float(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Must be a comma-separated list of numbers, e.g. 3,4,5,6"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError(
            "Must provide at least one comma-separated number, e.g. 3,4,5,6"
        )
    return parsed


def load_tree(path: Path) -> Tree:
    return Phylo.read(str(path), 'newick')


def iter_tree_files(trees_dir: Path) -> List[Path]:
    return sorted(p for p in trees_dir.iterdir() if p.is_file() and p.suffix.lower() in TREE_EXTS)


def branch_length_to_parent(clade: Clade) -> float:
    return float(clade.branch_length) if clade.branch_length is not None else 0.0


def parse_named_clades(path: Path) -> Dict[str, Set[str]]:
    clades: Dict[str, Set[str]] = {}
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(
                    f"Invalid line {line_number} in {path}: missing ':'. Expected format 'Clade: taxon1 taxon2'."
                )
            name, taxa = line.split(':', 1)
            if ',' in taxa:
                raise ValueError(f"Comma detected in taxa list for '{name.strip()}' on line {line_number}. Use spaces only.")
            taxa_set = {x.strip() for x in taxa.split() if x.strip()}
            if not taxa_set:
                raise ValueError(f"No taxa found for clade '{name.strip()}' on line {line_number}.")
            clades[name.strip()] = taxa_set
    return clades


def build_networkx_tree(tree: Tree) -> Tuple[nx.Graph, int]:
    graph = nx.Graph()
    clade_to_id: Dict[Clade, int] = {}
    for idx, clade in enumerate(tree.find_clades(order='level')):
        clade_to_id[clade] = idx
        graph.add_node(idx, clade=clade, is_tip=clade.is_terminal(), name=clade.name if clade.is_terminal() else None)
    for parent in tree.find_clades(order='level'):
        parent_id = clade_to_id[parent]
        for child in parent.clades:
            child_id = clade_to_id[child]
            graph.add_edge(parent_id, child_id, length=branch_length_to_parent(child))
    return graph, clade_to_id[tree.root]


def contract_degree_two_root(graph: nx.Graph, root_id: int) -> nx.Graph:
    contracted = graph.copy()
    if root_id not in contracted:
        return contracted
    neighbors = list(contracted.neighbors(root_id))
    if len(neighbors) != 2:
        return contracted
    a, b = neighbors
    new_len = float(contracted[root_id][a].get('length', 0.0)) + float(contracted[root_id][b].get('length', 0.0))
    contracted.remove_node(root_id)
    if contracted.has_edge(a, b):
        contracted[a][b]['length'] = min(float(contracted[a][b].get('length', 0.0)), new_len)
    else:
        contracted.add_edge(a, b, length=new_len)
    return contracted


def build_unrooted_graph(tree: Tree) -> Tuple[nx.Graph, int]:
    raw_graph, root_id = build_networkx_tree(tree)
    return contract_degree_two_root(raw_graph, root_id), root_id


def internal_edges_from_graph(graph: nx.Graph) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    for u, v, data in graph.edges(data=True):
        if graph.nodes[u].get('is_tip') or graph.nodes[v].get('is_tip'):
            continue
        edges.append((u, v, float(data.get('length', 0.0))))
    edges.sort(key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))
    return edges


def tip_names_in_component(graph: nx.Graph, component: Set[int]) -> Set[str]:
    return {graph.nodes[node]['name'] for node in component if graph.nodes[node].get('name')}


def detect_mask_groups_from_graph(tree: Tree, clade_defs: Dict[str, Set[str]], min_size: int = MIN_MASKING_CLADE_SIZE) -> MaskDetectionResult:
    mask_groups: Dict[str, Set[str]] = {}
    stem_lengths: Dict[str, float] = {}
    monophyletic_clades: List[Tuple[str, int]] = []
    if not clade_defs:
        return MaskDetectionResult(mask_groups, stem_lengths, monophyletic_clades)
    raw_graph, root_id = build_networkx_tree(tree)
    graph = contract_degree_two_root(raw_graph, root_id)
    all_named_tips = {data['name'] for _, data in graph.nodes(data=True) if data.get('name')}
    for clade_name, taxa in clade_defs.items():
        present_taxa = taxa & all_named_tips
        if len(present_taxa) < min_size:
            continue
        best_fragment: Set[str] = set()
        best_stem_length = float('nan')
        for u, v, data in list(graph.edges(data=True)):
            cut_graph = graph.copy()
            cut_graph.remove_edge(u, v)
            components = list(nx.connected_components(cut_graph))
            if len(components) != 2:
                continue
            comp1, comp2 = components
            side1 = tip_names_in_component(cut_graph, comp1)
            side2 = tip_names_in_component(cut_graph, comp2)
            edge_len = float(data.get('length', 0.0))
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


def component_tip_names_and_distances(
    graph: nx.Graph,
    component: Set[int],
    start_node: int,
) -> Tuple[List[str], List[float]]:
    subgraph = graph.subgraph(component).copy()
    lengths = nx.single_source_dijkstra_path_length(subgraph, source=start_node, weight='length')
    tip_names: List[str] = []
    tip_distances: List[float] = []
    for node in sorted(component):
        if subgraph.nodes[node].get('is_tip') and subgraph.nodes[node].get('name'):
            tip_names.append(subgraph.nodes[node]['name'])
            tip_distances.append(float(lengths.get(node, float('nan'))))
    return tip_names, [d for d in tip_distances if not math.isnan(d)]


def node_name_for_graph(graph: nx.Graph, node_id: int) -> str:
    if graph.nodes[node_id].get('is_tip'):
        return graph.nodes[node_id].get('name') or f'tip_{node_id}'
    return f'node_{node_id}'


def build_internal_edge_records(
    tree: Tree,
    tree_id: str,
    stem_groups: Dict[str, Set[str]],
) -> List[InternalEdgeRecord]:
    graph, _ = build_unrooted_graph(tree)
    fragment_to_known_stem: Dict[frozenset[str], str] = {
        frozenset(taxa): group_name for group_name, taxa in stem_groups.items()
    }
    records: List[InternalEdgeRecord] = []

    for idx, (u, v, edge_len) in enumerate(internal_edges_from_graph(graph), start=1):
        edge_id = f'internal_edge_{idx}'
        cut_graph = graph.copy()
        cut_graph.remove_edge(u, v)
        components = list(nx.connected_components(cut_graph))
        if len(components) != 2:
            continue

        comp_a, comp_b = components
        if u not in comp_a:
            comp_a, comp_b = comp_b, comp_a

        tip_names_a, tip_distances_a = component_tip_names_and_distances(cut_graph, comp_a, u)
        tip_names_b, tip_distances_b = component_tip_names_and_distances(cut_graph, comp_b, v)

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
                node_a=node_name_for_graph(graph, u),
                node_b=node_name_for_graph(graph, v),
                tip_names_a=sorted(tip_names_a),
                tip_names_b=sorted(tip_names_b),
                median_a=median_a if median_a is not None else float('nan'),
                median_b=median_b if median_b is not None else float('nan'),
            )
        )
    return records


def build_terminal_distribution_rows(tree_id: str, tree: Tree) -> List[TerminalDistributionRow]:
    rows: List[TerminalDistributionRow] = []
    for tip in tree.get_terminals():
        if tip.name:
            rows.append(TerminalDistributionRow(tree_id=tree_id, taxon=tip.name, branch_length=branch_length_to_parent(tip)))
    return rows


def write_stem_distribution_csv(path: Path, rows: List[StemDistributionRow]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tree_id', 'stem_group', 'stem_branch'])
        for row in rows:
            writer.writerow([row.tree_id, row.stem_group, row.stem_branch])


def write_terminal_distribution_csv(path: Path, rows: List[TerminalDistributionRow]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tree_id', 'taxon', 'branch_length'])
        for row in rows:
            writer.writerow([row.tree_id, row.taxon, row.branch_length])


def write_internal_distribution_csv(path: Path, rows: List[InternalDistributionRow]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tree_id', 'edge_id', 'branch_length'])
        for row in rows:
            writer.writerow([row.tree_id, row.edge_id, row.branch_length])


def calculate_summary_stats(values: List[float]) -> SummaryStats:
    return SummaryStats(
        median=median(values),
        mean=mean(values),
        q75=quantile(values, 0.75),
        q80=quantile(values, 0.80),
        q85=quantile(values, 0.85),
        q90=quantile(values, 0.90),
        q93=quantile(values, 0.93),
        q95=quantile(values, 0.95),
        q96=quantile(values, 0.96),
        q97=quantile(values, 0.97),
        q98=quantile(values, 0.98),
        q99=quantile(values, 0.99),
    )


def _interpolate_hex_color(color_a: str, color_b: str, t: float) -> str:
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    a = _hex_to_rgb(color_a)
    b = _hex_to_rgb(color_b)
    rgb = tuple(int(round(a[i] * (1.0 - t) + b[i] * t)) for i in range(3))
    return _rgb_to_hex(rgb)


def alternative_k_colors(k_values: List[float]) -> Dict[float, str]:
    if not k_values:
        return {}
    ordered = sorted(set(k_values))
    if len(ordered) == 1:
        return {ordered[0]: ALT_K_LOW_COLOR}
    colors: Dict[float, str] = {}
    for idx, k_value in enumerate(ordered):
        t = idx / (len(ordered) - 1)
        colors[k_value] = _interpolate_hex_color(ALT_K_LOW_COLOR, ALT_K_HIGH_COLOR, t)
    return colors


def plot_histogram(
    values: List[float],
    outpath: Path,
    title: str,
    xlabel: str,
    suggested_percentile_value: Optional[float],
) -> None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean, bins=min(40, max(10, int(math.sqrt(len(clean))))))
    med = median(clean)
    if med is not None:
        ax.axvline(med, color='black', linewidth=1.8, label='Global median')
    if suggested_percentile_value is not None and not math.isnan(suggested_percentile_value):
        ax.axvline(suggested_percentile_value, color=OPTIMAL_TRIGGER_COLOR, linestyle=':', linewidth=2.0, label='Suggested percentile cutoff (EXCESS)')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_internal_histogram(
    values: List[float],
    outpath: Path,
    title: str,
    xlabel: str,
    suggested_percentile_value: Optional[float],
    global_median: float,
    chosen_candidate: Optional[CandidateResult],
    alternative_cutoffs: List[float],
) -> None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean, bins=min(40, max(10, int(math.sqrt(len(clean))))))

    med = median(clean)
    if med is not None:
        ax.axvline(med, color='black', linewidth=1.8, label='Global median')
    if suggested_percentile_value is not None and not math.isnan(suggested_percentile_value):
        ax.axvline(suggested_percentile_value, color=OPTIMAL_TRIGGER_COLOR, linestyle=':', linewidth=2.0, label='Suggested percentile cutoff (EXCESS)')

    alt_k_values = [k for k in alternative_cutoffs if chosen_candidate is None or not math.isclose(k, chosen_candidate.k_value, rel_tol=1e-12, abs_tol=1e-12)]
    alt_colors = alternative_k_colors(alt_k_values)

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks([])
    top_ax.set_xlabel('Internal trigger positions (K × global median)')

    y_top = 1.0
    if chosen_candidate is not None:
        chosen_x = chosen_candidate.k_value * global_median
        top_ax.scatter([chosen_x], [y_top], transform=top_ax.get_xaxis_transform(), color=OPTIMAL_TRIGGER_COLOR, s=55, zorder=5)
        top_ax.text(chosen_x, 1.04, f'K={chosen_candidate.k_value:g}', color=OPTIMAL_TRIGGER_COLOR, ha='center', va='bottom', transform=top_ax.get_xaxis_transform(), fontsize=8)

    for k_value in sorted(alt_colors):
        x = k_value * global_median
        top_ax.scatter([x], [y_top], transform=top_ax.get_xaxis_transform(), color=alt_colors[k_value], edgecolors='black', linewidths=0.4, s=42, zorder=4)
        top_ax.text(x, 1.04, f'{k_value:g}', color=alt_colors[k_value], ha='center', va='bottom', transform=top_ax.get_xaxis_transform(), fontsize=8)

    legend_items = [
        Line2D([0], [0], color='black', lw=1.8, label='Global median'),
        Line2D([0], [0], color=OPTIMAL_TRIGGER_COLOR, lw=2.0, linestyle=':', label='Suggested percentile cutoff'),
    ]
    if chosen_candidate is not None:
        legend_items.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=OPTIMAL_TRIGGER_COLOR, markeredgecolor=OPTIMAL_TRIGGER_COLOR, markersize=7, label=f'Chosen K={chosen_candidate.k_value:g}'))
    if alt_k_values:
        legend_items.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=ALT_K_HIGH_COLOR, markeredgecolor='black', markersize=6, label='Alternative K values'))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.legend(handles=legend_items, frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def evaluate_candidates_generic(
    rows: List[Tuple[str, str, float]],
    local_medians: Dict[str, float],
    global_median: float,
    global_values: List[float],
    k_grid: List[float],
    p_grid: List[float],
    trigger_func: Callable[[float, Optional[float], float], Optional[float]],
    progress: bool = False,
    label: str = 'optimisation',
) -> List[CandidateResult]:
    candidates: List[CandidateResult] = []
    total_loops = len(k_grid) * len(p_grid)
    loop_index = 0
    percentile_to_cutoff: Dict[float, float] = {}
    for p in p_grid:
        cutoff = quantile(global_values, p)
        if cutoff is None or math.isnan(cutoff):
            raise ValueError(f'Could not compute global cutoff for percentile {p}')
        percentile_to_cutoff[p] = cutoff

    for k_value in k_grid:
        previous_signature: Optional[Tuple[Tuple[str, str, float], ...]] = None
        for p in p_grid:
            loop_index += 1
            cutoff_value = percentile_to_cutoff[p]
            flagged_rows: List[Tuple[str, str, float]] = []
            total_excess = 0.0
            for tree_id, unit_name, branch_length in rows:
                local_med = local_medians.get(unit_name)
                trigger = trigger_func(k_value, local_med, global_median)
                if trigger is None or math.isnan(trigger):
                    continue
                if branch_length > trigger and branch_length >= cutoff_value:
                    flagged_rows.append((tree_id, unit_name, branch_length))
                    total_excess += (branch_length - trigger)
            flagged_units = tuple(sorted({unit for _, unit, _ in flagged_rows}))
            flagged_row_keys = tuple(sorted((tree_id, unit_name, round(branch_length, 12)) for tree_id, unit_name, branch_length in flagged_rows))
            n_flagged_rows = len(flagged_rows)
            n_flagged_units = len(flagged_units)
            total_bl = sum(branch_length for _, _, branch_length in flagged_rows)
            raw_gain = (total_bl / n_flagged_rows) if n_flagged_rows > 0 else None
            excess_gain = (total_excess / n_flagged_rows) if n_flagged_rows > 0 else None
            candidates.append(CandidateResult(
                k_value=k_value,
                percentile_cutoff=p,
                cutoff_value=cutoff_value,
                n_flagged_rows=n_flagged_rows,
                n_flagged_units=n_flagged_units,
                total_bl_dropped=total_bl,
                total_excess_dropped=total_excess,
                raw_gain=raw_gain,
                excess_gain=excess_gain,
                flagged_units=flagged_units,
                flagged_row_keys=flagged_row_keys,
            ))
            if progress:
                print(f'[{label}] {loop_index}/{total_loops} K={k_value:g} P={p:.2f} deleted_rows={n_flagged_rows} unique_groups={n_flagged_units}', flush=True)
            if previous_signature is not None and flagged_row_keys == previous_signature and n_flagged_rows > 0:
                if progress:
                    print(f'[{label}] plateau reached for K={k_value:g} at P={p:.2f}; stopping lower P scan for this K', flush=True)
                break
            previous_signature = flagged_row_keys
    return candidates


def candidate_is_stricter(a: CandidateResult, b: CandidateResult) -> bool:
    if not math.isclose(a.k_value, b.k_value):
        return a.k_value > b.k_value
    if not math.isclose(a.percentile_cutoff, b.percentile_cutoff):
        return a.percentile_cutoff > b.percentile_cutoff
    return a.n_flagged_rows < b.n_flagged_rows


def build_gain_envelope(candidates: List[CandidateResult], gain_attr: str) -> List[EnvelopePoint]:
    grouped: Dict[int, CandidateResult] = {}
    for cand in candidates:
        gain_value = getattr(cand, gain_attr)
        if gain_value is None or math.isnan(gain_value):
            continue
        best = grouped.get(cand.n_flagged_rows)
        if best is None:
            grouped[cand.n_flagged_rows] = cand
            continue
        best_gain = getattr(best, gain_attr)
        if best_gain is None or gain_value > best_gain + EPS:
            grouped[cand.n_flagged_rows] = cand
        elif math.isclose(gain_value, best_gain, rel_tol=1e-12, abs_tol=1e-12) and candidate_is_stricter(cand, best):
            grouped[cand.n_flagged_rows] = cand
    return [EnvelopePoint(n, c, getattr(c, gain_attr)) for n, c in sorted(grouped.items())]


def choose_elbow_candidate(envelope: List[EnvelopePoint]) -> Optional[CandidateResult]:
    if not envelope:
        return None
    if len(envelope) == 1:
        return envelope[0].candidate
    if len(envelope) == 2:
        return envelope[0].candidate if envelope[0].gain_value >= envelope[1].gain_value else envelope[1].candidate
    xs = [float(point.n_flagged_events) for point in envelope]
    ys = [float(point.gain_value) for point in envelope]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
        return max(envelope, key=lambda p: p.gain_value).candidate
    norm_points = [((x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y)) for x, y in zip(xs, ys)]
    x1, y1 = norm_points[0]
    x2, y2 = norm_points[-1]
    denom = math.hypot(y2 - y1, x2 - x1)
    if denom <= EPS:
        return max(envelope, key=lambda p: p.gain_value).candidate
    best_idx = 0
    best_dist = -1.0
    for idx, (x0, y0) in enumerate(norm_points):
        dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denom
        if dist > best_dist + EPS:
            best_dist = dist
            best_idx = idx
        elif math.isclose(dist, best_dist, rel_tol=1e-12, abs_tol=1e-12) and candidate_is_stricter(envelope[idx].candidate, envelope[best_idx].candidate):
            best_idx = idx
    return envelope[best_idx].candidate


def candidate_by_k_p(candidates: List[CandidateResult], k_value: float, p_value: float) -> Optional[CandidateResult]:
    for cand in candidates:
        if math.isclose(cand.k_value, k_value, rel_tol=1e-12, abs_tol=1e-12) and math.isclose(cand.percentile_cutoff, p_value, rel_tol=1e-12, abs_tol=1e-12):
            return cand
    return None


def classify_regime(candidates: List[CandidateResult], chosen: Optional[CandidateResult], k_grid: List[float], p_grid: List[float]) -> Tuple[str, List[float], List[float]]:
    if chosen is None:
        return 'no-choice', [], []
    chosen_signature = chosen.flagged_row_keys
    equivalent_ks: List[float] = []
    for k in k_grid:
        cand = candidate_by_k_p(candidates, k, chosen.percentile_cutoff)
        if cand is not None and cand.flagged_row_keys == chosen_signature:
            equivalent_ks.append(k)
    equivalent_ps: List[float] = []
    for p in p_grid:
        cand = candidate_by_k_p(candidates, chosen.k_value, p)
        if cand is not None and cand.flagged_row_keys == chosen_signature:
            equivalent_ps.append(p)
    if len(equivalent_ks) > 1 and len(equivalent_ps) == 1:
        regime = 'percentile-dominated'
    elif len(equivalent_ps) > 1 and len(equivalent_ks) == 1:
        regime = 'trigger-dominated'
    elif len(equivalent_ps) > 1 and len(equivalent_ks) > 1:
        regime = 'degenerate-mixed'
    else:
        regime = 'mixed'
    return regime, equivalent_ks, equivalent_ps


def plot_gain_curve(
    candidates: List[CandidateResult],
    gain_attr: str,
    envelope: List[EnvelopePoint],
    chosen: Optional[CandidateResult],
    outpath: Path,
    title: str,
    ylabel: str,
    unit_label: str,
) -> None:
    usable = [c for c in candidates if getattr(c, gain_attr) is not None and not math.isnan(getattr(c, gain_attr))]
    if not usable:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([c.n_flagged_rows for c in usable], [getattr(c, gain_attr) for c in usable], alpha=0.35, color='grey', label='All candidate settings')
    if envelope:
        ax.plot([p.n_flagged_events for p in envelope], [p.gain_value for p in envelope], marker='o', linewidth=1.8, color='black', label='Best gain envelope')
    if chosen is not None:
        chosen_gain = getattr(chosen, gain_attr)
        if chosen_gain is not None and not math.isnan(chosen_gain):
            ax.scatter([chosen.n_flagged_rows], [chosen_gain], color=OPTIMAL_TRIGGER_COLOR, s=80, zorder=5, label='Chosen elbow')
            ax.annotate(f'K={chosen.k_value:g}, P={chosen.percentile_cutoff:.2f}', (chosen.n_flagged_rows, chosen_gain), xytext=(8, 8), textcoords='offset points', fontsize=8, color=OPTIMAL_TRIGGER_COLOR)
    ax.set_title(title)
    ax.set_xlabel(f'Number of deleted {unit_label}')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def compute_marginal_gain_loss(envelope: List[EnvelopePoint]) -> List[Tuple[float, float]]:
    if len(envelope) < 2:
        return []
    points: List[Tuple[float, float]] = []
    for left, right in zip(envelope[:-1], envelope[1:]):
        delta_d = float(right.n_flagged_events - left.n_flagged_events)
        delta_g = float(right.gain_value - left.gain_value)
        if delta_d <= EPS:
            continue
        midpoint_d = 0.5 * (left.n_flagged_events + right.n_flagged_events)
        gain_loss_per_extra_deletion = -delta_g / delta_d
        points.append((midpoint_d, gain_loss_per_extra_deletion))
    return points


def moving_average(values: List[float], window: int = 3) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def choose_percentile_fallback_candidate(
    candidates: List[CandidateResult],
    fallback_percentile: float,
    preferred_k: Optional[float] = 3.0,
) -> Optional[CandidateResult]:
    if preferred_k is not None:
        preferred = candidate_by_k_p(candidates, preferred_k, fallback_percentile)
        if preferred is not None and preferred.n_flagged_rows > 0:
            return preferred

    eligible = [
        cand for cand in candidates
        if math.isclose(cand.percentile_cutoff, fallback_percentile, rel_tol=1e-12, abs_tol=1e-12)
        and cand.n_flagged_rows > 0
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda c: (-c.k_value, c.n_flagged_rows, -c.total_excess_dropped))[0]


def choose_marginal_stability_candidate(
    envelope: List[EnvelopePoint],
    candidates: List[CandidateResult],
    p_grid: List[float],
    preferred_k: Optional[float] = 3.0,
) -> Tuple[Optional[CandidateResult], bool]:
    if not envelope:
        return None, False

    fallback_percentile = max(p_grid)
    fallback_choice = choose_percentile_fallback_candidate(candidates, fallback_percentile, preferred_k=preferred_k)

    if len(envelope) < 3:
        return (fallback_choice or envelope[0].candidate), True

    marginal_points = compute_marginal_gain_loss(envelope)
    if not marginal_points:
        return (fallback_choice or envelope[0].candidate), True

    abs_losses = [abs(y) for _, y in marginal_points]
    smooth = moving_average(abs_losses, window=3)
    max_loss = max(smooth) if smooth else 0.0

    no_signal_threshold = 0.10
    if max_loss < no_signal_threshold:
        return (fallback_choice or envelope[0].candidate), True

    threshold = max(0.03, 0.05 * max_loss)
    stability_span = 3

    for i in range(len(smooth)):
        if smooth[i] > threshold:
            continue
        end = min(len(smooth), i + stability_span + 1)
        stable = all(smooth[j] <= threshold * 1.5 for j in range(i, end))
        if stable:
            right_idx = min(i + 1, len(envelope) - 1)
            return envelope[right_idx].candidate, False

    return (fallback_choice or choose_elbow_candidate(envelope)), True


def plot_marginal_gain_curve(
    envelope: List[EnvelopePoint],
    chosen: Optional[CandidateResult],
    outpath: Path,
    title: str,
    ylabel: str,
    unit_label: str,
) -> None:
    marginal_points = compute_marginal_gain_loss(envelope)
    if not marginal_points:
        return
    xs = [x for x, _ in marginal_points]
    ys = [y for _, y in marginal_points]
    smooth = moving_average([abs(y) for y in ys], window=3)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker='o', linewidth=1.8, color='black', label='Marginal gain-loss curve')
    ax.plot(xs, smooth, linewidth=1.6, linestyle='--', color='grey', label='Smoothed |ΔG/ΔD|')

    if chosen is not None:
        chosen_x = float(chosen.n_flagged_rows)
        chosen_idx = min(range(len(xs)), key=lambda i: abs(xs[i] - chosen_x))
        ax.scatter([xs[chosen_idx]], [ys[chosen_idx]], color=OPTIMAL_TRIGGER_COLOR, s=80, zorder=5, label='Chosen point (marginal or fallback)')
        ax.annotate(
            f'K={chosen.k_value:g}, P={chosen.percentile_cutoff:.2f}',
            (xs[chosen_idx], ys[chosen_idx]),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=8,
            color=OPTIMAL_TRIGGER_COLOR,
        )

    ax.axhline(0.0, color='grey', linewidth=1.0, linestyle=':')

    ax.set_title(title)
    ax.set_xlabel(f'Number of deleted {unit_label}')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_violin_debug_generic(
    group_to_values: Dict[str, List[float]],
    global_values: List[float],
    chosen_candidate: CandidateResult,
    local_medians: Dict[str, float],
    global_median: float,
    outpath: Path,
    title: str,
    ylabel: str,
    regime_label: str,
    alternative_cutoffs: List[float],
) -> None:
    usable = []
    for group_name, vals in sorted(group_to_values.items()):
        clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
        if clean:
            usable.append((group_name, clean))
    if not usable:
        return
    labels = [g for g, _ in usable]
    data = [vals for _, vals in usable]
    global_clean = [float(v) for v in global_values if v is not None and not math.isnan(float(v))]

    alt_k_values = [k for k in alternative_cutoffs if not math.isclose(k, chosen_candidate.k_value, rel_tol=1e-12, abs_tol=1e-12)]
    alt_colors = alternative_k_colors(alt_k_values)

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(labels)), 6.5))
    vp = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for body in vp['bodies']:
        body.set_facecolor(VIOLIN_FACE_COLOR)
        body.set_edgecolor('black')
        body.set_alpha(0.8)
    if 'cmedians' in vp:
        vp['cmedians'].set_color('black')
        vp['cmedians'].set_linewidth(1.8)

    for p in PERCENTILES_FOR_VIOLIN_LINES:
        y = quantile(global_clean, p)
        if y is None or math.isnan(y):
            continue
        ax.axhline(y, color=PERCENTILE_REF_COLOR, linestyle=':', linewidth=1.0, zorder=0)
        ax.text(1.005, y, f'{int(round(p * 100))}', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8, color=PERCENTILE_REF_COLOR)

    ax.axhline(chosen_candidate.cutoff_value, color=OPTIMAL_TRIGGER_COLOR, linestyle=':', linewidth=2.0, zorder=1)
    ax.text(1.005, chosen_candidate.cutoff_value, f'chosen P={int(round(chosen_candidate.percentile_cutoff * 100))}', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8, color=OPTIMAL_TRIGGER_COLOR)

    for idx, group_name in enumerate(labels, start=1):
        local_med = local_medians[group_name]
        for k_value in sorted(alt_colors):
            trigger = (k_value - 1.0) * local_med + global_median
            ax.hlines(trigger, idx - 0.24, idx + 0.24, colors=alt_colors[k_value], linewidth=1.9, zorder=5)
        chosen_trigger = (chosen_candidate.k_value - 1.0) * local_med + global_median
        ax.hlines(chosen_trigger, idx - 0.12, idx + 0.12, colors=OPTIMAL_TRIGGER_COLOR, linewidth=2.4, zorder=6)

    legend_items = [
        Line2D([0], [0], color='black', lw=1.8, label='Median'),
        Line2D([0], [0], color=PERCENTILE_REF_COLOR, lw=1.0, linestyle=':', label='Global percentiles'),
        Line2D([0], [0], color=OPTIMAL_TRIGGER_COLOR, lw=2.0, linestyle=':', label='Chosen percentile cutoff'),
    ]
    if alt_k_values:
        legend_items.append(Line2D([0], [0], color=ALT_K_HIGH_COLOR, lw=2.0, label='Alternative K triggers'))
    legend_items.append(Line2D([0], [0], color=OPTIMAL_TRIGGER_COLOR, lw=2.4, label=f'Chosen hybrid trigger (K={chosen_candidate.k_value:g})'))

    ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'{title}\n{regime_label}')
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_candidate_table(path: Path, candidates: List[CandidateResult], raw_choice: Optional[CandidateResult], excess_choice: Optional[CandidateResult], suggested_choice: Optional[CandidateResult], unit_label: str) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'k_value', 'percentile_cutoff', 'cutoff_value', f'n_deleted_{unit_label}', f'n_unique_{unit_label}',
            'total_bl_dropped', 'total_excess_dropped', 'raw_gain', 'excess_gain',
            f'flagged_{unit_label}', 'is_raw_elbow', 'is_excess_elbow', 'is_suggested_excess_choice'
        ])
        for cand in sorted(candidates, key=lambda c: (-c.k_value, -c.percentile_cutoff, c.n_flagged_rows, -c.total_bl_dropped)):
            writer.writerow([
                format_scalar(cand.k_value),
                format_scalar(cand.percentile_cutoff),
                format_scalar(cand.cutoff_value),
                format_scalar(cand.n_flagged_rows),
                format_scalar(cand.n_flagged_units),
                format_scalar(cand.total_bl_dropped),
                format_scalar(cand.total_excess_dropped),
                format_scalar(cand.raw_gain),
                format_scalar(cand.excess_gain),
                ';'.join(cand.flagged_units) if cand.flagged_units else NA_STR,
                'yes' if raw_choice is not None and cand == raw_choice else 'no',
                'yes' if excess_choice is not None and cand == excess_choice else 'no',
                'yes' if suggested_choice is not None and cand == suggested_choice else 'no',
            ])


def _add_choice_block(lines: List[str], label: str, cand: Optional[CandidateResult], unit_label: str, regime: str, equivalent_ks: List[float], equivalent_ps: List[float]) -> None:
    lines.append(label)
    if cand is None:
        lines.append('  none found')
        lines.append('')
        return
    lines.extend([
        f'  K = {cand.k_value:g}',
        f'  percentile cutoff = {cand.percentile_cutoff:.2f}',
        f'  cutoff value = {cand.cutoff_value}',
        f'  deleted {unit_label} = {cand.n_flagged_rows}',
        f'  unique {unit_label} affected = {cand.n_flagged_units}',
        f'  total branch length dropped = {cand.total_bl_dropped}',
        f'  total excess dropped = {cand.total_excess_dropped}',
        f'  raw gain = {format_scalar(cand.raw_gain)}',
        f'  excess gain = {format_scalar(cand.excess_gain)}',
        f'  regime = {regime}',
    ])
    if equivalent_ks:
        lines.append(f'  equivalent K values at this P = {", ".join(f"{k:g}" for k in equivalent_ks)}')
    if equivalent_ps:
        lines.append(f'  equivalent percentile values at this K = {", ".join(f"{p:.2f}" for p in equivalent_ps)}')
    if regime == 'percentile-dominated':
        lines.append('  Percentile-dominated regime detected: the percentile cutoff determines the flagged set; the reported K is representative only.')
    elif regime == 'trigger-dominated':
        lines.append('  Trigger-dominated regime detected: the trigger determines the flagged set; the reported percentile is representative only.')
    elif regime == 'degenerate-mixed':
        lines.append('  Degenerate mixed regime detected: multiple K and multiple percentile values produce the same flagged set.')
    else:
        lines.append('  Mixed regime: both K and percentile contribute to the flagged set.')
    lines.extend([
        f'  {unit_label} = {", ".join(cand.flagged_units) if cand.flagged_units else NA_STR}',
        ''
    ])


def write_summary_text(
    path: Path,
    header: str,
    global_stats: SummaryStats,
    n_trees: int,
    n_units: int,
    n_rows: int,
    raw_choice: Optional[CandidateResult],
    excess_choice: Optional[CandidateResult],
    suggested_choice: Optional[CandidateResult],
    raw_regime: Tuple[str, List[float], List[float]],
    excess_regime: Tuple[str, List[float], List[float]],
    unit_label: str,
    value_label: str,
) -> None:
    lines = [
        header,
        '=' * len(header),
        '',
        'Suggested default for users: EXCESS gain solution. If there is no clear optimisation signal in the marginal curve, the selector uses a conservative fallback (P=0.99), corresponding to an outlier-filtering regime.',
        'RAW and EXCESS are both reported below for transparency.',
        '',
        f'Trees processed: {n_trees}',
        f'{unit_label.capitalize()} with distributions: {n_units}',
        f'Total observations: {n_rows}',
        '',
        f'Global {value_label} distribution reference:',
        f'  median = {format_scalar(global_stats.median)}',
        f'  mean = {format_scalar(global_stats.mean)}',
        f'  q75 = {format_scalar(global_stats.q75)}',
        f'  q80 = {format_scalar(global_stats.q80)}',
        f'  q85 = {format_scalar(global_stats.q85)}',
        f'  q90 = {format_scalar(global_stats.q90)}',
        f'  q93 = {format_scalar(global_stats.q93)}',
        f'  q95 = {format_scalar(global_stats.q95)}',
        f'  q96 = {format_scalar(global_stats.q96)}',
        f'  q97 = {format_scalar(global_stats.q97)}',
        f'  q98 = {format_scalar(global_stats.q98)}',
        f'  q99 = {format_scalar(global_stats.q99)}',
        '',
    ]
    _add_choice_block(lines, 'RAW-gain marginal-stability choice:', raw_choice, unit_label, *raw_regime)
    _add_choice_block(lines, 'EXCESS-gain marginal-stability choice (suggested):', excess_choice, unit_label, *excess_regime)
    _add_choice_block(lines, 'Suggested choice used for default histogram overlay:', suggested_choice, unit_label, *excess_regime)
    path.write_text('\n'.join(lines))


def collect_clade_data(tree_files: List[Path], stem_test_defs: Dict[str, Set[str]], progress: bool) -> Tuple[List[StemDistributionRow], Dict[str, List[float]], List[float]]:
    all_rows: List[StemDistributionRow] = []
    by_clade: Dict[str, List[float]] = defaultdict(list)
    global_values: List[float] = []
    for idx, tree_file in enumerate(tree_files, start=1):
        if progress:
            print(f'[collect-clades] {idx}/{len(tree_files)} trees ({tree_file.stem})', flush=True)
        tree = load_tree(tree_file)
        stem_raw = detect_mask_groups_from_graph(tree=tree, clade_defs=stem_test_defs, min_size=MIN_MASKING_CLADE_SIZE)
        _, current_stems, _ = collapse_identical_stem_fragments(raw_groups=stem_raw.mask_groups, raw_stems=stem_raw.stem_lengths, clade_defs=stem_test_defs)
        for group_name, stem_len in current_stems.items():
            row = StemDistributionRow(tree_id=tree_file.stem, stem_group=group_name, stem_branch=stem_len)
            all_rows.append(row)
            by_clade[group_name].append(stem_len)
            global_values.append(stem_len)
    return all_rows, by_clade, global_values


def collect_terminal_data(tree_files: List[Path], progress: bool) -> Tuple[List[TerminalDistributionRow], Dict[str, List[float]], List[float]]:
    all_rows: List[TerminalDistributionRow] = []
    by_taxon: Dict[str, List[float]] = defaultdict(list)
    global_values: List[float] = []
    for idx, tree_file in enumerate(tree_files, start=1):
        if progress:
            print(f'[collect-terminals] {idx}/{len(tree_files)} trees ({tree_file.stem})', flush=True)
        tree = load_tree(tree_file)
        rows = build_terminal_distribution_rows(tree_file.stem, tree)
        all_rows.extend(rows)
        for row in rows:
            by_taxon[row.taxon].append(row.branch_length)
            global_values.append(row.branch_length)
    return all_rows, by_taxon, global_values


def collect_internal_data(
    tree_files: List[Path],
    stem_test_defs: Dict[str, Set[str]],
    progress: bool,
) -> Tuple[List[InternalDistributionRow], Dict[str, List[float]], List[float]]:
    all_rows: List[InternalDistributionRow] = []
    by_internal_instance: Dict[str, List[float]] = defaultdict(list)
    global_values: List[float] = []

    for idx, tree_file in enumerate(tree_files, start=1):
        if progress:
            print(f'[collect-internal] {idx}/{len(tree_files)} trees ({tree_file.stem})', flush=True)
        tree = load_tree(tree_file)
        stem_groups: Dict[str, Set[str]] = {}
        if stem_test_defs:
            stem_raw = detect_mask_groups_from_graph(
                tree=tree,
                clade_defs=stem_test_defs,
                min_size=MIN_MASKING_CLADE_SIZE,
            )
            stem_groups, _, _ = collapse_identical_stem_fragments(
                raw_groups=stem_raw.mask_groups,
                raw_stems=stem_raw.stem_lengths,
                clade_defs=stem_test_defs,
            )

        for rec in build_internal_edge_records(tree, tree_file.stem, stem_groups):
            if rec.matched_known_stem_group is not None:
                continue
            unique_edge_name = f'{tree_file.stem}::{rec.edge_id}'
            row = InternalDistributionRow(tree_id=tree_file.stem, edge_id=unique_edge_name, branch_length=rec.edge_len)
            all_rows.append(row)
            by_internal_instance[unique_edge_name].append(rec.edge_len)
            global_values.append(rec.edge_len)

    return all_rows, by_internal_instance, global_values


def local_median_map(group_to_values: Dict[str, List[float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, values in group_to_values.items():
        med = median(values)
        if med is not None and not math.isnan(med):
            out[name] = med
    return out


def terminal_stem_trigger(k_value: float, local_median_value: Optional[float], global_median: float) -> Optional[float]:
    if local_median_value is None or math.isnan(local_median_value):
        return None
    return (k_value - 1.0) * local_median_value + global_median


def internal_trigger(k_value: float, local_median_value: Optional[float], global_median: float) -> Optional[float]:
    return k_value * global_median


def run_distribution_automation(
    rows: List[Tuple[str, str, float]],
    group_to_values: Dict[str, List[float]],
    global_values: List[float],
    output_prefix: str,
    output_dir: Path,
    n_trees: int,
    progress: bool,
    value_label: str,
    unit_label: str,
    violin_title: str,
    histogram_title: str,
    histogram_xlabel: str,
    gain_curve_title_prefix: str,
    distribution_header: str,
    k_grid: List[float],
    p_grid: List[float],
    trigger_func: Callable[[float, Optional[float], float], Optional[float]],
    preferred_fallback_k: Optional[float],
    alternative_cutoffs: List[float],
    use_violin: bool,
    use_internal_histogram: bool,
) -> AutomationArtifacts:
    if not rows:
        raise SystemExit(f'No {output_prefix} observations could be collected.')
    local_medians = local_median_map(group_to_values)
    global_median = median(global_values)
    if global_median is None or math.isnan(global_median):
        raise SystemExit(f'Global median for {output_prefix} could not be computed.')
    if progress:
        print(f'[{output_prefix}] finished collection: {len(rows)} observations across {len(group_to_values)} named groups for {unit_label}', flush=True)

    candidates = evaluate_candidates_generic(
        rows,
        local_medians,
        global_median,
        global_values,
        k_grid,
        p_grid,
        trigger_func=trigger_func,
        progress=progress,
        label=f'optimisation-{output_prefix}',
    )
    candidates = [c for c in candidates if c.n_flagged_rows > 0]
    if not candidates:
        raise SystemExit(f'No candidate settings flagged any {unit_label} for {output_prefix}.')

    raw_envelope = build_gain_envelope(candidates, 'raw_gain')
    excess_envelope = build_gain_envelope(candidates, 'excess_gain')
    raw_choice, raw_used_fallback = choose_marginal_stability_candidate(raw_envelope, candidates, p_grid, preferred_k=preferred_fallback_k)
    excess_choice, excess_used_fallback = choose_marginal_stability_candidate(excess_envelope, candidates, p_grid, preferred_k=preferred_fallback_k)
    suggested_choice = excess_choice or raw_choice
    raw_regime = classify_regime(candidates, raw_choice, k_grid, p_grid)
    excess_regime = classify_regime(candidates, excess_choice, k_grid, p_grid)

    candidate_table_path = output_dir / f'{output_prefix}_automation_candidate_table.csv'
    summary_text_path = output_dir / f'{output_prefix}_automation_summary.txt'
    histogram_path = output_dir / f'{output_prefix}_global_histogram_automated.png'
    raw_violin_path = output_dir / f'{output_prefix}_violins_raw.png'
    excess_violin_path = output_dir / f'{output_prefix}_violins_excess.png'
    raw_gain_path = output_dir / f'{output_prefix}_raw_gain_curve.png'
    excess_gain_path = output_dir / f'{output_prefix}_excess_gain_curve.png'
    raw_marginal_path = output_dir / f'{output_prefix}_raw_marginal_dg_dd_curve.png'
    excess_marginal_path = output_dir / f'{output_prefix}_excess_marginal_dg_dd_curve.png'

    write_candidate_table(candidate_table_path, candidates, raw_choice, excess_choice, suggested_choice, unit_label)
    write_summary_text(summary_text_path, distribution_header, calculate_summary_stats(global_values), n_trees, len(group_to_values), len(rows), raw_choice, excess_choice, suggested_choice, raw_regime, excess_regime, unit_label, value_label)

    if use_internal_histogram:
        plot_internal_histogram(
            global_values,
            histogram_path,
            histogram_title,
            histogram_xlabel,
            suggested_choice.cutoff_value if suggested_choice else None,
            global_median,
            suggested_choice,
            alternative_cutoffs=alternative_cutoffs,
        )
    else:
        plot_histogram(global_values, histogram_path, histogram_title, histogram_xlabel, suggested_choice.cutoff_value if suggested_choice else None)

    written_files = [candidate_table_path, summary_text_path, histogram_path]

    if use_violin:
        if raw_choice is not None:
            plot_violin_debug_generic(
                group_to_values,
                global_values,
                raw_choice,
                local_medians,
                global_median,
                raw_violin_path,
                f'{violin_title} (RAW optimisation)',
                histogram_xlabel,
                raw_regime[0],
                alternative_cutoffs=alternative_cutoffs,
            )
            written_files.append(raw_violin_path)
        if excess_choice is not None:
            plot_violin_debug_generic(
                group_to_values,
                global_values,
                excess_choice,
                local_medians,
                global_median,
                excess_violin_path,
                f'{violin_title} (EXCESS optimisation)',
                histogram_xlabel,
                excess_regime[0],
                alternative_cutoffs=alternative_cutoffs,
            )
            written_files.append(excess_violin_path)

    plot_gain_curve(candidates, 'raw_gain', raw_envelope, raw_choice, raw_gain_path, f'Raw gain curve for {gain_curve_title_prefix}', f'Raw gain = total dropped branch length / deleted {unit_label}', unit_label)
    plot_gain_curve(candidates, 'excess_gain', excess_envelope, excess_choice, excess_gain_path, f'Excess gain curve for {gain_curve_title_prefix}', f'Excess gain = total excess above trigger / deleted {unit_label}', unit_label)
    plot_marginal_gain_curve(raw_envelope, raw_choice, raw_marginal_path, f'RAW marginal ΔG/ΔD curve for {gain_curve_title_prefix}', 'Gain loss per additional deleted branch (-ΔG/ΔD)', unit_label)
    plot_marginal_gain_curve(excess_envelope, excess_choice, excess_marginal_path, f'EXCESS marginal ΔG/ΔD curve for {gain_curve_title_prefix}', 'Gain loss per additional deleted branch (-ΔG/ΔD)', unit_label)
    written_files.extend([raw_gain_path, excess_gain_path, raw_marginal_path, excess_marginal_path])

    lines = [
        f'{output_prefix} suggestion: use EXCESS gain',
        f'{output_prefix} raw-gain marginal choice: K={raw_choice.k_value:g}, P={raw_choice.percentile_cutoff:.2f}, deleted {unit_label}={raw_choice.n_flagged_rows}, regime={raw_regime[0]}' if raw_choice else f'{output_prefix} raw-gain marginal choice: none',
        f'{output_prefix} excess-gain marginal choice: K={excess_choice.k_value:g}, P={excess_choice.percentile_cutoff:.2f}, deleted {unit_label}={excess_choice.n_flagged_rows}, regime={excess_regime[0]}' if excess_choice else f'{output_prefix} excess-gain marginal choice: none',
    ]
    if raw_used_fallback:
        lines.append(f'{output_prefix} raw-gain: no clear optimisation signal in marginal curve -> using fallback P={max(p_grid):.2f}')
    if excess_used_fallback:
        lines.append(f'{output_prefix} excess-gain: no clear optimisation signal in marginal curve -> using fallback P={max(p_grid):.2f}')
    if excess_regime[0] == 'percentile-dominated' and excess_regime[1]:
        lines.append(f'{output_prefix} Percentile-dominated regime detected. Equivalent K values at chosen P: {", ".join(f"{k:g}" for k in excess_regime[1])}')
    if excess_regime[0] == 'trigger-dominated' and excess_regime[2]:
        lines.append(f'{output_prefix} Trigger-dominated regime detected. Equivalent percentile values at chosen K: {", ".join(f"{p:.2f}" for p in excess_regime[2])}')

    return AutomationArtifacts(
        output_prefix=output_prefix,
        files_written=written_files,
        console_lines=lines,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Automated exploratory analysis for long-branch detection with RAW and EXCESS reporting for terminal, stem, and non-stem internal branches.')
    parser.add_argument('--trees', type=Path, required=True, help='Directory of Newick tree files.')
    parser.add_argument('--stem_to_test', type=Path, required=True, help='Named clade file used to define retained stems.')
    parser.add_argument('--k_values', default=','.join(str(v) for v in DEFAULT_K_GRID), help='Comma-separated K grid, strict to liberal.')
    parser.add_argument('--percentiles', default=','.join(str(v) for v in DEFAULT_P_GRID), help='Comma-separated global percentile grid, strict to liberal.')
    parser.add_argument('--alternative_cutoffs', type=parse_comma_floats, default=[], help='Comma-separated K values to visualise. For terminal/stem these are shown on violin plots; for internal they are shown as dots along the top axis of the histogram.')
    parser.add_argument('--progress', action='store_true', help='Print progress messages.')
    args = parser.parse_args()

    stem_test_defs = parse_named_clades(args.stem_to_test)
    tree_files = iter_tree_files(args.trees)
    if not tree_files:
        raise SystemExit(f'No tree files found in {args.trees}')
    k_grid = [float(x.strip()) for x in args.k_values.split(',') if x.strip()]
    p_grid = [float(x.strip()) for x in args.percentiles.split(',') if x.strip()]
    if not k_grid or not p_grid:
        raise SystemExit('Need at least one K and one percentile value.')
    if any(p <= 0 or p >= 1 for p in p_grid):
        raise SystemExit('Percentiles must be strictly between 0 and 1.')

    output_dir = args.trees

    all_stem_rows, stem_by_clade, global_stem_values = collect_clade_data(tree_files, stem_test_defs, args.progress)
    if not all_stem_rows:
        raise SystemExit('No stem distributions could be collected.')
    stem_distribution_path = output_dir / 'stem_branch_distributions.csv'
    write_stem_distribution_csv(stem_distribution_path, all_stem_rows)

    all_terminal_rows, terminal_by_taxon, global_terminal_values = collect_terminal_data(tree_files, args.progress)
    if not all_terminal_rows:
        raise SystemExit('No terminal distributions could be collected.')
    terminal_distribution_path = output_dir / 'terminal_branch_distributions.csv'
    write_terminal_distribution_csv(terminal_distribution_path, all_terminal_rows)

    all_internal_rows, internal_by_instance, global_internal_values = collect_internal_data(tree_files, stem_test_defs, args.progress)
    if not all_internal_rows:
        raise SystemExit('No non-stem internal branch distributions could be collected.')
    internal_distribution_path = output_dir / 'non_stem_internal_branch_distributions.csv'
    write_internal_distribution_csv(internal_distribution_path, all_internal_rows)

    stem_artifacts = run_distribution_automation(
        rows=[(row.tree_id, row.stem_group, row.stem_branch) for row in all_stem_rows],
        group_to_values=stem_by_clade,
        global_values=global_stem_values,
        output_prefix='stem_branch',
        output_dir=output_dir,
        n_trees=len(tree_files),
        progress=args.progress,
        value_label='stem branch',
        unit_label='stem branches',
        violin_title='Stem branch distributions by clade',
        histogram_title='Global stem branch distribution',
        histogram_xlabel='Stem branch length',
        gain_curve_title_prefix='clade stem automation',
        distribution_header='Long-branch clade automation summary',
        k_grid=k_grid,
        p_grid=p_grid,
        trigger_func=terminal_stem_trigger,
        preferred_fallback_k=3.0,
        alternative_cutoffs=args.alternative_cutoffs,
        use_violin=True,
        use_internal_histogram=False,
    )

    terminal_artifacts = run_distribution_automation(
        rows=[(row.tree_id, row.taxon, row.branch_length) for row in all_terminal_rows],
        group_to_values=terminal_by_taxon,
        global_values=global_terminal_values,
        output_prefix='terminal_branch',
        output_dir=output_dir,
        n_trees=len(tree_files),
        progress=args.progress,
        value_label='terminal branch',
        unit_label='terminal branches',
        violin_title='Terminal branch distributions by taxon',
        histogram_title='Global terminal branch distribution',
        histogram_xlabel='Terminal branch length',
        gain_curve_title_prefix='terminal branch automation',
        distribution_header='Long-branch terminal automation summary',
        k_grid=k_grid,
        p_grid=p_grid,
        trigger_func=terminal_stem_trigger,
        preferred_fallback_k=3.0,
        alternative_cutoffs=args.alternative_cutoffs,
        use_violin=True,
        use_internal_histogram=False,
    )

    internal_artifacts = run_distribution_automation(
        rows=[(row.tree_id, row.edge_id, row.branch_length) for row in all_internal_rows],
        group_to_values=internal_by_instance,
        global_values=global_internal_values,
        output_prefix='non_stem_internal_branch',
        output_dir=output_dir,
        n_trees=len(tree_files),
        progress=args.progress,
        value_label='non-stem internal branch',
        unit_label='non-stem internal branches',
        violin_title='',
        histogram_title='Global non-stem internal branch distribution',
        histogram_xlabel='Non-stem internal branch length',
        gain_curve_title_prefix='non-stem internal branch automation',
        distribution_header='Long-branch non-stem internal automation summary',
        k_grid=k_grid,
        p_grid=p_grid,
        trigger_func=internal_trigger,
        preferred_fallback_k=3.0,
        alternative_cutoffs=args.alternative_cutoffs,
        use_violin=False,
        use_internal_histogram=True,
    )

    print('\n=== FINAL OUTPUTS ===', flush=True)
    for path in [stem_distribution_path, terminal_distribution_path, internal_distribution_path]:
        print(f'Wrote: {path}', flush=True)
    for artifact in [stem_artifacts, terminal_artifacts, internal_artifacts]:
        for path in artifact.files_written:
            print(f'Wrote: {path}', flush=True)
        for line in artifact.console_lines:
            print(line, flush=True)
    print(f'Processed {len(tree_files)} tree(s)', flush=True)


if __name__ == '__main__':
    main()
