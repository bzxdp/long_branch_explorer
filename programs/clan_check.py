#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree

TREE_EXTS = {".nwk", ".tree", ".tre", ".newick", ".txt", ".treefile"}


def load_tree(path: Path) -> Tree:
    return Phylo.read(str(path), "newick")


def iter_tree_files(trees_dir: Path) -> List[Path]:
    return sorted(
        p for p in trees_dir.iterdir()
        if p.is_file() and p.suffix.lower() in TREE_EXTS
    )


def parse_named_clades(path: Path) -> Dict[str, Set[str]]:
    clades: Dict[str, Set[str]] = {}
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Invalid line {line_number}: expected 'CLADE: tax1 tax2'"
                )
            name, taxa = line.split(":", 1)
            taxa_set = {x.strip() for x in taxa.split() if x.strip()}
            if not taxa_set:
                raise ValueError(f"No taxa for clade '{name.strip()}'")
            clades[name.strip()] = taxa_set
    return clades


def get_terminals(tree: Tree) -> Set[str]:
    return {t.name for t in tree.get_terminals() if t.name}


def prune_tree_copy(tree: Tree, taxa_to_drop: Set[str]) -> Tree:
    pruned = deepcopy(tree)
    for taxon in sorted(taxa_to_drop):
        target = next((t for t in pruned.get_terminals() if t.name == taxon), None)
        if target:
            try:
                pruned.prune(target)
            except Exception:
                pass
    return pruned


def largest_fragment_clade(tree: Tree, taxa: Set[str]) -> Set[str]:
    present = taxa & get_terminals(tree)
    best: Set[str] = set()
    for clade in tree.find_clades(order="preorder"):
        tips = {t.name for t in clade.get_terminals() if t.name}
        overlap = tips & present
        if overlap == tips and len(overlap) > len(best):
            best = overlap
    return best


def clan_pass(tree: Tree, taxa: Set[str]) -> bool:
    present = taxa & get_terminals(tree)
    if len(present) <= 1:
        return True
    fragment = largest_fragment_clade(tree, present)
    return fragment == present


def load_dropped_taxa(tree_path: Path, ext: str | None) -> Set[str]:
    if not ext:
        return set()
    drop_file = tree_path.with_suffix(ext)
    if not drop_file.exists():
        return set()
    with drop_file.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="ClanCheck validation (no prediction)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tree", type=Path)
    group.add_argument("--trees", type=Path)
    parser.add_argument("--clan_check", type=Path, required=True)
    parser.add_argument("--lbr_results_ext", type=str)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    clan_defs = parse_named_clades(args.clan_check)

    if args.tree:
        tree_files = [args.tree]
        output_dir = args.tree.parent
    else:
        tree_files = iter_tree_files(args.trees)
        if not tree_files:
            raise SystemExit("No tree files found")
        output_dir = args.trees

    summary_rows: List[List[str]] = []
    initial_pass: Set[str] = set()
    rescued: Set[str] = set()
    failed: Set[str] = set()

    for i, tree_path in enumerate(tree_files, start=1):
        if args.progress:
            print(f"{i}/{len(tree_files)}: {tree_path.name}", flush=True)

        tree = load_tree(tree_path)
        dropped = load_dropped_taxa(tree_path, args.lbr_results_ext)
        pruned_tree = prune_tree_copy(tree, dropped) if dropped else deepcopy(tree)

        all_initial = True
        all_post = True

        for clan_name in sorted(clan_defs):
            taxa = clan_defs[clan_name]

            initial_ok = clan_pass(tree, taxa)
            post_ok = clan_pass(pruned_tree, taxa)

            if not initial_ok:
                all_initial = False
            if not post_ok:
                all_post = False

            summary_rows.append([
                tree_path.stem,
                clan_name,
                "TRUE" if initial_ok else "FALSE",
                "TRUE" if post_ok else "FALSE",
                "TRUE" if bool(dropped) else "FALSE",
                ";".join(sorted(dropped)) if dropped else "NA",
            ])

        if all_initial:
            initial_pass.add(tree_path.stem)
        elif all_post:
            rescued.add(tree_path.stem)
        else:
            failed.add(tree_path.stem)

    # write outputs
    (output_dir / "clancheck_CC_initial_pass.txt").write_text("\n".join(sorted(initial_pass)))
    (output_dir / "clancheck_CC_rescued_after_pruning.txt").write_text("\n".join(sorted(rescued)))
    (output_dir / "clancheck_CC_failed_after_pruning.txt").write_text("\n".join(sorted(failed)))

    summary_csv = output_dir / "clancheck_CC_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tree_id",
            "tested_clade",
            "initial_pass",
            "post_prune_pass",
            "used_long_branch_output",
            "long_branch_deleted_taxa",
        ])
        writer.writerows(summary_rows)

    print(f"\nProcessed {len(tree_files)} tree(s)")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
