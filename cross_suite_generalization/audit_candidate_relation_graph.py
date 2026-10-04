#!/usr/bin/env python3
"""Audit whether identity errors are duplicate/part masks or other objects."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.candidate_relation_graph import build_candidate_relation_graph


def relation_to_target(graph, choice, acceptable):
    if choice is None:
        return "unknown"
    if choice in acceptable:
        return "exact_acceptable"
    for target in acceptable:
        if target in graph.equivalents(choice):
            return "equivalent_mask"
        if target in graph.whole_candidates_for(choice):
            return "choice_part_of_target"
        if choice in graph.whole_candidates_for(target):
            return "choice_contains_target"
        if target in graph.adjacent_candidates(choice):
            return "adjacent_to_target"
    return "different_object_or_disjoint"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = {r["anonymous_id"]: r for r in json.loads(args.annotations.read_text(encoding="utf-8"))["records"]}
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))["records"]
    counts = Counter()
    records = []
    for row in comparison:
        identifier = row["anonymous_id"]
        archive = np.load(args.mask_dir / f"{identifier}_initial_masks.npz")
        masks = {index + 1: mask for index, mask in enumerate(archive["masks"])}
        graph = build_candidate_relation_graph(masks)
        acceptable = set(labels[identifier]["acceptable_target_mask_ids"])
        v1_relation = relation_to_target(graph, row.get("v1_choice"), acceptable)
        v2_relation = relation_to_target(graph, row.get("v2_choice"), acceptable)
        counts[f"v1:{v1_relation}"] += 1
        counts[f"v2:{v2_relation}"] += 1
        edge_counts = Counter(edge.relation for edge in graph.relations)
        records.append({
            "anonymous_id": identifier,
            "acceptable_ids": sorted(acceptable),
            "v1_choice": row.get("v1_choice"),
            "v1_relation_to_target": v1_relation,
            "v2_choice": row.get("v2_choice"),
            "v2_relation_to_target": v2_relation,
            "graph_edge_counts": dict(edge_counts),
        })
    result = {"schema_version": 1, "summary": dict(sorted(counts.items())), "records": records}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(records), "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
