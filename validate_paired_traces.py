#!/usr/bin/env python3
"""Validate strict counterfactual pairing before the first disturbed action."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    groups = {}; ends = {}; noise = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line); event = row.get("event")
        key = (int(row.get("task_id", -1)), int(row.get("episode_idx", -1)))
        if event == "step": groups.setdefault(key, []).append(row)
        elif event == "episode_end": ends[key] = row
        elif event == "inference": noise.setdefault(key, []).append(row.get("sampling_noise_sha256"))
    for rows in groups.values(): rows.sort(key=lambda r: r["action_index"])
    return groups, ends, noise


def main():
    p = argparse.ArgumentParser(); p.add_argument("--normal", type=Path, required=True)
    p.add_argument("--disturbed", type=Path, required=True); p.add_argument("--out", type=Path)
    args = p.parse_args(); normal, nend, nnoise = load(args.normal); disturbed, dend, dnoise = load(args.disturbed)
    violations, episodes = [], []
    if set(normal) != set(disturbed): violations.append({"type":"episode_key_mismatch"})
    for key in sorted(set(normal) & set(disturbed)):
        nr, dr = normal[key], disturbed[key]
        active = [int(r["action_index"]) for r in dr if r.get("disturbance_active", False)]
        start = active[0] if active else None
        if start is None: violations.append({"type":"no_active_steps","episode":key}); continue
        nmap, dmap = {int(r["action_index"]):r for r in nr}, {int(r["action_index"]):r for r in dr}
        pre = list(range(start)); missing = [i for i in pre if i not in nmap or i not in dmap]
        max_action = max((float(np.max(np.abs(np.asarray(nmap[i]["intended_action"])-np.asarray(dmap[i]["intended_action"])))) for i in pre if i not in missing), default=0.)
        max_eef = max((float(np.max(np.abs(np.asarray(nmap[i]["eef_pos_after"])-np.asarray(dmap[i]["eef_pos_after"])))) for i in pre if i not in missing), default=0.)
        contiguous = active == list(range(start, start+len(active)))
        same_noise_prefix = nnoise.get(key, [])[:(start+4)//5] == dnoise.get(key, [])[:(start+4)//5]
        item={"task_id":key[0],"episode_idx":key[1],"start":start,"active_steps":len(active),
              "active_contiguous":contiguous,"pre_max_intended_action_difference":max_action,
              "pre_max_eef_after_difference_m":max_eef,"noise_hash_prefix_equal":same_noise_prefix,
              "normal_success":nend.get(key,{}).get("success"),"disturbed_success":dend.get(key,{}).get("success")}
        episodes.append(item)
        if missing or max_action != 0 or max_eef != 0 or not contiguous or len(active) != 10 or not same_noise_prefix:
            violations.append({"type":"pair_gate_failed","episode":item,"missing_pre_indices":missing})
    report={"passed":not violations,"episodes":episodes,"violations":violations,
            "max_pre_action_difference":max((e["pre_max_intended_action_difference"] for e in episodes),default=None),
            "max_pre_eef_difference_m":max((e["pre_max_eef_after_difference_m"] for e in episodes),default=None)}
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if args.out: args.out.write_text(text,encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__": main()
