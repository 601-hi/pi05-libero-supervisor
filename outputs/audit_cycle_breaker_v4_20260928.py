import collections
import json
import math
import pathlib

import numpy as np


TRACE = pathlib.Path(__file__).parent / "goal_relation_direct_rollback_gpu_v4_mask_paired_20260927" / "control.jsonl"
OUT = pathlib.Path(__file__).parent / "cycle_breaker_v4_counterfactual_20260928.json"


def direction_key(candidate_id):
    parts = str(candidate_id).split("_")
    return "_".join(parts[:4]) if len(parts) == 5 and parts[:2] == ["escape", "axis"] else None


def nominal_select(rows):
    safe = [row for row in rows if row.get("clearance_safe")]
    direct = [row for row in safe if row.get("candidate_kind") == "direct"]
    if direct:
        return max(direct, key=lambda row: float(row["candidate_scale"]))
    if not safe:
        return None
    minimum_scale = min(float(row["candidate_scale"]) for row in safe)
    pool = [row for row in safe if float(row["candidate_scale"]) == minimum_scale]
    gains = [float(row["worst_clearance_gain_m"]) for row in pool]
    progress = [float(row["projected_progress_m"]) for row in pool]
    gain_low, gain_high = min(gains), max(gains)
    progress_low, progress_high = min(progress), max(progress)
    baseline = min(min(float(row["baseline_self_clearance_m"]),
                       float(row["baseline_environment_clearance_m"])) for row in pool)
    progress_weight = 1.0 / (1.0 + math.exp(-max(-60., min(60., (baseline - .0045) / .0005))))
    def score(row):
        safety = .5 if gain_high - gain_low <= 1e-12 else (
            float(row["worst_clearance_gain_m"]) - gain_low) / (gain_high - gain_low)
        goal = .5 if progress_high - progress_low <= 1e-12 else (
            float(row["projected_progress_m"]) - progress_low) / (progress_high - progress_low)
        return ((1. - progress_weight) * safety + progress_weight * goal, goal, safety)
    return max(pool, key=score)


rows = [json.loads(line) for line in TRACE.open(encoding="utf-8") if line.strip()]
steps = {
    (row["episode_idx"], row["action_index"]): np.asarray(row["eef_pos_after"], float)
    for row in rows if row.get("event") == "step"
}
candidates = collections.defaultdict(list)
proposals = {}
responses = []
for row in rows:
    key = (row.get("episode_idx"), row.get("action_index"))
    if row.get("event") == "complex_rollback_candidate":
        candidates[key].append(row)
    elif row.get("event") == "complex_rollback_proposal":
        proposals[key] = row
    elif row.get("event") == "complex_rollback_response":
        responses.append(row)

states = collections.defaultdict(lambda: {
    "samples": collections.deque(maxlen=6), "cooldowns": {}})
trigger_rows = []
changed_next_choices = []
pulse_opportunities = []

for response in responses:
    episode = response["episode_idx"]
    action_index = response["action_index"]
    key = (episode, action_index)
    state = states[episode]
    state["cooldowns"] = {
        candidate: remaining - 1 for candidate, remaining in state["cooldowns"].items()
        if remaining > 1
    }
    proposal = proposals.get(key)
    candidate_id = response.get("candidate_id")
    target_index = None if proposal is None else proposal.get("target_action_index")
    position = steps.get(key)
    target = steps.get((episode, target_index))
    clearance = min(response["actual_self_clearance_m"], response["actual_environment_clearance_m"])
    if direction_key(candidate_id) is not None and position is not None and target is not None:
        join_error = float(np.linalg.norm(position - target))
        sample = (candidate_id, position, join_error, clearance)
        state["samples"].append(sample)
        if len(state["samples"]) == 6:
            first, last = state["samples"][0], state["samples"][-1]
            join_progress = first[2] - last[2]
            clearance_progress = last[3] - first[3]
            repeated = sum(item[0] == last[0] for item in state["samples"])
            returned = any(np.linalg.norm(last[1] - item[1]) <= .00035
                           for item in list(state["samples"])[:-2])
            no_progress = join_progress < .00020 and clearance_progress < .00005
            if no_progress and (repeated >= 3 or returned) and candidate_id not in state["cooldowns"]:
                state["cooldowns"][candidate_id] = 8
                trigger_rows.append({
                    "episode": episode, "action_index": action_index,
                    "candidate_id": candidate_id, "join_progress_m": join_progress,
                    "clearance_progress_m": clearance_progress,
                    "reason": "return_cycle" if returned else "repeated_no_progress",
                })
                state["samples"].clear()
    else:
        state["samples"].clear()

    next_key = (episode, action_index + 1)
    next_proposal = proposals.get(next_key)
    next_candidates = candidates.get(next_key, [])
    if next_proposal and next_candidates:
        original = next_proposal.get("selected_candidate_id")
        alternatives = [row for row in next_candidates
                        if row.get("clearance_safe") and row.get("candidate_id") not in state["cooldowns"]]
        if not alternatives:
            alternatives = [row for row in next_candidates if row.get("clearance_safe")]
        alternate_ids = [row["candidate_id"] for row in alternatives]
        if original in state["cooldowns"] and any(item != original for item in alternate_ids):
            changed_next_choices.append({
                "episode": episode, "action_index": action_index + 1,
                "original": original, "available_after_tabu": alternate_ids,
            })

for key, proposal in proposals.items():
    if not proposal.get("escape_progress_mask_active"):
        continue
    selected = nominal_select(candidates[key])
    if selected is None:
        continue
    selected_id = selected.get("candidate_id")
    direction = direction_key(selected_id)
    if direction is None:
        continue
    eligible = []
    for row in candidates[key]:
        if (direction_key(row.get("candidate_id")) == direction
                and row.get("clearance_safe")
                and float(row["candidate_scale"]) > float(selected["candidate_scale"])
                and float(row["worst_clearance_gain_m"]) >= 0.0
                and float(row["projected_progress_m"]) - float(selected["projected_progress_m"]) >= .00010):
            eligible.append(row)
    if eligible:
        chosen = min(eligible, key=lambda row: float(row["candidate_scale"]))
        pulse_opportunities.append({
            "episode": key[0], "action_index": key[1],
            "base": selected_id, "pulse": chosen["candidate_id"],
        })

result = {
    "source": str(TRACE),
    "episodes_with_responses": sorted({row["episode_idx"] for row in responses}),
    "response_count": len(responses),
    "tabu_trigger_count": len(trigger_rows),
    "tabu_trigger_episodes": sorted({row["episode"] for row in trigger_rows}),
    "next_choice_change_opportunities": len(changed_next_choices),
    "pulse_opportunity_count": len(pulse_opportunities),
    "pulse_opportunity_episodes": sorted({row["episode"] for row in pulse_opportunities}),
    "tabu_triggers": trigger_rows,
    "next_choice_changes": changed_next_choices,
    "pulse_opportunities": pulse_opportunities,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items()
                  if key not in {"tabu_triggers", "next_choice_changes", "pulse_opportunities"}},
                 ensure_ascii=False, indent=2))
