import json

from vla_supervisor.paired_evaluation import evaluate_manifest, evaluate_pair


def write_trace(path, *, success, actions, intervention_at=None, replans=0):
    rows = []
    for index, action in enumerate(actions):
        rows.append({
            "event": "step", "action_index": index, "intended_action": action,
            "eef_pos_before": [float(index), 0.0, 0.0],
            "supervisor_action_source": "recovery_bridge" if index == intervention_at else "policy_chunk",
            "supervisor_control_epoch_before_action": int(
                intervention_at is not None and index >= intervention_at
            ),
            "supervisor_intervention": ({"mode": "hold_then_replan"}
                                        if index == intervention_at else {"mode": "nominal"}),
        })
    rows.append({
        "event": "episode_end", "success": success,
        "executed_actions": len(actions), "inference_calls": 2,
        "supervisor_replans": replans,
        "supervisor_intervention_counts": {"hold_then_replan": replans},
    })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pair_labels_rescue_and_checks_prefix_identity(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    control = tmp_path / "control.jsonl"
    common = [[0.1] * 7, [0.2] * 7]
    write_trace(baseline, success=False, actions=common + [[0.3] * 7])
    write_trace(control, success=True, actions=common + [[0.0] * 7], intervention_at=2, replans=1)
    result = evaluate_pair("pair", baseline, control)
    assert result.outcome == "rescued"
    assert result.first_intervention_index == 2
    assert result.pre_intervention_actions_equal
    assert result.pre_intervention_states_equal


def test_manifest_reports_harm_and_invalid_causal_prefix(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    control = tmp_path / "control.jsonl"
    write_trace(baseline, success=True, actions=[[0.1] * 7, [0.2] * 7])
    write_trace(control, success=False, actions=[[0.9] * 7, [0.0] * 7], intervention_at=1, replans=1)
    summary = evaluate_manifest({"pairs": [{
        "pair_id": "p", "baseline_trace": baseline.name, "control_trace": control.name,
    }]}, tmp_path)
    assert summary["outcomes"] == {"harmed": 1}
    assert summary["harm_rate_among_baseline_successes"] == 1.0
    assert not summary["valid_for_causal_claims"]


def test_one_based_trace_excludes_first_intervention_from_identity_prefix(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    control = tmp_path / "control.jsonl"
    baseline_rows = [
        {"event": "step", "action_index": 1, "intended_action": [0.1] * 7,
         "eef_pos_before": [0, 0, 0], "supervisor_control_epoch_before_action": 0},
        {"event": "step", "action_index": 2, "intended_action": [0.2] * 7,
         "eef_pos_before": [1, 0, 0], "supervisor_control_epoch_before_action": 0},
        {"event": "episode_end", "success": False, "executed_actions": 2},
    ]
    control_rows = [
        {"event": "step", "action_index": 1, "intended_action": [0.1] * 7,
         "eef_pos_before": [0, 0, 0], "supervisor_control_epoch_before_action": 0},
        {"event": "step", "action_index": 2, "intended_action": [9.9] * 7,
         "eef_pos_before": [9, 0, 0], "supervisor_control_epoch_before_action": 1},
        {"event": "episode_end", "success": True, "executed_actions": 2},
    ]
    for path, rows in ((baseline, baseline_rows), (control, control_rows)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = evaluate_pair("one_based", baseline, control)
    assert result.first_intervention_index == 2
    assert result.pre_intervention_actions_equal
    assert result.pre_intervention_states_equal
