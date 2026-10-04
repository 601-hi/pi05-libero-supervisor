import json

import numpy as np

from evaluate_natural_failure_scores import Rule, conformal_upper, load_outcomes, window_statistic


def test_vote_and_mean_statistics():
    scores = np.asarray([1.0, 4.0, 2.0, 5.0])
    np.testing.assert_allclose(window_statistic(scores, Rule("vote", 2, 3)), [2.0, 4.0])
    np.testing.assert_allclose(window_statistic(scores, Rule("mean", 0, 3)), [7 / 3, 11 / 3])


def test_conformal_upper_is_finite_sample_rank():
    threshold, rank = conformal_upper(np.arange(10), alpha=0.2)
    assert rank == 9
    assert threshold == 8


def test_load_outcomes_preserves_trace_order(tmp_path):
    paths = []
    for task in (29, 9):
        path = tmp_path / f"task{task}.jsonl"
        path.write_text(json.dumps({
            "event": "episode_end", "task_id": task, "episode_idx": 0,
            "success": task == 9,
        }) + "\n", encoding="utf-8")
        paths.append(path)
    outcomes = load_outcomes(paths)
    assert outcomes[(0, 0)]["task_id"] == 29
    assert not outcomes[(0, 0)]["success"]
    assert outcomes[(1, 0)]["success"]
