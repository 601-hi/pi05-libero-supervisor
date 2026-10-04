from cross_suite_generalization.evaluate_cross_view_confirmation_gate import chosen_similarity, summarise


def test_chosen_similarity_uses_chosen_candidate() -> None:
    bridge = {
        "chosen_wrist_id": 2,
        "candidate_scores": [
            {"candidate_id": 1, "appearance": 0.99},
            {"candidate_id": 2, "appearance": 0.83},
        ],
    }
    assert chosen_similarity(bridge) == 0.83


def test_summary_keeps_evaluable_and_all_episode_denominators_separate() -> None:
    rows = [
        {"confirmed": True, "correct": True},
        {"confirmed": True, "correct": False},
        {"confirmed": False, "correct": False},
    ]
    result = summarise(rows, total_episodes=5)
    assert result["evaluable"] == 3
    assert result["confirmed"] == 2
    assert result["correct_confirmed"] == 1
    assert result["confirmation_precision"] == 0.5
    assert result["coverage_all_episodes"] == 0.4
    assert result["coverage_evaluable"] == 2 / 3
    assert result["unknown_rate_all_episodes"] == 0.6


def test_summary_handles_no_confirmations() -> None:
    result = summarise([{"confirmed": False, "correct": False}], total_episodes=1)
    assert result["confirmation_precision"] is None
    assert result["coverage_all_episodes"] == 0.0
