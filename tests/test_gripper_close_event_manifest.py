from cross_suite_generalization.build_gripper_close_event_manifest import build_events


def test_all_close_events_are_preserved_and_linked():
    row = {"anonymous_id": "episode-a", "scored_steps": 100}
    events = build_events(row, [4, 20, 98], pre=15, post=40)
    assert [e["close_frame"] for e in events] == [4, 20, 98]
    assert events[0]["window_start"] == 0
    assert events[-1]["window_end"] == 99
    assert events[0]["has_later_close"] is True
    assert events[-1]["has_later_close"] is False
    assert events[1]["supersedes_event_id"] == events[0]["event_id"]
