import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "cross_suite_generalization" / "build_multi_event_visual_manifest.py"
SPEC = importlib.util.spec_from_file_location("multi_event_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(start, response, target=0.03):
    return {
        "start_action_index": start,
        "end_action_index": start + 4,
        "phase": start / 100,
        "target_path_m": target,
        "actual_path_m": target * response,
        "response_ratio": response,
    }


def test_separated_extremes_do_not_double_count_adjacent_windows():
    windows = [row(0, .1), row(1, .05), row(30, .2), row(60, .3)]
    selected = MODULE.separated_extremes(windows, count=3, minimum_gap=20, reverse=False)
    assert [item["start_action_index"] for item in selected] == [1, 30, 60]


def test_controls_are_separated_and_matched_without_labels():
    windows = [row(0, .1), row(30, .2), row(60, .8), row(90, .9)]
    events = [windows[0]]
    controls = MODULE.matched_controls(windows, events, minimum_gap=20)
    assert len(controls) == 1
    assert controls[0]["start_action_index"] in {60, 90}
    assert controls[0]["matched_event_start_action_index"] == 0
