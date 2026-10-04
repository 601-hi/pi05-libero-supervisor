import importlib.util
from pathlib import Path
import sys
import types


# The job builder itself does not need GPU packages; stub imports for a CPU unit test.
torch = types.ModuleType("torch")
sam2 = types.ModuleType("sam2")
automatic = types.ModuleType("sam2.automatic_mask_generator")
automatic.SAM2AutomaticMaskGenerator = object
sys.modules.setdefault("torch", torch)
sys.modules.setdefault("sam2", sam2)
sys.modules.setdefault("sam2.automatic_mask_generator", automatic)

MODULE_PATH = Path(__file__).parents[1] / "cross_suite_generalization" / "export_visual_window_candidates_gpu.py"
SPEC = importlib.util.spec_from_file_location("window_jobs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_multi_event_jobs_exclude_outcome_label():
    manifest = {"episodes": [{
        "episode_id": "libero_goal-task00-episode000",
        "split": "development",
        "sidecar_status": "matched",
        "sidecar_path": "/tmp/example.npz",
        "success": False,
        "label_blind_low_response_events": [
            {"start_action_index": 10}, {"start_action_index": 40},
        ],
        "label_blind_matched_controls": [
            {"start_action_index": 70}, {"start_action_index": 100},
        ],
    }]}
    jobs = MODULE.build_label_blind_jobs(manifest, "development")
    assert [row["role"] for row in jobs] == ["event_0", "event_1", "control_0", "control_1"]
    assert [row["action_index"] for row in jobs] == [10, 40, 70, 100]
    assert all("success" not in row for row in jobs)
