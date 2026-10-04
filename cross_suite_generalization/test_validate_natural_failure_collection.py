import json
import subprocess
import sys

import numpy as np


def test_minimal_aligned_collection_passes(tmp_path):
    root = tmp_path / "run"
    traces = root / "traces"; visuals = root / "visual_sidecars"
    traces.mkdir(parents=True); visuals.mkdir()
    rows = [
        {"event": "step", "task_id": 0, "episode_idx": 0, "action_index": 0,
         "disturbance_active": False, "visual_frame_index": 0},
        {"event": "episode_end", "task_id": 0, "episode_idx": 0,
         "success": True, "disturbed_steps": 0},
    ]
    (traces / "sample.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8"
    )
    np.savez_compressed(
        visuals / "rollout_task00_episode000_fixed_startnone_success.npz",
        action_indices=np.asarray([0], np.int32),
        agent_images=np.zeros((1, 2, 2, 3), np.uint8),
        wrist_images=np.zeros((1, 2, 2, 3), np.uint8),
    )
    output = tmp_path / "validation.json"
    script = __file__.replace("test_validate_", "validate_")
    result = subprocess.run(
        [sys.executable, script, "--root", str(root), "--expected-episodes", "1",
         "--output", str(output)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["status"] == "PASS"
