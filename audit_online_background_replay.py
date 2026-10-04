import json
from pathlib import Path

import numpy as np

from vla_supervisor.background_motion import estimate_background_motion


source_dir = Path(
    "/root/gpufree-data/libero-traces/e2e_mvp_20260918/"
    "libero90_task0_ep0_visual_replay.npz"
)
source = next(source_dir.glob("*.npz"))
data = np.load(source)


def audit(name):
    frames = data[name]
    rows = []
    for index in range(1, len(frames)):
        estimate = estimate_background_motion(frames[index - 1], frames[index])
        rows.append({
            "action_index": int(data["action_indices"][index]),
            "valid": estimate.valid,
            "confidence": estimate.confidence,
            "tracked_points": estimate.tracked_points,
            "inlier_ratio": estimate.inlier_ratio,
            "median_reprojection_error_px": estimate.median_reprojection_error_px,
            "spatial_coverage": estimate.spatial_coverage,
        })
    confidence = np.asarray([row["confidence"] for row in rows], float)
    valid = np.asarray([row["valid"] for row in rows], bool)
    return {
        "frames": len(frames),
        "pairs": len(rows),
        "valid_pairs": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "confidence_quantiles": {
            f"p{q:02d}": float(np.percentile(confidence, q))
            for q in (1, 5, 10, 50, 90, 95, 99)
        },
        "invalid_action_indices": [row["action_index"] for row in rows if not row["valid"]],
        "rows": rows,
    }


result = {
    "source": str(source),
    "agent_fixed": audit("agent_images"),
    "wrist": audit("wrist_images"),
}
output = Path(
    "/root/gpufree-data/supervisor-tools-v1/outputs/"
    "libero90_task0_ep0_online_background_audit.json"
)
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
for key in ("agent_fixed", "wrist"):
    summary = {k: v for k, v in result[key].items() if k != "rows"}
    print(key, json.dumps(summary))
print(output)
