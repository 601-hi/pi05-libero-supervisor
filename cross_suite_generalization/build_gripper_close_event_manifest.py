"""Build a causal, multi-close visual-event manifest.

Every open-to-close transition is an independent *event*, but events from one
episode remain statistically clustered.  No success label or semantic detector
output is used to select events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_events(record: dict, close_frames: list[int], pre: int, post: int) -> list[dict]:
    scored_steps = int(record["scored_steps"])
    events = []
    for ordinal, close_frame in enumerate(close_frames):
        close_frame = int(close_frame)
        events.append({
            "event_id": f'{record["anonymous_id"]}-close{ordinal:02d}-step{close_frame:04d}',
            "anonymous_id": record["anonymous_id"],
            "close_ordinal": ordinal,
            "close_frame": close_frame,
            "window_start": max(0, close_frame - pre),
            "window_end": min(scored_steps - 1, close_frame + post),
            "has_later_close": ordinal + 1 < len(close_frames),
            "supersedes_event_id": (
                None if ordinal == 0
                else f'{record["anonymous_id"]}-close{ordinal - 1:02d}-step{int(close_frames[ordinal - 1]):04d}'
            ),
        })
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-frames", type=int, default=15)
    parser.add_argument("--post-frames", type=int, default=40)
    args = parser.parse_args()

    features = json.loads(args.features.read_text(encoding="utf-8"))["records"]
    private = {
        row["anonymous_id"]: row
        for row in json.loads(args.private.read_text(encoding="utf-8"))["records"]
    }
    episodes = []
    for row in features:
        identity = private[row["anonymous_id"]]
        episode = dict(row)
        episode["sidecar"] = identity["original_sidecar"]
        episode["trace"] = identity["original_trace"]
        episode["events"] = build_events(
            row, row["close_frames"], args.pre_frames, args.post_frames
        )
        episodes.append(episode)

    result = {
        "schema_version": 1,
        "selection_firewall": [
            "events are selected only from gripper open-to-close transitions",
            "success, reward, language, semantic detections, and final outcome are not used",
            "overlapping event windows remain separate because each close is a distinct intervention hypothesis",
        ],
        "causal_rules": [
            "an event may diagnose only evidence at or after its close frame",
            "a later close may supersede an earlier empty-grasp hypothesis",
            "control-established persists until explicit release or loss evidence",
            "placement is evaluated only after control has been established",
            "events within an episode are correlated and are not independent evaluation samples",
        ],
        "parameters": {"pre_frames": args.pre_frames, "post_frames": args.post_frames},
        "counts": {
            "episodes": len(episodes),
            "events": sum(len(row["events"]) for row in episodes),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
