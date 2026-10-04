"""Cache deterministic projected semantic features for repeated head trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from train_semantic_visual_head import features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    x = features(data, args.projection_dim, args.seed)
    np.savez_compressed(
        args.out,
        features=x,
        episode_id=data["episode_id"],
        projection_dim=np.asarray(args.projection_dim),
        projection_seed=np.asarray(args.seed),
    )
    print(json.dumps({"rows": len(x), "dimension": x.shape[1], "finite": bool(np.isfinite(x).all())}))


if __name__ == "__main__":
    main()
