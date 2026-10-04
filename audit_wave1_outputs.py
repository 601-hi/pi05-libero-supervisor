import json
from pathlib import Path


output_dir = Path("outputs/causal_identity_holdout_wave1")
with Path("outputs/CAUSAL_IDENTITY_HOLDOUT_PUBLIC_V1.json").open(encoding="utf-8") as stream:
    public = json.load(stream)
with (output_dir / "predictions.json").open(encoding="utf-8") as stream:
    predictions = json.load(stream)

if isinstance(public, dict):
    collection_key = next((key for key in ("episodes", "items", "records", "holdout") if key in public), None)
    if collection_key is None:
        print("PUBLIC_KEYS", sorted(public))
        raise SystemExit("Could not identify public manifest collection")
    items = public[collection_key]
else:
    items = public
if isinstance(predictions, dict):
    prediction_key = next(
        (key for key in ("predictions", "items", "records", "episodes", "results") if isinstance(predictions.get(key), list)),
        None,
    )
    if prediction_key is None:
        print("PREDICTION_KEYS", {key: type(value).__name__ for key, value in predictions.items()})
        raise SystemExit("Could not identify prediction collection")
    rows = predictions[prediction_key]
else:
    rows = predictions
public_ids = [
    row.get("anonymous_id") or row.get("id")
    for row in items
    if (row.get("evaluation_wave") or row.get("wave")) == 1
    and row.get("family") == "rigid_object_transport"
]
if not public_ids:
    print("PUBLIC_TOP_LEVEL", {key: (type(value).__name__, len(value) if hasattr(value, "__len__") else None) for key, value in public.items()} if isinstance(public, dict) else type(public).__name__)
    print("PUBLIC_FIRST", items[0] if items else None)
prediction_ids = [row.get("anonymous_id") or row.get("id") for row in rows]

print("PUBLIC_WAVE1", len(public_ids))
print("PREDICTIONS", len(rows))
print("IDS_EXACT", public_ids == prediction_ids)
print("MISSING", sorted(set(public_ids) - set(prediction_ids)))
print("EXTRA", sorted(set(prediction_ids) - set(public_ids)))
print("LOCKED", sum(row.get("locked_candidate_id") is not None for row in rows))
print("PRED_KEYS", sorted(rows[0]))
print("FILES", sorted(path.name for path in output_dir.iterdir()))
