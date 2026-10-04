import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("label_outcome_risk_candidates.py")
SPEC = importlib.util.spec_from_file_location("labels", MODULE_PATH)
labels = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(labels)


def test_suite_inference_is_explicit_and_conservative():
    assert labels.infer_suite("/x/libero_90/run.jsonl") == "libero_90"
    assert labels.infer_suite("/x/normal_spatial_seed.jsonl") == "libero_spatial"
    assert labels.infer_suite("/x/unidentified.jsonl") == "unknown"


def test_auc_pairwise_with_ties():
    assert labels.auc([0, 0, 1, 1], [0.0, 1.0, 1.0, 2.0]) == 0.875


def test_robust_scale_never_collapses():
    location, scale = labels.robust_location_scale([2.0] * 10)
    assert location == 2.0
    assert scale > 0.0
