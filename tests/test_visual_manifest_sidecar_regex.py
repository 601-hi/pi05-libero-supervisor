import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "cross_suite_generalization" / "build_visual_development_manifest.py"
SPEC = importlib.util.spec_from_file_location("visual_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sidecar_regex_captures_suite_and_indices():
    match = MODULE.SIDECAR_RE.fullmatch(
        "rollout_libero_goal_seed36_noise2026091836_task02_episode003_"
        "fixed_startnone_n000_scale1p000_success.npz"
    )
    assert match is not None
    assert match.groups() == ("libero_goal", "02", "003", "success")


def test_sidecar_regex_keeps_legacy_format_compatible():
    match = MODULE.SIDECAR_RE.fullmatch(
        "rollout_seed34_noise20260919_task00_episode001_fixed_failure.npz"
    )
    assert match is not None
    assert match.groups() == (None, "00", "001", "failure")
