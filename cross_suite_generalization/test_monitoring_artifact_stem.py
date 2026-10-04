import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_module(path):
    spec = importlib.util.spec_from_file_location("monitoring_main_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_stem_separates_suites():
    path = Path(__file__).parents[1] / "work" / "server_current" / "monitoring_main.py"
    module = load_module(path)
    common = dict(seed=34, sampling_noise_seed=2026091134, translation_action_scale=1.0,
                  disturbance_start_mode="fixed", disturbance_num_steps=0)
    spatial = SimpleNamespace(**common, task_suite_name="libero_spatial")
    libero90 = SimpleNamespace(**common, task_suite_name="libero_90")
    a = module._episode_artifact_stem(spatial, 9, 0, None, "success")
    b = module._episode_artifact_stem(libero90, 9, 0, None, "success")
    assert a != b
    assert a.startswith("rollout_libero_spatial_")
    assert b.startswith("rollout_libero_90_")
