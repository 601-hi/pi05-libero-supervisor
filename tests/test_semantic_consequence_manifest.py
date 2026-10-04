import importlib.util
from pathlib import Path
import sys
import types


libero = types.ModuleType("libero")
libero_libero = types.ModuleType("libero.libero")
libero_libero.benchmark = object()
sys.modules.setdefault("libero", libero)
sys.modules.setdefault("libero.libero", libero_libero)
MODULE_PATH = Path(__file__).parents[1] / "cross_suite_generalization" / "build_semantic_consequence_manifest.py"
SPEC = importlib.util.spec_from_file_location("semantic_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_language_goal_families():
    assert MODULE.runtime_family("turn off the stove") == "binary_state_change"
    assert MODULE.runtime_family("close the top drawer") == "articulated_state_change"
    assert MODULE.runtime_family("put the bowl in the drawer") == "containment_relation"
    assert MODULE.runtime_family("put the bowl on the plate") == "support_relation"
    assert MODULE.runtime_family("pick up the bowl in the drawer and place it on the plate") == "support_relation"
    assert MODULE.runtime_family("pick up the soup and place it in the basket") == "containment_relation"


def test_offline_predicate_parser(tmp_path):
    path = tmp_path / "task.bddl"
    path.write_text("(:goal (And (In bowl_1 drawer_1) (Turnoff stove_1)))", encoding="utf-8")
    assert MODULE.parse_offline_goal(path) == [
        {"predicate": "In", "arguments": ["bowl_1", "drawer_1"]},
        {"predicate": "Turnoff", "arguments": ["stove_1"]},
    ]
