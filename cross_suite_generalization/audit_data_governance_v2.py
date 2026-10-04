#!/usr/bin/env python3
"""Dependency-light checks for the frozen zero-shot data governance contract."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    governance = json.loads((ROOT / "DATA_GOVERNANCE_V2.json").read_text(encoding="utf-8"))
    datasets = json.loads((ROOT / "EXTERNAL_DATASET_AUDIT_V1.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "COMMON_DYNAMICS_SCHEMA_V1.json").read_text(encoding="utf-8"))

    libero = governance["libero_domains"]
    assert libero["libero_spatial"]["status"] == "development"
    assert libero["libero_object"]["status"] != "sealed_candidate_final_test"
    assert libero["libero_goal"]["status"] != "sealed_candidate_final_test"
    assert libero["libero_10"]["status"] == "sealed_final_test"
    assert libero["libero_10"]["provisional"] is False
    assert governance["external_training_policy"]["target_domain_data_allowed_for_zero_shot_fit"] is False
    assert governance["external_training_policy"]["target_domain_data_allowed_for_zero_shot_thresholds"] is False

    names = [item["name"] for item in datasets["candidates"]]
    assert len(names) == len(set(names))
    assert datasets["candidates"][0]["name"] == "DROID"
    assert datasets["candidates"][0]["decision"] == "primary_schema_pilot"
    assert "dataset_id_allowed_as_detector_feature" in governance["external_training_policy"]
    assert "missing or ambiguous command semantics" in schema["exclusion_rules"]

    print(json.dumps({
        "ok": True,
        "sealed_final_test": "libero_10",
        "external_candidates": names,
        "primary_schema_pilot": "DROID",
        "zero_shot_target_domain_fit": False,
        "zero_shot_target_domain_thresholds": False
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
