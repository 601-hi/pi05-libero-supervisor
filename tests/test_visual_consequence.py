import numpy as np

from vla_supervisor.adaptive_motion_baseline import AdaptiveMotionBaseline
from vla_supervisor.visual_consequence import BackgroundConsequenceEvidenceScorer


def textured_frame(shift=0):
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    for y in range(8, 88, 12):
        for x in range(8, 88, 12):
            xx = x + shift
            if 1 <= xx < 94:
                image[y-2:y+3, xx-2:xx+3] = (180, 220, 80)
    return image


def test_missing_candidate_mask_abstains_instead_of_inventing_probability():
    scorer = BackgroundConsequenceEvidenceScorer()
    result = scorer(
        {"agent": textured_frame(), "wrist": textured_frame()},
        {"agent": textured_frame(1), "wrist": textured_frame(1)},
        [0, 0, 0, 0, 0, 0, -1], (),
    )
    assert result["independent_motion_probability"] is None
    assert result["blocked_motion_probability"] is None
    assert scorer.last_diagnostics["abstention_reason"] == "candidate_mask_unavailable"


def test_candidate_measurement_requires_explicit_calibrator_for_probability():
    mask = np.zeros((96, 96), dtype=bool)
    mask[30:60, 30:60] = True
    provider = lambda *_: {"candidate_mask": mask, "gripper_candidate_proximity": .8}
    scorer = BackgroundConsequenceEvidenceScorer(provider)
    result = scorer(
        {"agent": textured_frame(), "wrist": textured_frame()},
        {"agent": textured_frame(1), "wrist": textured_frame(1)},
        [1, 0, 0, 0, 0, 0, 1], (),
    )
    assert scorer.last_diagnostics["status"] == "candidate_measured"
    assert result["independent_motion_probability"] is None
    assert result["gripper_candidate_proximity"] == .8


def test_frozen_calibrator_is_the_only_path_to_motion_probability():
    mask = np.ones((96, 96), dtype=bool)
    provider = lambda *_: {"candidate_mask": mask}
    scorer = BackgroundConsequenceEvidenceScorer(
        provider, motion_calibrator=lambda features: .75,
        blocked_calibrator=lambda features: .2,
    )
    result = scorer(
        {"agent": textured_frame(), "wrist": textured_frame()},
        {"agent": textured_frame(1), "wrist": textured_frame(1)},
        [1, 0, 0, 0, 0, 0, 1], (),
    )
    assert result["independent_motion_probability"] == .75
    assert result["blocked_motion_probability"] == .2


def test_missing_after_frame_fails_closed():
    scorer = BackgroundConsequenceEvidenceScorer()
    result = scorer({"agent": textured_frame()}, None, [0] * 7, ())
    assert result["background_confidence"] == 0
    assert result["independent_motion_probability"] is None
    assert scorer.last_diagnostics["status"] == "missing_paired_images"


def test_relative_motion_baseline_is_diagnostic_not_a_probability():
    mask = np.ones((96, 96), dtype=bool)
    provider = lambda *_: {"candidate_mask": mask, "allow_reference_update": True}
    scorer = BackgroundConsequenceEvidenceScorer(
        provider, adaptive_motion_baseline=AdaptiveMotionBaseline(
            minimum_samples=2, minimum_scale=.001, minimum_background_confidence=0
        )
    )
    for shift in (1, 2, 3):
        result = scorer(
            {"agent": textured_frame(), "wrist": textured_frame()},
            {"agent": textured_frame(shift), "wrist": textured_frame(shift)},
            [1, 0, 0, 0, 0, 0, 1], (),
        )
    assert scorer.last_diagnostics["relative_motion"]["ready"]
    assert result["independent_motion_probability"] is None
    assert result["blocked_motion_probability"] is None
