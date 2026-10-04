import numpy as np
import json

from vla_supervisor.conditional_execution import CalibratedFourStateConditionalScorer


class BaseStub:
    def __init__(self, evidence):
        self.evidence = evidence

    def reset(self):
        pass

    def __call__(self, *_):
        return 0.0, 0.0, dict(self.evidence)


class CalibrationStub:
    def score(self, normal, abnormal, ensemble_std, sensor_quality):
        assert sensor_quality == 1.0
        return {"decision": "known_abnormal", "lr": abnormal-normal, "reliability": .8}


def scorer_with_stubs(evidence):
    scorer = object.__new__(CalibratedFourStateConditionalScorer)
    scorer.base = BaseStub(evidence)
    scorer.calibration = CalibrationStub()
    from cross_suite_generalization.reliability_sequence_supervisor import SensorQualityMonitor
    scorer.sensor = SensorQualityMonitor(expected_dt=.05)
    scorer.expected_dt = .05
    return scorer


def state(eef=(0, 0, 0), joint_vel=(0,) * 7):
    return {"eef_pos": eef, "joint_vel": joint_vel}


def test_warmup_is_unknown_not_normal():
    result = scorer_with_stubs({"status": "warming_up"})(None, state(), state(), ())
    assert result["decision"] == "unknown"
    assert result["reliability"] == 0


def test_ready_scores_are_calibrated_to_four_state():
    result = scorer_with_stubs({
        "normal_logp_mean": -8.0, "abnormal_logp_mean": 3.0, "ensemble_std": .1,
        "sequence_window_fill": 10, "sequence_window": 10,
    })(None, state(), state((.001, 0, 0)), ())
    assert result["decision"] == "known_abnormal"
    assert result["lr"] == 11.0
    assert result["four_state_status"] == "ready"


def test_bad_sensor_quality_forces_calibrator_to_receive_zero():
    class UnknownCalibration:
        def score(self, normal, abnormal, ensemble_std, sensor_quality):
            assert sensor_quality == 0.0
            return {"decision": "unknown", "lr": abnormal-normal, "reliability": 0.0}
    scorer = scorer_with_stubs({
        "normal_logp_mean": -3.0, "abnormal_logp_mean": -2.0, "ensemble_std": .1,
        "sequence_window_fill": 10, "sequence_window": 10,
    })
    scorer.calibration = UnknownCalibration()
    result = scorer(None, state(), state((.2, 0, 0)), ())
    assert result["decision"] == "unknown"


def test_partial_sequence_window_is_unknown_even_with_log_likelihoods():
    result = scorer_with_stubs({
        "normal_logp_mean": -8.0, "abnormal_logp_mean": 3.0, "ensemble_std": .1,
        "sequence_window_fill": 3, "sequence_window": 10,
    })(None, state(), state(), ())
    assert result["decision"] == "unknown"
    assert result["four_state_status"] == "accumulating_sequence"


def test_factory_requires_real_model_files(tmp_path):
    from vla_supervisor.factory import create_frozen_four_state_execution
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "source_only": True, "overlap_warning": False,
        "best_logp_reference": [-2, -1], "ensemble_std_reference": [.1, .2],
        "normal_lr_threshold": -1, "abnormal_lr_threshold": 1,
    }), encoding="utf-8")
    try:
        create_frozen_four_state_execution([tmp_path/"missing"]*3, calibration)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing deployment models must fail closed")
