import math

from vla_supervisor.articulation_provider import (
    ArticulationRelationContext,
    ValidatedArticulationProvider,
)


CTX = ArticulationRelationContext(
    relation_id="top_drawer_closed", predicate="Close",
    calibration_id="task0-close-v1", provenance="test_backend")


def call(provider, index):
    return provider({}, {}, [0] * 7, (), index)


def test_valid_backend_measurement_is_preserved():
    def backend(**_):
        return dict(area_fraction=.4, confidence=.9, observable=True,
                    identity_reliable=True, calibrated=True,
                    calibration_id="task0-close-v1")
    value = call(ValidatedArticulationProvider(backend, CTX), 4)
    assert value.area_fraction == .4
    assert value.observable and value.identity_reliable and value.calibrated
    assert value.relation_id == "top_drawer_closed"


def test_missing_or_invalid_backend_output_abstains():
    for raw in (None, {}, {"area_fraction": math.nan, "confidence": .9,
                           "observable": True, "identity_reliable": True,
                           "calibrated": True}):
        provider = ValidatedArticulationProvider(lambda **_: raw, CTX)
        value = call(provider, 0)
        assert not value.observable
        assert value.area_fraction is None


def test_wrong_calibration_and_stale_frame_cannot_claim_reliability():
    def backend(**_):
        return dict(area_fraction=.5, confidence=.9, observable=True,
                    identity_reliable=True, calibrated=True,
                    calibration_id="different-calibration")
    provider = ValidatedArticulationProvider(backend, CTX)
    first = call(provider, 2)
    stale = call(provider, 2)
    assert not first.calibrated
    assert not stale.observable
    assert not stale.identity_reliable


def test_backend_errors_fail_closed_and_reset_is_forwarded():
    class Backend:
        resets = 0
        def __call__(self, **_):
            raise ValueError("bad mask")
        def reset(self):
            self.resets += 1
    backend = Backend()
    provider = ValidatedArticulationProvider(backend, CTX)
    assert not call(provider, 0).observable
    provider.reset()
    assert backend.resets == 1
