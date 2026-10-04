import numpy as np

from vla_supervisor.diagnostic_router import AmbiguityGatedObjectMonitor
from vla_supervisor.events import EventType
from vla_supervisor.fusion import PersistenceRule, TemporalFusion
from vla_supervisor.instruction_guard import GuardConfig, InstructionGuard
from vla_supervisor.monitors import FourStateExecutionMonitor, StructuredObjectResultMonitor
from vla_supervisor.recovery import RecoveryController
from vla_supervisor.runtime import SupervisorRuntime


def state(x=0.0):
    return {"eef_pos": [x, 0, 0], "joint_pos": [0] * 7, "joint_vel": [0] * 7}


class SequenceScorer:
    def __init__(self, rows):
        self.rows = iter(rows)

    def __call__(self, *_):
        return next(self.rows)


def test_end_to_end_visual_runs_only_for_execution_ambiguity():
    execution = FourStateExecutionMonitor(SequenceScorer([
        {"decision": "known_normal", "lr": -2, "reliability": .9},
        {"decision": "ambiguous_overlap", "lr": .1, "reliability": .8},
    ]))
    calls = {"count": 0}

    def relation(*_):
        calls["count"] += 1
        return {"object_visible_prob": 1, "target_visible_prob": 1,
                "gripper_object_contact_prob": .9, "object_in_target_prob": 0,
                "release_prob": 0, "confidence": .9}

    visual = AmbiguityGatedObjectMonitor(StructuredObjectResultMonitor(relation))
    runtime = SupervisorRuntime(
        InstructionGuard(GuardConfig()), [execution], visual,
        TemporalFusion({EventType.OBJECT_FAILURE: PersistenceRule(1, 1, .5)}),
        RecoveryController(),
    )
    runtime.install_chunk([[0] * 7] * 2)
    for index in range(2):
        action, _ = runtime.next_action(state(), index)
        runtime.observe_step(intended_action=action, state_before=state(), state_after=state(),
                             action_index=index, images_before={}, images_after={})
    assert calls["count"] == 1
    assert runtime.history[0]["events"][-1]["evidence"]["visual_invoked"] is False
    assert runtime.history[1]["events"][-1]["evidence"]["visual_invoked"] is True


def test_known_abnormal_is_actionable_without_visual_veto():
    execution = FourStateExecutionMonitor(SequenceScorer([
        {"decision": "known_abnormal", "lr": 4, "reliability": .9},
    ]))
    calls = {"count": 0}

    def relation(*_):
        calls["count"] += 1
        raise AssertionError("visual must not run for a clear execution alarm")

    runtime = SupervisorRuntime(
        InstructionGuard(GuardConfig()), [execution],
        AmbiguityGatedObjectMonitor(StructuredObjectResultMonitor(relation)),
        TemporalFusion({EventType.EXECUTION_MISMATCH: PersistenceRule(1, 1, .5)}),
        RecoveryController(),
    )
    runtime.install_chunk([[0] * 7])
    action, _ = runtime.next_action(state(), 0)
    decision = runtime.observe_step(intended_action=action, state_before=state(), state_after=state(),
                                    action_index=0, images_before={}, images_after={})
    assert calls["count"] == 0
    assert decision.reason is EventType.EXECUTION_MISMATCH
    assert decision.request_replan
