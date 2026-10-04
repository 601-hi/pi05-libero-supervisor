import numpy as np

from vla_supervisor.events import EventType, MonitorEvent, RecommendedAction, SupervisorDecision
from vla_supervisor.intervention import InterventionConfig, InterventionMode, InterventionPlanner
from vla_supervisor.fusion import TemporalFusion
from vla_supervisor.instruction_guard import GuardConfig, InstructionGuard
from vla_supervisor.recovery import RecoveryController
from vla_supervisor.runtime import SupervisorRuntime


def event(kind, **evidence):
    return MonitorEvent(kind, 1.0, .9, evidence=evidence)


def test_ambiguity_reduces_horizon_without_forcing_replan():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.CONTINUE, EventType.NORMAL, 1.0),
        [event(EventType.EXECUTION_AMBIGUOUS)], np.ones(7),
    )
    assert directive.mode is InterventionMode.CAUTIOUS_CONTINUE
    assert directive.truncate_remaining_to == 1
    assert not directive.replan_after_bridge


def test_ambiguity_is_observational_when_independent_diagnosis_is_offline():
    directive = InterventionPlanner(
        InterventionConfig(act_on_ambiguity=False)
    ).plan(
        SupervisorDecision(RecommendedAction.CONTINUE, EventType.NORMAL, 1.0),
        [event(EventType.EXECUTION_AMBIGUOUS)], np.ones(7),
    )
    assert directive.mode is InterventionMode.CAUTIOUS_CONTINUE
    assert directive.next_chunk_horizon == 5
    assert directive.truncate_remaining_to is None


def test_execution_fault_holds_then_replans():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.EXECUTION_MISMATCH, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.EXECUTION_MISMATCH)], [1, 2, 3, 4, 5, 6, -1],
    )
    assert directive.mode is InterventionMode.HOLD_THEN_REPLAN
    assert directive.bridge_actions == ((0, 0, 0, 0, 0, 0, -1),)


def test_jam_generates_bounded_reverse_translation():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.OBJECT_FAILURE, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.OBJECT_FAILURE, diagnostic_state="fixed_obstacle_or_jam")],
        [1, 0, 0, .5, .5, .5, 1],
    )
    assert directive.mode is InterventionMode.RETREAT_THEN_REPLAN
    assert len(directive.bridge_actions) == 2
    assert directive.bridge_actions[0][:6] == (-.15, 0, 0, 0, 0, 0)
    assert directive.bridge_actions[0][6] == 1


def test_persistent_stall_can_opt_into_bounded_retreat():
    directive = InterventionPlanner(
        InterventionConfig(retreat_on_policy_stall=True)
    ).plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.POLICY_STALL, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.POLICY_STALL)], [1, 0, 0, 0, 0, 0, -1],
    )
    assert directive.mode is InterventionMode.RETREAT_THEN_REPLAN
    assert len(directive.bridge_actions) == 2
    assert directive.bridge_actions[0][:3] == (-.15, 0, 0)


def test_jam_without_direction_falls_back_to_hold():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.OBJECT_FAILURE, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.OBJECT_FAILURE, diagnostic_state="fixed_obstacle_or_jam")],
        [0] * 7,
    )
    assert directive.mode is InterventionMode.HOLD_THEN_REPLAN


def test_empty_grasp_releases_then_retreats_before_replan():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.OBJECT_FAILURE, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.OBJECT_FAILURE, diagnostic_state="empty_grasp_or_miss")],
        [1, 0, 0, .5, .5, .5, 1],
    )
    assert directive.mode is InterventionMode.RETREAT_THEN_REPLAN
    assert len(directive.bridge_actions) == 3
    assert directive.bridge_actions[0] == (0, 0, 0, 0, 0, 0, -1)
    assert directive.bridge_actions[1][:6] == (-.15, 0, 0, 0, 0, 0)
    assert directive.bridge_actions[1][6] == -1
    assert directive.bridge_actions[2] == directive.bridge_actions[1]
    assert directive.replan_after_bridge


def test_wrong_object_without_safe_retreat_still_releases():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.OBJECT_FAILURE, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.OBJECT_FAILURE, diagnostic_state="wrong_object_control")],
        [0, 0, 0, 0, 0, 0, 1],
    )
    assert directive.mode is InterventionMode.RETREAT_THEN_REPLAN
    assert directive.bridge_actions == ((0, 0, 0, 0, 0, 0, -1),)
    assert directive.replan_after_bridge


def test_instruction_unsafe_discards_without_bridge_motion():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.INSTRUCTION_UNSAFE, 1.0,
                           clear_remaining_chunk=True, request_replan=True), [], [1] * 7,
    )
    assert directive.mode is InterventionMode.REPLAN_SHORT
    assert directive.bridge_actions == ()


class AmbiguousMonitor:
    def reset(self):
        pass

    def observe(self, **kwargs):
        return event(EventType.EXECUTION_AMBIGUOUS)


class NullObject:
    def reset(self):
        pass

    def observe(self, **kwargs):
        return event(EventType.NORMAL)


def test_runtime_truncates_remaining_chunk_after_ambiguity():
    runtime = SupervisorRuntime(
        InstructionGuard(GuardConfig()), [AmbiguousMonitor()], NullObject(),
        TemporalFusion(), RecoveryController(),
    )
    runtime.install_chunk([[0] * 7] * 5)
    action, _ = runtime.next_action({"joint_pos": [0] * 7, "joint_vel": [0] * 7}, 0)
    decision = runtime.observe_step(
        intended_action=action,
        state_before={"eef_pos": [0, 0, 0]},
        state_after={"eef_pos": [0, 0, 0]},
        action_index=0,
    )
    assert decision.action is RecommendedAction.CONTINUE
    assert runtime.last_intervention.mode is InterventionMode.CAUTIOUS_CONTINUE
    assert len(runtime.chunk) == 1


def test_shadow_runtime_logs_ambiguity_without_mutating_chunk():
    runtime = SupervisorRuntime(
        InstructionGuard(GuardConfig()), [AmbiguousMonitor()], NullObject(),
        TemporalFusion(), RecoveryController(), control_enabled=False,
    )
    runtime.install_chunk([[0] * 7] * 5)
    action, _ = runtime.next_action({"joint_pos": [0] * 7, "joint_vel": [0] * 7}, 0)
    decision = runtime.observe_step(
        intended_action=action,
        state_before={"eef_pos": [0, 0, 0]},
        state_after={"eef_pos": [0, 0, 0]},
        action_index=0,
    )
    assert decision.action is RecommendedAction.CONTINUE
    assert runtime.last_intervention.mode is InterventionMode.CAUTIOUS_CONTINUE
    assert len(runtime.chunk) == 4
    assert runtime.recovery.replans == 0


def test_runtime_stages_guarded_hold_then_exposes_replan_boundary():
    class FaultMonitor:
        def reset(self):
            pass

        def observe(self, **kwargs):
            return event(EventType.EXECUTION_MISMATCH)

    from vla_supervisor.fusion import PersistenceRule
    runtime = SupervisorRuntime(
        InstructionGuard(GuardConfig()), [FaultMonitor()], NullObject(),
        TemporalFusion({EventType.EXECUTION_MISMATCH: PersistenceRule(1, 1, .5)}),
        RecoveryController(),
    )
    state = {"eef_pos": [0, 0, 0], "joint_pos": [0] * 7, "joint_vel": [0] * 7}
    runtime.install_chunk([[.5, 0, 0, 0, 0, 0, -1]] * 3)
    action, _ = runtime.next_action(state, 0)
    decision = runtime.observe_step(
        intended_action=action, state_before=state, state_after=state, action_index=0,
    )
    assert decision.request_replan and not runtime.chunk
    assert runtime.stage_intervention_bridge()
    bridge, pre = runtime.next_action(state, 1)
    assert pre.action is RecommendedAction.CONTINUE
    assert np.allclose(bridge[:6], 0) and bridge[6] == -1
    assert runtime.needs_policy_replan
    runtime.mark_replan_completed([[.1, 0, 0, 0, 0, 0, -1]])
    assert not runtime.needs_policy_replan
    assert runtime.recommended_chunk_horizon == 1


def test_wrong_object_directive_exposes_recovery_milestone():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.OBJECT_FAILURE, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.OBJECT_FAILURE, diagnostic_state="wrong_object_control")],
        [1, 0, 0, 0, 0, 0, 1],
    )
    assert directive.recovery_target == "release_wrong_object"
    assert directive.recovery_target_confidence == .9
    assert directive.recovery_prompt_suffix.startswith("Immediate recovery milestone:")


def test_stall_alone_does_not_invent_recovery_milestone():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.POLICY_STALL, .95,
                           clear_remaining_chunk=True, request_replan=True),
        [MonitorEvent(EventType.POLICY_STALL, 1.0, .95, source="policy_stall")],
        [0] * 7,
    )
    assert directive.recovery_target == "request_more_evidence"
    assert directive.recovery_target_confidence == .95


def test_reliable_goal_progress_stall_changes_replan_milestone_without_risky_motion():
    directive = InterventionPlanner().plan(
        SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, EventType.OBJECT_FAILURE, .9,
                           clear_remaining_chunk=True, request_replan=True),
        [event(EventType.OBJECT_FAILURE, diagnostic_state="goal_relation_progress_stalled")],
        [1, 0, 0, 0, 0, 0, 1],
    )
    assert directive.recovery_target == "restore_goal_relation"
    assert directive.mode is InterventionMode.HOLD_THEN_REPLAN
    assert directive.bridge_actions == ((0, 0, 0, 0, 0, 0, 1),)
    assert "restore the required goal relation" in directive.recovery_prompt_suffix
