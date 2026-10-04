import json

import numpy as np

from vla_supervisor.events import EventType, RecommendedAction
from vla_supervisor.fusion import PersistenceRule, TemporalFusion
from vla_supervisor.instruction_guard import GuardConfig, InstructionGuard
from vla_supervisor.jsonl_log import Utf8JsonlLogger
from vla_supervisor.monitors import (CallbackExecutionMonitor, CallbackObjectResultMonitor,
                                     ContactAwareExecutionMonitor,
                                    NullObjectResultMonitor, PolicyStallMonitor, StallConfig,
                                    StructuredObjectResultMonitor)
from vla_supervisor.recovery import RecoveryConfig, RecoveryController, RecoveryState
from vla_supervisor.runtime import SupervisorRuntime
from vla_supervisor.semantic_consequence import (
    CalibratedSemanticConsequenceAdapter, SemanticConsequenceEvidence,
)


def state(x=0.0): return {"eef_pos": [x, 0, 0], "joint_pos": [0]*7, "joint_vel": [0]*7}


def runtime(log=None):
    return SupervisorRuntime(InstructionGuard(GuardConfig()),
        [PolicyStallMonitor(StallConfig(low_progress_run_threshold=2,gripper_reversal_threshold=99))],
        NullObjectResultMonitor(),
        TemporalFusion({EventType.POLICY_STALL:PersistenceRule(2,3,0.5)}),
        RecoveryController(RecoveryConfig(max_replans=1,cooldown_steps=2)),log)


def test_unsafe_instruction_immediately_clears_chunk():
    rt=runtime();rt.install_chunk([[2,0,0,0,0,0,0],[0]*7])
    action,decision=rt.next_action(state(),0)
    assert action is None and decision.reason is EventType.INSTRUCTION_UNSAFE
    assert decision.request_replan and not rt.chunk


def test_persistent_stall_interrupts_and_replans(tmp_path):
    path=tmp_path/"决策日志.jsonl"
    with Utf8JsonlLogger(path) as log:
        rt=runtime(log);rt.install_chunk([[1,0,0,0,0,0,1]]*8)
        for i in range(3):
            action,_=rt.next_action(state(),i);assert action is not None
            decision=rt.observe_step(intended_action=action,state_before=state(),state_after=state(),action_index=i)
        assert decision.action is RecommendedAction.STOP_AND_REPLAN
        assert decision.reason is EventType.POLICY_STALL and not rt.chunk
        rt.mark_replan_completed([[0]*7])
        assert rt.recovery.state is RecoveryState.COOLDOWN
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert any(x["event"]=="supervisor_decision" and x["decision"]["request_replan"] for x in rows)


def test_second_replan_beyond_budget_safe_stops():
    controller=RecoveryController(RecoveryConfig(max_replans=1,cooldown_steps=0))
    from vla_supervisor.events import SupervisorDecision
    alarm=SupervisorDecision(RecommendedAction.STOP_AND_REPLAN,EventType.POLICY_STALL,.9,
                             clear_remaining_chunk=True,request_replan=True)
    assert controller.apply(alarm).request_replan
    controller.replan_completed()
    assert controller.apply(alarm).action is RecommendedAction.SAFE_STOP


def test_pending_bridge_does_not_consume_another_replan_budget():
    from vla_supervisor.events import SupervisorDecision
    controller=RecoveryController(RecoveryConfig(max_replans=3,cooldown_steps=0))
    alarm=SupervisorDecision(RecommendedAction.STOP_AND_REPLAN,EventType.POLICY_STALL,.9,
                             clear_remaining_chunk=True,request_replan=True)
    assert controller.apply(alarm).action is RecommendedAction.STOP_AND_REPLAN
    assert controller.replans == 1
    repeated=controller.apply(alarm)
    assert repeated.action is RecommendedAction.CONTINUE
    assert controller.replans == 1


def test_cooldown_suppresses_stall_but_not_execution_fault():
    from vla_supervisor.events import SupervisorDecision
    controller=RecoveryController(RecoveryConfig(max_replans=3,cooldown_steps=2))
    stall=SupervisorDecision(RecommendedAction.STOP_AND_REPLAN,EventType.POLICY_STALL,.9,
                             clear_remaining_chunk=True,request_replan=True)
    execution=SupervisorDecision(RecommendedAction.STOP_AND_REPLAN,EventType.EXECUTION_MISMATCH,.9,
                                 clear_remaining_chunk=True,request_replan=True)
    controller.apply(stall);controller.replan_completed()
    assert controller.apply(stall).action is RecommendedAction.CONTINUE
    assert controller.apply(execution).request_replan


def test_callback_execution_and_object_monitors_share_fusion_interface():
    def execution(*_): return 2.0,.9,{"residual":"large"}
    def object_result(*_): return 2.0,.8,{"object_target":"outside"}
    guard=InstructionGuard(GuardConfig())
    fusion=TemporalFusion({EventType.EXECUTION_MISMATCH:PersistenceRule(2,3,.5),
                           EventType.OBJECT_FAILURE:PersistenceRule(2,3,.5)})
    rt=SupervisorRuntime(guard,[CallbackExecutionMonitor(execution,1.0)],
        CallbackObjectResultMonitor(object_result,1.0),fusion,RecoveryController())
    rt.install_chunk([[0]*7]*5)
    for i in range(3):
        action,_=rt.next_action(state(),i)
        decision=rt.observe_step(intended_action=action,state_before=state(),state_after=state(),action_index=i)
    assert decision.reason is EventType.EXECUTION_MISMATCH
    assert decision.request_replan and not rt.chunk


def test_contact_routes_high_execution_score_to_non_actionable_ambiguity():
    execution=lambda *_:(4.0,.9,{"likelihood_ratio":4.0})
    contact=lambda *_:(.8,.95,{"source":"independent_sensor"})
    monitor=ContactAwareExecutionMonitor(execution,2.0,contact)
    event=monitor.observe(intended_action=np.zeros(7),state_before=state(),state_after=state(),
                          history=(),action_index=4)
    assert event.event_type is EventType.EXECUTION_AMBIGUOUS
    assert event.recommended_action is RecommendedAction.REQUEST_MORE_EVIDENCE
    decision=TemporalFusion().update([event])
    assert decision.action is RecommendedAction.CONTINUE


def test_high_execution_score_without_contact_evidence_remains_actionable():
    execution=lambda *_:(4.0,.9,{})
    contact=lambda *_:(.1,.9,{"source":"independent_sensor"})
    monitor=ContactAwareExecutionMonitor(execution,2.0,contact)
    event=monitor.observe(intended_action=np.zeros(7),state_before=state(),state_after=state(),
                          history=(),action_index=4)
    assert event.event_type is EventType.EXECUTION_MISMATCH
    assert event.recommended_action is RecommendedAction.STOP_AND_REPLAN


def test_low_support_high_score_is_logged_as_ambiguous():
    monitor=CallbackExecutionMonitor(lambda *_:(4.0,0.0,{"support_status":"unknown_low_support"}),2.0)
    event=monitor.observe(intended_action=np.zeros(7),state_before=state(),state_after=state(),
                          history=(),action_index=5)
    assert event.event_type is EventType.EXECUTION_AMBIGUOUS
    assert event.recommended_action is RecommendedAction.REQUEST_MORE_EVIDENCE


def test_object_callback_history_has_no_measured_state():
    captured={}
    def scorer(_before,_after,_action,history):captured["history"]=history;return 0.0,1.0,{}
    monitor=CallbackObjectResultMonitor(scorer,1.0)
    monitor.observe(images_before={},images_after={},intended_action=np.zeros(7),
                    history=({"action_index":1,"intended_action":[0]*7,
                              "state_after":{"eef_pos":[1,2,3]},"reward":1},),action_index=2)
    assert captured["history"] == ({"action_index":1,"intended_action":[0]*7},)


def test_structured_object_monitor_detects_unexpected_drop():
    observations=iter([
        {"object_visible_prob":1,"target_visible_prob":1,"gripper_object_contact_prob":.9,
         "object_in_target_prob":0,"release_prob":0,"confidence":.9},
        {"object_visible_prob":1,"target_visible_prob":1,"gripper_object_contact_prob":.1,
         "object_in_target_prob":0,"release_prob":0,"confidence":.9}])
    monitor=StructuredObjectResultMonitor(lambda *_:next(observations))
    first=monitor.observe(images_before={},images_after={},intended_action=np.zeros(7),history=(),action_index=1)
    second=monitor.observe(images_before={},images_after={},intended_action=np.zeros(7),history=(),action_index=2)
    assert first.event_type is EventType.NORMAL
    assert second.event_type is EventType.OBJECT_FAILURE
    assert second.evidence["failure_mechanism"] == "unexpected_drop"


def test_structured_object_monitor_routes_occlusion_to_ambiguity():
    relation={"object_visible_prob":.1,"target_visible_prob":1,"gripper_object_contact_prob":0,
              "object_in_target_prob":0,"release_prob":0,"confidence":.8}
    event=StructuredObjectResultMonitor(lambda *_:relation).observe(
        images_before={},images_after={},intended_action=np.zeros(7),history=(),action_index=1)
    assert event.event_type is EventType.OBJECT_AMBIGUOUS
    assert TemporalFusion().update([event]).action is RecommendedAction.CONTINUE


def test_conditioned_task_progress_reaches_bounded_recovery_target(tmp_path):
    def semantic_scorer(*_):
        return SemanticConsequenceEvidence(
            True, True, no_progress_score=.95, progress_score=.05,
            metadata={"phase": "place", "phase_reliable": True},
        )

    semantic = CalibratedSemanticConsequenceAdapter(
        semantic_scorer, control_enabled=True)
    log_path = tmp_path / "在线语义链路.jsonl"
    with Utf8JsonlLogger(log_path) as log:
        rt = SupervisorRuntime(
            InstructionGuard(GuardConfig()), [], NullObjectResultMonitor(),
            TemporalFusion({EventType.OBJECT_FAILURE: PersistenceRule(1, 1, .5)}),
            RecoveryController(), log, conditioned_monitors=(semantic,),
        )
        rt.install_chunk([[0] * 7] * 5)
        for index in range(2):
            action, _ = rt.next_action(state(), index)
            decision = rt.observe_step(
                intended_action=action, state_before=state(), state_after=state(),
                action_index=index, images_before={}, images_after={})
    assert decision.request_replan
    assert rt.last_intervention.recovery_target == "restore_goal_relation"
    assert rt.last_intervention.mode.value == "hold_then_replan"
    assert "restore the required goal relation" in rt.last_intervention.recovery_prompt_suffix
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    last = [row for row in rows if row["event"] == "supervisor_decision"][-1]
    assert last["intervention"]["recovery_target"] == "restore_goal_relation"
    assert any(event["source"] == "calibrated_semantic_task_consequence"
               for event in last["events"])


def test_uncalibrated_conditioned_monitor_cannot_clear_policy_queue():
    semantic = CalibratedSemanticConsequenceAdapter(
        lambda *_: SemanticConsequenceEvidence(
            True, False, wrong_object_score=.99,
            metadata={"phase": "grasp", "phase_reliable": True}),
        control_enabled=True,
    )
    rt = SupervisorRuntime(
        InstructionGuard(GuardConfig()), [], NullObjectResultMonitor(),
        TemporalFusion({EventType.OBJECT_FAILURE: PersistenceRule(1, 1, .5)}),
        RecoveryController(), conditioned_monitors=(semantic,),
    )
    rt.install_chunk([[0] * 7] * 3)
    action, _ = rt.next_action(state(), 0)
    decision = rt.observe_step(
        intended_action=action, state_before=state(), state_after=state(),
        action_index=0, images_before={}, images_after={})
    assert decision.action is RecommendedAction.CONTINUE
    assert len(rt.chunk) == 2
    assert rt.last_intervention.recovery_target == "request_more_evidence"


def test_recovery_prompt_is_active_only_while_replan_is_pending():
    rt = runtime()
    nominal, active, mechanism = rt.build_policy_prompt(
        "put the object in the target", enabled=True, fallback_suffix="retry")
    assert nominal == "put the object in the target"
    assert not active and mechanism == "none"

    rt.install_chunk([[1, 0, 0, 0, 0, 0, 1]] * 8)
    for index in range(3):
        action, _ = rt.next_action(state(), index)
        rt.observe_step(
            intended_action=action, state_before=state(), state_after=state(),
            action_index=index)
    assert rt.recovery.state is RecoveryState.REPLAN_PENDING
    prompt, active, mechanism = rt.build_policy_prompt(
        "put the object in the target", enabled=True, fallback_suffix="retry")
    assert active and mechanism == "policy_stall"
    assert "Immediate recovery" in prompt

    rt.mark_replan_completed([[0] * 7])
    prompt, active, mechanism = rt.build_policy_prompt(
        "put the object in the target", enabled=True, fallback_suffix="retry")
    assert prompt == "put the object in the target"
    assert not active and mechanism == "none"


def test_novelty_only_window_logs_stall_without_reclaiming_control():
    rt = runtime()
    rt.install_chunk([[1, 0, 0, 0, 0, 0, 1]] * 8)
    rt.begin_novelty_only_window(3)
    for index in range(3):
        action, _ = rt.next_action(state(), index)
        decision = rt.observe_step(
            intended_action=action, state_before=state(), state_after=state(),
            action_index=index)
        assert rt.last_observation_novelty_only
    assert decision.reason is EventType.POLICY_STALL
    assert len(rt.chunk) == 5
    assert rt.recovery.state is RecoveryState.RUNNING
    assert rt.novelty_only_steps_remaining == 0

    action, _ = rt.next_action(state(), 3)
    decision = rt.observe_step(
        intended_action=action, state_before=state(), state_after=state(),
        action_index=3)
    assert not rt.last_observation_novelty_only
    assert decision.request_replan


def test_novelty_only_window_does_not_disable_instruction_guard():
    rt = runtime()
    rt.install_chunk([[2, 0, 0, 0, 0, 0, 0]])
    rt.begin_novelty_only_window(1)
    action, decision = rt.next_action(state(), 0)
    assert action is None
    assert decision.reason is EventType.INSTRUCTION_UNSAFE
