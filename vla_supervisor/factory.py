from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .events import EventType
from .fusion import PersistenceRule, TemporalFusion
from .instruction_guard import GuardConfig, InstructionGuard
from .jsonl_log import Utf8JsonlLogger
from .monitors import (CallbackExecutionMonitor, NullExecutionMonitor,
                       FourStateExecutionMonitor, NullObjectResultMonitor,
                       PolicyStallMonitor, StallConfig)
from .diagnostic_router import (AmbiguityGatedObjectMonitor, GripperInteractionGate,
                                ShadowObjectMonitor, VisualRoutingConfig)
from .recovery import RecoveryConfig, RecoveryController
from .runtime import SupervisorRuntime
from .intervention import InterventionConfig, InterventionPlanner
from .contact_diagnosis import ContactDiagnosticObjectMonitor


@dataclass(frozen=True)
class SupervisorConfig:
    guard: GuardConfig = field(default_factory=GuardConfig)
    stall: StallConfig = field(default_factory=StallConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    persistence_required: int = 2
    persistence_window: int = 3
    minimum_confidence: float = .5


def create_safe_default(config: SupervisorConfig = SupervisorConfig(), log_path: str | Path | None = None):
    rule=PersistenceRule(config.persistence_required,config.persistence_window,config.minimum_confidence)
    logger=Utf8JsonlLogger(log_path) if log_path is not None else None
    runtime=SupervisorRuntime(InstructionGuard(config.guard),
        [NullExecutionMonitor(),PolicyStallMonitor(config.stall)],NullObjectResultMonitor(),
        TemporalFusion({EventType.EXECUTION_MISMATCH:rule,EventType.POLICY_STALL:rule,
                        EventType.OBJECT_FAILURE:rule}),RecoveryController(config.recovery),logger)
    runtime.component_status={"instruction_guard":"active","execution_consistency":"placeholder",
                              "policy_stall":"active","object_result":"placeholder"}
    return runtime


def create_source_calibrated_conditional(models, *, sequence_threshold: float,
                                         reliability_calibration=None,
                                         config: SupervisorConfig = SupervisorConfig(),
                                         log_path: str | Path | None = None):
    """Build the source-calibrated candidate; no target-domain guarantee."""
    from .conditional_execution import ConditionalDualExpertScorer
    persistence=PersistenceRule(config.persistence_required,config.persistence_window,
                                config.minimum_confidence)
    execution_rule=PersistenceRule(1,1,config.minimum_confidence)
    scorer=ConditionalDualExpertScorer(models,window=10,beta=1.0,
                                       reliability_calibration=reliability_calibration)
    logger=Utf8JsonlLogger(log_path) if log_path is not None else None
    runtime=SupervisorRuntime(InstructionGuard(config.guard),
        [CallbackExecutionMonitor(scorer,sequence_threshold),PolicyStallMonitor(config.stall)],
        NullObjectResultMonitor(),TemporalFusion({EventType.EXECUTION_MISMATCH:execution_rule,
        EventType.POLICY_STALL:persistence,EventType.OBJECT_FAILURE:persistence}),
        RecoveryController(config.recovery),logger)
    runtime.component_status={"instruction_guard":"active",
        "execution_consistency":"source_calibrated_candidate",
        "execution_target_domain_guarantee":False,"policy_stall":"active",
        "object_result":"placeholder"}
    return runtime


def create_selective_multimodal(execution_scorer, object_monitor, *,
                                routing: VisualRoutingConfig = VisualRoutingConfig(),
                                manipulation_phase_steps: int | None = 40,
                                conditioned_monitors=(),
                                config: SupervisorConfig = SupervisorConfig(),
                                log_path: str | Path | None = None):
    """Build execution triage -> ambiguity-gated vision -> temporal recovery.

    ``execution_scorer`` must emit the calibrated four-state mapping consumed by
    :class:`FourStateExecutionMonitor`. ``object_monitor`` must emit structured
    visual consequence events.  Clear execution alarms bypass vision and cannot
    be vetoed by it.
    """
    persistence = PersistenceRule(config.persistence_required,
                                  config.persistence_window,
                                  config.minimum_confidence)
    logger = Utf8JsonlLogger(log_path) if log_path is not None else None
    runtime = SupervisorRuntime(
        InstructionGuard(config.guard),
        [FourStateExecutionMonitor(execution_scorer), PolicyStallMonitor(config.stall)],
        AmbiguityGatedObjectMonitor(
            object_monitor, routing,
            None if manipulation_phase_steps is None else
            GripperInteractionGate(active_steps_after_transition=manipulation_phase_steps),
        ),
        TemporalFusion({EventType.EXECUTION_MISMATCH: persistence,
                        EventType.POLICY_STALL: persistence,
                        EventType.OBJECT_FAILURE: persistence}),
        RecoveryController(config.recovery), logger,
        conditioned_monitors=conditioned_monitors,
    )
    runtime.component_status = {
        "instruction_guard": "active",
        "execution_consistency": "calibrated_four_state_adapter",
        "visual_routing": "execution_ambiguity_with_gripper_interaction_phase",
        "object_result": "caller_supplied_not_yet_validated_online",
        "policy_stall": "active",
    }
    return runtime


def create_frozen_execution_with_task_consequence(
        model_paths, calibration_path, semantic_evidence_scorer, *,
        semantic_control_enabled: bool = False,
        config: SupervisorConfig = SupervisorConfig(),
        log_path: str | Path | None = None):
    """Build the mainline execution layer plus calibrated task-progress evidence.

    The task-consequence branch is shadow-only by default.  Enabling control
    is an explicit deployment decision and does not bypass temporal fusion,
    the recovery budget, the instruction guard, or bounded bridge actions.
    """
    from .conditional_execution import CalibratedFourStateConditionalScorer
    from .semantic_consequence import CalibratedSemanticConsequenceAdapter
    scorer = CalibratedFourStateConditionalScorer(
        model_paths, calibration_path=calibration_path,
    )
    semantic_monitor = CalibratedSemanticConsequenceAdapter(
        semantic_evidence_scorer, control_enabled=semantic_control_enabled)
    runtime = create_selective_multimodal(
        scorer, NullObjectResultMonitor(), manipulation_phase_steps=None,
        conditioned_monitors=(semantic_monitor,), config=config,
        log_path=log_path,
    )
    runtime.component_status.update({
        "execution_consistency": "frozen_calibrated_four_state_conditional_ensemble",
        "execution_models": [str(Path(path)) for path in model_paths],
        "execution_calibration": str(Path(calibration_path)),
        "object_result": "explicit_null_contact_diagnosis",
        "task_consequence": ("online_calibrated_control" if semantic_control_enabled
                             else "online_calibrated_shadow"),
        "task_consequence_control_enabled": bool(semantic_control_enabled),
    })
    return runtime


def create_frozen_execution_with_articulation_progress(
        model_paths, calibration_path, articulation_provider,
        articulation_config, *, articulation_control_enabled: bool = False,
        config: SupervisorConfig = SupervisorConfig(),
        log_path: str | Path | None = None):
    """Build execution supervision plus a fail-closed relation watchdog.

    The provider owns perception only.  The watchdog owns temporal semantics,
    and control authority remains an explicit deployment switch.  This split
    lets detector/tracker implementations change without silently changing
    recovery policy.
    """
    from .articulation_progress import ArticulationProgressMonitor
    from .conditional_execution import CalibratedFourStateConditionalScorer
    scorer = CalibratedFourStateConditionalScorer(
        model_paths, calibration_path=calibration_path,
    )
    relation_monitor = ArticulationProgressMonitor(
        articulation_provider, articulation_config,
        control_enabled=articulation_control_enabled,
    )
    runtime = create_selective_multimodal(
        scorer, NullObjectResultMonitor(), manipulation_phase_steps=None,
        conditioned_monitors=(relation_monitor,), config=config,
        log_path=log_path,
    )
    runtime.component_status.update({
        "execution_consistency": "frozen_calibrated_four_state_conditional_ensemble",
        "execution_models": [str(Path(path)) for path in model_paths],
        "execution_calibration": str(Path(calibration_path)),
        "object_result": "explicit_null_contact_diagnosis",
        "articulation_progress": ("online_calibrated_control"
                                  if articulation_control_enabled else
                                  "online_calibrated_shadow"),
        "articulation_control_enabled": bool(articulation_control_enabled),
        "articulation_calibration_id": articulation_config.calibration_id,
    })
    return runtime


def create_frozen_four_state_execution(model_paths, calibration_path, *,
                                       config: SupervisorConfig = SupervisorConfig(),
                                       log_path: str | Path | None = None):
    """Build the deployable CPU execution layer with explicit visual placeholder.

    This closes the real dual-expert execution path while keeping vision
    fail-closed until an online visual evidence scorer is supplied.
    """
    from .conditional_execution import CalibratedFourStateConditionalScorer
    scorer = CalibratedFourStateConditionalScorer(
        model_paths, calibration_path=calibration_path,
    )
    runtime = create_selective_multimodal(
        scorer, NullObjectResultMonitor(), config=config, log_path=log_path,
    )
    # The visual branch is deliberately null in this factory.  Ambiguity is
    # therefore logged for later diagnosis but cannot yet justify changing the
    # policy cadence; only explicit mismatch/stall alarms may intervene.
    runtime.intervention = InterventionPlanner(
        InterventionConfig(act_on_ambiguity=False)
    )
    runtime.component_status.update({
        "execution_consistency": "frozen_calibrated_four_state_conditional_ensemble",
        "execution_models": [str(Path(path)) for path in model_paths],
        "execution_calibration": str(Path(calibration_path)),
        "object_result": "explicit_null_until_online_visual_adapter",
    })
    return runtime


def create_frozen_execution_with_visual_consequence(
        model_paths, calibration_path, visual_evidence_scorer, *,
        visual_control_enabled: bool = False,
        config: SupervisorConfig = SupervisorConfig(),
        log_path: str | Path | None = None):
    """Build four-state execution plus a caller-supplied online consequence scorer.

    The scorer is responsible for explicit abstention and calibrated evidence;
    this factory does not silently substitute heuristic optical-flow scores.
    """
    from .conditional_execution import CalibratedFourStateConditionalScorer
    scorer = CalibratedFourStateConditionalScorer(
        model_paths, calibration_path=calibration_path,
    )
    object_monitor = ContactDiagnosticObjectMonitor(visual_evidence_scorer)
    if not visual_control_enabled:
        object_monitor = ShadowObjectMonitor(object_monitor)
    runtime = create_selective_multimodal(
        scorer, object_monitor, manipulation_phase_steps=None,
        config=config, log_path=log_path,
    )
    runtime.component_status.update({
        "execution_consistency": "frozen_calibrated_four_state_conditional_ensemble",
        "execution_models": [str(Path(path)) for path in model_paths],
        "execution_calibration": str(Path(calibration_path)),
        "object_result": ("online_calibrated_contact_consequence_control"
                          if visual_control_enabled else
                          "online_calibrated_contact_consequence_shadow"),
        "visual_control_enabled": bool(visual_control_enabled),
    })
    return runtime
