import collections
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import sys
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image, ImageDraw
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from episode_schedule import reset_env_preserving_episode_index

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
PI05_ACTION_HORIZON = 10
PI05_ACTION_DIM = 32
FIXED_NOISE_PROTOCOL_VERSION = 1


def _annotate_recovery_frame(frame, *, action_index, source, state,
                             target_action_index, candidate_id,
                             join_steps, replay_steps,
                             predicted_environment_clearance_m):
    """Overlay causal recovery diagnostics on the already-saved RGB frame."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    lines = [
        f"action={action_index} source={source or 'policy'} state={state or '-'}",
        f"target={target_action_index if target_action_index is not None else '-'} "
        f"candidate={candidate_id or '-'}",
        f"join={join_steps} replay={replay_steps} env_clearance="
        f"{predicted_environment_clearance_m * 1000:.2f}mm"
        if predicted_environment_clearance_m is not None
        else f"join={join_steps} replay={replay_steps} env_clearance=-",
    ]
    draw.rectangle((0, 0, image.width, 35), fill=(0, 0, 0))
    for line_index, line in enumerate(lines):
        draw.text((3, 2 + 11 * line_index), line, fill=(255, 255, 255))
    return np.asarray(image)


def _camera_intrinsic(sim, camera_name, image_height, image_width):
    """Camera intrinsics in the saved image scale, without optional h5py dependencies."""
    camera_id = sim.model.camera_name2id(camera_name)
    fovy = float(sim.model.cam_fovy[camera_id])
    focal = 0.5 * image_height / np.tan(fovy * np.pi / 360.0)
    return np.asarray([[focal, 0.0, image_width / 2.0],
                       [0.0, focal, image_height / 2.0], [0.0, 0.0, 1.0]])


def _camera_to_world(sim, camera_name):
    """OpenCV-axis camera pose in world coordinates, matching robosuite geometry."""
    camera_id = sim.model.camera_name2id(camera_name)
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3)
    pose[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id])
    axis_correction = np.diag([1.0, -1.0, -1.0, 1.0])
    return pose @ axis_correction


def _synchronize_robot_controller_to_current_state(env):
    """Discard a stale policy target when recovery takes ownership.

    A MuJoCo state snapshot does not include robosuite controller caches.
    Recovery deliberately starts from the measured pose, so both the live
    takeover and snapshot replay must reset the controller's null-space and
    end-effector goals to that same pose.
    """
    robot = env.robots[0]
    controller = robot.controller
    joint_positions = np.asarray(
        env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=float).copy()
    if isinstance(controller, dict):
        for arm_controller in controller.values():
            arm_controller.update_initial_joints(joint_positions)
    else:
        controller.update_initial_joints(joint_positions)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    # Optional comma-separated initial-state indexes, e.g. "0,2".  This is
    # evaluation scheduling only and never enters policy or supervisor input.
    episode_indices: Optional[str] = None

    # Optional controlled execution disturbance. The policy output is preserved as the intended action, while only the
    # translation components sent to the environment are scaled during the selected action-index interval.
    disturbance_start_step: Optional[int] = None
    disturbance_num_steps: int = 0
    translation_action_scale: float = 1.0
    # Start policy for the controlled disturbance:
    #   fixed: preserve the original behavior and use disturbance_start_step;
    #   random: draw one deterministic start per episode from the inclusive range below;
    #   translation_event: start causally when intended translation stays sufficiently large.
    disturbance_start_mode: str = "fixed"
    disturbance_random_start_min: int = 20
    disturbance_random_start_max: int = 100
    disturbance_event_min_step: int = 15
    disturbance_event_translation_norm: float = 0.02
    disturbance_event_consecutive_steps: int = 3

    # Optional experimental control for paired rollouts. When set, each episode gets a deterministic sequence of
    # pi0.5 diffusion noises derived from (sampling_noise_seed, task_id, episode_idx). This controls policy sampling only;
    # it does not use simulator state or privileged information.
    sampling_noise_seed: Optional[int] = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    trace_out_path: Optional[str] = None  # Optional JSONL path for action / state traces.
    # Optional compressed, step-aligned visual observations for diagnostics. Disabled by default to avoid bulk storage.
    visual_out_path: Optional[str] = None
    visual_stride: int = 1
    # Optional pi0.5 visual-token sidecar. This requests read-only 4x4 SigLIP/PaliGemma
    # features at replanning instants and stores them separately from the JSONL trace.
    visual_latent_out_path: Optional[str] = None
    task_id: Optional[int] = None  # Evaluate one task only; None evaluates the full suite.
    # Optional first-fault recovery snapshot. The template may contain
    # {task_id}, {episode_idx}, and {action_index}. It stores complete MuJoCo
    # state plus the prepared historical rollback plan, never policy answers.
    supervisor_fault_snapshot_path: Optional[str] = None
    # Optional checkpoint at the exact state where the second recovery attempt
    # is about to begin.  It removes the deterministic first-recovery prefix
    # from repeated rollback experiments without approximating object state.
    supervisor_second_recovery_snapshot_path: Optional[str] = None
    # Start directly from a previously captured fault state. This bypasses
    # the settling / pre-fault policy prefix for causal rollback ablations.
    supervisor_fault_snapshot_load_path: Optional[str] = None

    # Mechanism-decomposed supervisor. Disabled by default so historical
    # evaluation behavior remains unchanged.
    supervisor_enabled: bool = False
    supervisor_tools_path: str = "/root/gpufree-data/supervisor-tools-v1"
    supervisor_log_path: Optional[str] = None
    # Plumbing-only deterministic event injection. Never use in evaluation.
    supervisor_test_event: str = "none"
    supervisor_test_start_action: int = 10
    supervisor_test_num_steps: int = 3
    # Execute bounded hold/retreat bridge actions and adapt policy chunk length
    # from the supervisor's intervention directive. Disabled by default until
    # paired live-loop validation is complete.
    supervisor_intervention_enabled: bool = False
    # Generic, causal third-layer recovery context.  It is appended only after
    # an actual supervisor replan; it contains no task-specific answer or
    # simulator-only state.
    supervisor_recovery_prompt_enabled: bool = False
    supervisor_stall_retreat_enabled: bool = False
    supervisor_recovery_prompt: str = (
        "The previous motion made no progress. Re-observe the scene and retry "
        "the task from the current state. If the robot is already in contact, "
        "back away slightly before trying again."
    )
    supervisor_execution_mode: str = "placeholder"  # placeholder | frozen_four_state | frozen_four_state_visual_background | frozen_four_state_articulation
    supervisor_execution_model_dir: str = (
        "/root/gpufree-data/supervisor-tools-v1/outputs/remote_results/reliability_sequence_v1"
    )
    supervisor_execution_calibration_path: str = (
        "/root/gpufree-data/supervisor-tools-v1/outputs/remote_results/reliability_sequence_v1/"
        "conditional_reliability_calibration.json"
    )
    # Optional isolated articulation-perception sidecar.  The sidecar runs in
    # vision-env; this control process only receives fail-closed measurements.
    supervisor_articulation_sidecar_host: str = "127.0.0.1"
    supervisor_articulation_sidecar_port: int = 8766
    supervisor_articulation_sidecar_timeout_seconds: float = 1.5
    supervisor_articulation_calibration_id: str = "libero90-task0-close-area-v1"
    supervisor_articulation_control_enabled: bool = False
    # Component-ablation mode: keep every monitor observable, but grant only
    # the articulation relation monitor authority over fusion and recovery.
    supervisor_articulation_isolated_control: bool = False
    # Record fail-closed recovery-checkpoint evidence.  Simulator geometry is
    # evaluation-only and must be explicitly enabled; the default never
    # invents contact clearance from the absence of an alarm.
    supervisor_checkpoint_audit_enabled: bool = True
    supervisor_checkpoint_oracle_enabled: bool = False
    # Execute the prepared 6D join and reverse replay.  V1 requires the
    # evaluation-only geometry oracle; default remains observational.
    supervisor_rollback_enabled: bool = False
    # Experimental feedback mask.  Disabled by default: when false the mask
    # still runs and logs shadow state, but cannot alter candidate ranking.
    supervisor_escape_progress_mask_control_enabled: bool = False
    # Candidate-level cycle breaking and certified step pulses are shadow-only
    # by default.  Neither option weakens the swept collision gate.
    supervisor_escape_cycle_breaker_control_enabled: bool = False
    supervisor_safe_step_pulse_control_enabled: bool = False
    supervisor_rollback_step_budget: int = 320
    supervisor_rollback_replay_step_budget: int = 160
    # Soft budgets may be extended only while closed-loop target error keeps
    # converging. Collision gates remain authoritative. Disabled by default.
    supervisor_rollback_adaptive_budget_enabled: bool = False
    supervisor_rollback_hard_step_budget: int = 640
    supervisor_rollback_hard_replay_step_budget: int = 480
    supervisor_rollback_budget_extension_chunk: int = 80
    supervisor_rollback_mvp_near_trajectory_enabled: bool = False
    supervisor_rollback_position_tolerance_m: float = .004
    supervisor_rollback_orientation_tolerance_rad: float = .08
    # Exceptional REPLAY fallback: if every direct swept path is blocked,
    # allow only a few collision-certified self-clearance-improving microsteps.
    supervisor_replay_temporary_escape_max_steps: int = 3
    supervisor_replay_temporary_escape_step_budget: int = 24
    # Fault-specific paired ablation. Positive world Z gets only a temporary,
    # decaying preference during the second JOIN; it never bypasses collision
    # safety and does not become a cross-task default.
    supervisor_taskspace_upward_escape_control_enabled: bool = False
    supervisor_taskspace_upward_escape_minimum_recovery_depth: int = 2
    supervisor_taskspace_upward_escape_step_budget: int = 24
    supervisor_taskspace_upward_escape_minimum_predicted_dz_m: float = .0002
    supervisor_taskspace_upward_escape_minimum_axis_alignment: float = .75
    supervisor_taskspace_upward_escape_target_actual_dz_m: float = .008
    supervisor_taskspace_upward_escape_history_height_margin_m: float = .005
    supervisor_taskspace_upward_escape_maximum_target_regression_m: float = .001
    supervisor_taskspace_escape_protected_progress_axis: Optional[int] = None
    supervisor_taskspace_escape_protected_progress_desired_sign: int = 1
    supervisor_taskspace_escape_protected_progress_regression_weight: float = 1.25
    supervisor_taskspace_escape_progress_protection_decay_start: float = .55
    supervisor_taskspace_escape_progress_protection_release_completion: float = .85
    supervisor_taskspace_escape_progress_protection_reactivate_completion: float = .70
    supervisor_taskspace_escape_detour_control_enabled: bool = False
    supervisor_taskspace_escape_detour_stagnation_window: int = 24
    supervisor_taskspace_escape_detour_minimum_window_gain_m: float = .0005
    supervisor_taskspace_escape_detour_steps: int = 8
    supervisor_taskspace_escape_detour_probe_steps: int = 12
    supervisor_taskspace_escape_detour_minimum_probe_gain_m: float = .0003
    supervisor_taskspace_escape_detour_direction_cooldown_steps: int = 48
    # Use each recorded historical joint configuration as OSC's public
    # nullspace posture target during JOIN/REPLAY. Disabled for paired ablation.
    supervisor_joint_history_control_enabled: bool = False
    # Replace OSC_POSE with robosuite JOINT_POSITION for each recovery step.
    # This drives all seven arm joints directly toward recorded history; it is
    # mutually exclusive with the weaker OSC nullspace experiment above.
    supervisor_direct_joint_replay_enabled: bool = False
    supervisor_joint_replay_maximum_step_rad: float = .02
    supervisor_joint_replay_maximum_step_change_rad: float = .006
    supervisor_joint_replay_tolerance_rad: float = .025
    supervisor_joint_replay_brake_minimum_steps: int = 3
    supervisor_joint_replay_brake_maximum_steps: int = 12
    supervisor_joint_replay_brake_velocity_rad_s: float = .10
    # Reject a post-rollback policy chunk only when it closely repeats the
    # chunk active at the original failure, then retry from an earlier anchor.
    supervisor_replan_novelty_control_enabled: bool = False
    supervisor_replan_max_rollback_levels: int = 3
    # Require meaningful retreat before policy handoff.  Each failed changed
    # plan increases both physical execution and historical-depth minima.
    supervisor_rollback_minimum_replan_steps: int = 20
    supervisor_rollback_replan_step_increment: int = 15
    supervisor_rollback_minimum_history_depth: int = 6
    supervisor_rollback_history_depth_increment: int = 6
    supervisor_rollback_minimum_spatial_retreat_m: float = .10
    supervisor_rollback_spatial_retreat_increment_m: float = .05
    supervisor_rollback_maximum_spatial_retreat_m: float = .18
    supervisor_rollback_progress_only_replan_enabled: bool = False
    supervisor_rollback_progress_only_replan_minimum_recovery_depth: int = 2
    supervisor_rollback_progress_only_replan_minimum_steps: int = 20
    supervisor_rollback_progress_only_replan_minimum_spatial_retreat_m: float = .095
    # Once a materially changed plan is accepted, execution / visual experts
    # remain observable but cannot retake control for this many policy actions.
    supervisor_post_replan_novelty_only_steps: int = 25
    # Calibrated only on clean successful development traces.  The score is
    # an ensemble support/stability measure, not a probability.
    supervisor_checkpoint_minimum_reliability: float = 0.2588455491502583
    # Contact zero band; near-obstacle risk is handled by the swept candidate
    # gate and must not be conflated with actual contact.
    supervisor_checkpoint_contact_zero_band_m: float = 1e-5

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    valid_disturbance_modes = {"fixed", "random", "translation_event"}
    if args.disturbance_start_mode not in valid_disturbance_modes:
        raise ValueError(
            f"disturbance_start_mode must be one of {sorted(valid_disturbance_modes)}, "
            f"got {args.disturbance_start_mode!r}"
        )
    if args.disturbance_start_step is not None and args.disturbance_start_step < 0:
        raise ValueError("disturbance_start_step must be non-negative or None")
    if args.disturbance_num_steps < 0:
        raise ValueError("disturbance_num_steps must be non-negative")
    if not 0.0 <= args.translation_action_scale <= 1.0:
        raise ValueError("translation_action_scale must be between 0.0 and 1.0")
    if args.disturbance_random_start_min < 0:
        raise ValueError("disturbance_random_start_min must be non-negative")
    if args.disturbance_random_start_max < args.disturbance_random_start_min:
        raise ValueError("disturbance_random_start_max must be >= disturbance_random_start_min")
    if args.disturbance_event_min_step < 0:
        raise ValueError("disturbance_event_min_step must be non-negative")
    if args.disturbance_event_translation_norm < 0.0:
        raise ValueError("disturbance_event_translation_norm must be non-negative")
    if args.disturbance_event_consecutive_steps <= 0:
        raise ValueError("disturbance_event_consecutive_steps must be positive")
    if args.visual_stride <= 0:
        raise ValueError("visual_stride must be positive")
    valid_supervisor_test_events = {"none", "execution_mismatch", "policy_stall", "object_failure"}
    if args.supervisor_test_event not in valid_supervisor_test_events:
        raise ValueError(f"supervisor_test_event must be one of {sorted(valid_supervisor_test_events)}")
    if args.supervisor_test_event != "none" and not args.supervisor_enabled:
        raise ValueError("supervisor_test_event requires supervisor_enabled")
    if args.supervisor_test_start_action < 0 or args.supervisor_test_num_steps <= 0:
        raise ValueError("supervisor test start must be non-negative and duration positive")
    if args.supervisor_intervention_enabled and not args.supervisor_enabled:
        raise ValueError("supervisor_intervention_enabled requires supervisor_enabled")
    if args.supervisor_checkpoint_oracle_enabled and not args.supervisor_enabled:
        raise ValueError("supervisor_checkpoint_oracle_enabled requires supervisor_enabled")
    if args.supervisor_rollback_enabled and not (
            args.supervisor_intervention_enabled
            and args.supervisor_checkpoint_oracle_enabled):
        raise ValueError(
            "supervisor_rollback_enabled requires intervention and checkpoint oracle")
    if args.supervisor_rollback_step_budget <= 0:
        raise ValueError("supervisor_rollback_step_budget must be positive")
    if args.supervisor_rollback_replay_step_budget <= 0:
        raise ValueError("supervisor_rollback_replay_step_budget must be positive")
    if args.supervisor_rollback_hard_step_budget < args.supervisor_rollback_step_budget:
        raise ValueError("rollback hard join budget must be >= soft join budget")
    if (args.supervisor_rollback_hard_replay_step_budget
            < args.supervisor_rollback_replay_step_budget):
        raise ValueError("rollback hard replay budget must be >= soft replay budget")
    if args.supervisor_rollback_budget_extension_chunk <= 0:
        raise ValueError("rollback budget extension chunk must be positive")
    if args.supervisor_rollback_position_tolerance_m <= 0:
        raise ValueError("rollback position tolerance must be positive")
    if args.supervisor_rollback_orientation_tolerance_rad <= 0:
        raise ValueError("rollback orientation tolerance must be positive")
    if args.supervisor_replay_temporary_escape_max_steps < 0:
        raise ValueError("temporary replay escape maximum must be non-negative")
    if args.supervisor_replay_temporary_escape_step_budget < 0:
        raise ValueError("temporary replay escape budget must be non-negative")
    if args.supervisor_taskspace_upward_escape_minimum_recovery_depth < 1:
        raise ValueError("task-space upward minimum recovery depth must be positive")
    if args.supervisor_taskspace_upward_escape_step_budget < 1:
        raise ValueError("task-space upward step budget must be positive")
    if args.supervisor_taskspace_upward_escape_minimum_predicted_dz_m < 0:
        raise ValueError("task-space upward predicted dz must be non-negative")
    if not 0.0 <= args.supervisor_taskspace_upward_escape_minimum_axis_alignment <= 1.0:
        raise ValueError("task-space upward alignment must be between zero and one")
    if args.supervisor_taskspace_upward_escape_target_actual_dz_m <= 0:
        raise ValueError("task-space upward actual dz target must be positive")
    if args.supervisor_taskspace_upward_escape_history_height_margin_m < 0:
        raise ValueError("task-space upward history height margin must be non-negative")
    if args.supervisor_taskspace_upward_escape_maximum_target_regression_m < 0:
        raise ValueError("task-space upward target regression must be non-negative")
    if (args.supervisor_taskspace_escape_protected_progress_axis is not None
            and args.supervisor_taskspace_escape_protected_progress_axis not in (0, 1, 2)):
        raise ValueError("protected progress axis must be 0, 1, 2, or None")
    if args.supervisor_taskspace_escape_protected_progress_desired_sign not in (-1, 1):
        raise ValueError("protected progress desired sign must be -1 or 1")
    if args.supervisor_taskspace_escape_protected_progress_regression_weight < 0:
        raise ValueError("protected progress regression weight must be non-negative")
    if not (0.0 <= args.supervisor_taskspace_escape_progress_protection_decay_start
            < args.supervisor_taskspace_escape_progress_protection_release_completion
            <= 1.0):
        raise ValueError("progress protection decay/release must satisfy 0 <= start < release <= 1")
    if not (0.0 <= args.supervisor_taskspace_escape_progress_protection_reactivate_completion
            < args.supervisor_taskspace_escape_progress_protection_release_completion):
        raise ValueError("progress protection reactivation must be below release completion")
    if min(args.supervisor_taskspace_escape_detour_stagnation_window,
           args.supervisor_taskspace_escape_detour_steps,
           args.supervisor_taskspace_escape_detour_probe_steps,
           args.supervisor_taskspace_escape_detour_direction_cooldown_steps) < 1:
        raise ValueError("task-space detour windows and budgets must be positive")
    if min(args.supervisor_taskspace_escape_detour_minimum_window_gain_m,
           args.supervisor_taskspace_escape_detour_minimum_probe_gain_m) < 0:
        raise ValueError("task-space detour progress thresholds must be non-negative")
    if args.supervisor_replan_max_rollback_levels <= 0:
        raise ValueError("supervisor_replan_max_rollback_levels must be positive")
    if min(args.supervisor_rollback_minimum_replan_steps,
           args.supervisor_rollback_replan_step_increment,
           args.supervisor_rollback_minimum_history_depth,
           args.supervisor_rollback_history_depth_increment) < 0:
        raise ValueError("rollback replan depth settings must be non-negative")
    if min(args.supervisor_rollback_minimum_spatial_retreat_m,
           args.supervisor_rollback_spatial_retreat_increment_m,
           args.supervisor_rollback_maximum_spatial_retreat_m) < 0:
        raise ValueError("rollback spatial retreat settings must be non-negative")
    if args.supervisor_rollback_progress_only_replan_minimum_recovery_depth < 1:
        raise ValueError("progress-only replan recovery depth must be positive")
    if args.supervisor_rollback_progress_only_replan_minimum_steps < 0:
        raise ValueError("progress-only replan minimum steps must be non-negative")
    if args.supervisor_rollback_progress_only_replan_minimum_spatial_retreat_m < 0:
        raise ValueError("progress-only replan spatial retreat must be non-negative")
    if args.supervisor_post_replan_novelty_only_steps <= 0:
        raise ValueError("supervisor_post_replan_novelty_only_steps must be positive")
    if not 0.0 <= args.supervisor_checkpoint_minimum_reliability <= 1.0:
        raise ValueError("supervisor_checkpoint_minimum_reliability must be in [0, 1]")
    if args.supervisor_checkpoint_contact_zero_band_m < 0.0:
        raise ValueError("supervisor_checkpoint_contact_zero_band_m must be non-negative")
    if args.supervisor_execution_mode not in {
        "placeholder", "frozen_four_state", "frozen_four_state_visual_background",
        "frozen_four_state_articulation",
    }:
        raise ValueError(
            "supervisor_execution_mode must be placeholder, frozen_four_state, "
            "frozen_four_state_visual_background, or frozen_four_state_articulation"
        )
    if args.supervisor_execution_mode != "placeholder" and not args.supervisor_enabled:
        raise ValueError("non-placeholder supervisor_execution_mode requires supervisor_enabled")
    if not 0 < args.supervisor_articulation_sidecar_port < 65536:
        raise ValueError("supervisor_articulation_sidecar_port must be a valid TCP port")
    if args.supervisor_articulation_sidecar_timeout_seconds <= 0:
        raise ValueError("supervisor_articulation_sidecar_timeout_seconds must be positive")
    if (args.supervisor_articulation_control_enabled
            and args.supervisor_execution_mode != "frozen_four_state_articulation"):
        raise ValueError("articulation control requires frozen_four_state_articulation mode")
    if args.supervisor_articulation_control_enabled and not args.supervisor_intervention_enabled:
        raise ValueError("articulation control requires supervisor intervention")
    if (args.supervisor_direct_joint_replay_enabled
            and args.supervisor_joint_history_control_enabled):
        raise ValueError(
            "direct joint replay and OSC nullspace joint history are mutually exclusive")
    if args.supervisor_direct_joint_replay_enabled and not args.supervisor_rollback_enabled:
        raise ValueError("direct joint replay requires supervisor rollback")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    if args.visual_out_path is not None:
        pathlib.Path(args.visual_out_path).mkdir(parents=True, exist_ok=True)
    if args.visual_latent_out_path is not None:
        pathlib.Path(args.visual_latent_out_path).mkdir(parents=True, exist_ok=True)
    trace_file = None
    if args.trace_out_path is not None:
        trace_path = pathlib.Path(args.trace_out_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", encoding="utf-8")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    supervisor = None
    checkpoint_recorder = None
    checkpoint_phase_tracker = None
    recovery_preparer = None
    recovery_port = None
    rollback_cursor = None
    escape_progress_mask = None
    escape_cycle_breaker = None
    replan_novelty_gate = None
    MuJoCoSweptCollisionOracle = None
    BoundedJointPositionServo = None
    RobosuiteJointControllerSwitch = None
    # The policy deadline is shared by baseline and supervised runs so paired
    # experiments differ only when a completed recovery explicitly resets it.
    sys.path.insert(0, args.supervisor_tools_path)
    from vla_supervisor.execution_budget import RecoveryAwarePolicyBudget
    if args.supervisor_enabled:
        from vla_supervisor.events import EventType
        from vla_supervisor.factory import (
            create_frozen_execution_with_articulation_progress,
            create_frozen_execution_with_visual_consequence,
            create_frozen_four_state_execution,
            create_safe_default,
        )
        from vla_supervisor.intervention import InterventionPlanner
        from vla_supervisor.kinematic_safety import measure_public_kinematic_safety
        from vla_supervisor.visual_consequence import BackgroundConsequenceEvidenceScorer
        from vla_supervisor.monitors import DeterministicTestMonitor
        from vla_supervisor.recovery import RecoveryState
        from vla_supervisor.checkpoint_runtime import (
            CheckpointEvidence, ConservativePrecontactPhaseTracker,
            OnlineCheckpointRecorder, OnlineRecoveryPreparer,
            contact_clear_from_signed_distances, estimate_camera_health,
            normal_response_reliability,
        )
        from vla_supervisor.checkpoint_recovery import (
            CheckpointAdmissionConfig, CheckpointBuffer)
        from vla_supervisor.online_rollback import (
            AxisEscapeDetourController, EscapeCycleBreaker, EscapeProgressMask,
            OnlineRollbackCursor, RollbackState,
            escape_controls_allowed, phase_scoped_candidate_summaries,
            generate_rollback_candidate_actions,
            open_history_replay_clearance_safe, rollback_collision_stage,
            rollback_clearance_safe,
            rollback_pairwise_clearance_safe, rollback_progress_weight,
            replay_corridor_floor_m, select_replay_direct_candidate_index,
            select_temporary_replay_escape_index,
            temporary_escape_direction_key,
            temporary_escape_progress_weight,
            protected_progress_weight,
            required_axis_escape_displacement,
            select_axis_priority_taskspace_candidate_index,
            select_lateral_detour_candidate_index,
            progress_only_replan_ready,
            update_progress_protection_release,
            select_rollback_candidate_index, select_safe_step_pulse_index)
        from vla_supervisor.recovery_handoff import RecoveryExecutorPort, RecoveryHandoffState
        from vla_supervisor.replan_novelty import ReplanNoveltyGate
        from fault_snapshot import restore_fault_snapshot, write_fault_snapshot
        if args.supervisor_checkpoint_oracle_enabled:
            from vla_supervisor.mujoco_collision_oracle import (
                MuJoCoSweptCollisionOracle, predict_joint_delta_from_pose)
        if args.supervisor_direct_joint_replay_enabled:
            from joint_space_rollback import (
                BoundedJointPositionServo, RobosuiteJointControllerSwitch,
                historical_joint_contact_exit_safe)
            from task_space_candidate_review import (
                EndEffectorTrajectory, trajectory_distance_m)

        supervisor_log_path = args.supervisor_log_path
        if supervisor_log_path is None:
            if args.trace_out_path is None:
                raise ValueError("supervisor_enabled requires supervisor_log_path or trace_out_path")
            supervisor_log_path = f"{args.trace_out_path}.supervisor.jsonl"
        if args.supervisor_execution_mode in {
            "frozen_four_state", "frozen_four_state_visual_background",
            "frozen_four_state_articulation",
        }:
            model_paths = sorted(pathlib.Path(args.supervisor_execution_model_dir).glob(
                "conditional_relational_pilot_seed*.pt"
            ))
            if len(model_paths) != 5:
                raise FileNotFoundError(
                    f"Expected five frozen conditional ensemble members, found {len(model_paths)}"
                )
            calibration_path = pathlib.Path(args.supervisor_execution_calibration_path)
            if not calibration_path.is_file():
                raise FileNotFoundError(calibration_path)
            if args.supervisor_execution_mode == "frozen_four_state_articulation":
                from vla_supervisor.articulation_progress import ArticulationProgressConfig
                from vla_supervisor.articulation_provider import (
                    ArticulationRelationContext, ValidatedArticulationProvider)
                from vla_supervisor.articulation_sidecar import ArticulationSidecarClient
                sidecar = ArticulationSidecarClient(
                    args.supervisor_articulation_sidecar_host,
                    args.supervisor_articulation_sidecar_port,
                    args.supervisor_articulation_sidecar_timeout_seconds,
                )
                provider = ValidatedArticulationProvider(
                    sidecar,
                    ArticulationRelationContext(
                        relation_id="top_drawer_closed", predicate="Close",
                        calibration_id=args.supervisor_articulation_calibration_id,
                        provenance="isolated_dino_sam2_sidecar",
                    ),
                )
                supervisor = create_frozen_execution_with_articulation_progress(
                    model_paths, calibration_path, provider,
                    ArticulationProgressConfig(
                        calibration_id=args.supervisor_articulation_calibration_id),
                    articulation_control_enabled=(
                        args.supervisor_articulation_control_enabled),
                    isolated_articulation_control=(
                        args.supervisor_articulation_isolated_control),
                    log_path=supervisor_log_path,
                )
                supervisor.component_status["articulation_sidecar"] = (
                    f"{args.supervisor_articulation_sidecar_host}:"
                    f"{args.supervisor_articulation_sidecar_port}"
                )
            elif args.supervisor_execution_mode == "frozen_four_state_visual_background":
                supervisor = create_frozen_execution_with_visual_consequence(
                    model_paths, calibration_path,
                    BackgroundConsequenceEvidenceScorer(),
                    log_path=supervisor_log_path,
                )
                supervisor.component_status["visual_candidate_provider"] = (
                    "explicit_none_background_health_only"
                )
            else:
                supervisor = create_frozen_four_state_execution(
                    model_paths, calibration_path, log_path=supervisor_log_path,
                )
        else:
            supervisor = create_safe_default(log_path=supervisor_log_path)
        # In monitor-only experiments the full scorer still runs and logs its
        # hypothetical directive, but it must be causally transparent to the
        # action queue and policy inference schedule.
        supervisor.control_enabled = args.supervisor_intervention_enabled
        if args.supervisor_rollback_enabled:
            recovery_port = RecoveryExecutorPort()
            rollback_cursor = OnlineRollbackCursor(
                maximum_steps=args.supervisor_rollback_step_budget,
                maximum_replay_steps=args.supervisor_rollback_replay_step_budget,
                adaptive_budget_enabled=(
                    args.supervisor_rollback_adaptive_budget_enabled),
                maximum_hard_steps=args.supervisor_rollback_hard_step_budget,
                maximum_hard_replay_steps=(
                    args.supervisor_rollback_hard_replay_step_budget),
                budget_extension_chunk=(
                    args.supervisor_rollback_budget_extension_chunk),
                replan_when_join_reached=(
                    args.supervisor_rollback_mvp_near_trajectory_enabled),
                position_tolerance_m=(
                    args.supervisor_rollback_position_tolerance_m),
                orientation_tolerance_rad=(
                    args.supervisor_rollback_orientation_tolerance_rad),
                joint_tolerance_rad=(
                    args.supervisor_joint_replay_tolerance_rad))
            escape_progress_mask = EscapeProgressMask()
            escape_cycle_breaker = EscapeCycleBreaker()
            supervisor.recovery_executor = recovery_port
            replan_novelty_gate = ReplanNoveltyGate()
        if args.supervisor_stall_retreat_enabled:
            supervisor.intervention = InterventionPlanner(dataclasses.replace(
                supervisor.intervention.config, retreat_on_policy_stall=True,
            ))
        if args.supervisor_test_event != "none":
            event_type = EventType(args.supervisor_test_event)
            supervisor.step_monitors = [DeterministicTestMonitor(
                event_type, args.supervisor_test_start_action, args.supervisor_test_num_steps
            )]
            supervisor.component_status["test_injection"] = args.supervisor_test_event
        logging.info("Supervisor enabled: %s", supervisor.component_status)
        if args.supervisor_checkpoint_audit_enabled:
            checkpoint_recorder = OnlineCheckpointRecorder(CheckpointBuffer(
                CheckpointAdmissionConfig(
                    minimum_reliability=args.supervisor_checkpoint_minimum_reliability)))
            checkpoint_phase_tracker = ConservativePrecontactPhaseTracker()
            recovery_preparer = OnlineRecoveryPreparer(
                allow_recent_unknown_reliability=(
                    args.supervisor_rollback_mvp_near_trajectory_enabled))

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids = range(num_tasks_in_suite) if args.task_id is None else [args.task_id]
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_indices = range(args.num_trials_per_task)
        if args.episode_indices is not None:
            episode_indices = tuple(
                int(value.strip())
                for value in args.episode_indices.split(",")
                if value.strip())
            if not episode_indices:
                raise ValueError("episode_indices must contain at least one index")
            if len(set(episode_indices)) != len(episode_indices):
                raise ValueError("episode_indices must not contain duplicates")
            if tuple(sorted(episode_indices)) != episode_indices:
                raise ValueError("episode_indices must be strictly increasing")
            if min(episode_indices) < 0 or max(episode_indices) >= len(initial_states):
                raise ValueError(
                    f"episode_indices must be within [0, {len(initial_states) - 1}]")
        previous_reset_index = -1
        for episode_idx in tqdm.tqdm(episode_indices):
            logging.info(f"\nTask: {task_description}")

            # Preserve the environment RNG/reset sequence of a full batch even
            # when evaluation schedules only selected episodes.  Merely using
            # ``initial_states[episode_idx]`` is insufficient: LIBERO reset
            # consumes seeded environment randomness before set_init_state.
            # Skipped resets are cheap and prevent episode ``2`` in a ``0,2``
            # run from silently becoming the second rather than third reset.
            reset_env_preserving_episode_index(
                env, episode_idx=episode_idx,
                previous_reset_index=previous_reset_index)
            previous_reset_index = episode_idx
            # LIBERO can rebuild ``env.sim`` during reset.  Bind the
            # evaluation-only geometry oracle afterwards so it never retains
            # the released simulator from environment construction.
            checkpoint_collision_oracle = None
            if args.supervisor_checkpoint_oracle_enabled:
                checkpoint_collision_oracle = MuJoCoSweptCollisionOracle(
                    env.sim, env.robots[0], interpolation_samples=2)
            action_plan = collections.deque()
            if supervisor is not None:
                supervisor.reset_episode()
            if rollback_cursor is not None:
                rollback_cursor.reset()
                escape_progress_mask.reset()
                escape_cycle_breaker.reset()
            if checkpoint_recorder is not None:
                checkpoint_recorder.reset()
                checkpoint_phase_tracker.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            loaded_fault_snapshot = None
            if args.supervisor_fault_snapshot_load_path is not None:
                if not args.supervisor_rollback_enabled:
                    raise ValueError("fault snapshot load requires rollback_enabled")
                snapshot_path = args.supervisor_fault_snapshot_load_path.format(
                    task_id=task_id, episode_idx=episode_idx)
                snapshot_payload, snapshot_preparation = restore_fault_snapshot(
                    snapshot_path, sim=env.sim)
                _synchronize_robot_controller_to_current_state(env)
                # robosuite observables are cached independently from MuJoCo.
                # Without a forced refresh, the first replay decision sees
                # the episode's initial joint pose while physics is already at
                # the fault pose, producing a spurious full-arm jump.
                env._update_observables(force=True)
                snapshot_metadata = snapshot_payload.get("metadata", {})
                if (int(snapshot_metadata.get("task_id", task_id)) != task_id
                        or int(snapshot_metadata.get("episode_idx", episode_idx))
                        != episode_idx):
                    raise ValueError("fault snapshot task / episode metadata mismatch")
                obs = env.env._get_observations()
                loaded_fault_snapshot = (
                    snapshot_payload, snapshot_preparation)
            joint_replay_servo = None
            joint_controller_switch = None
            joint_replay_target_action_index = None
            joint_replay_active_attempt = None
            joint_replay_brake_steps = 0
            if args.supervisor_direct_joint_replay_enabled:
                joint_replay_servo = BoundedJointPositionServo(
                    maximum_step_rad=(
                        args.supervisor_joint_replay_maximum_step_rad),
                    maximum_step_change_rad=(
                        args.supervisor_joint_replay_maximum_step_change_rad),
                    tolerance_rad=args.supervisor_joint_replay_tolerance_rad)
                joint_controller_switch = RobosuiteJointControllerSwitch(
                    env, env.robots[0])
                if loaded_fault_snapshot is not None:
                    joint_controller_switch.synchronize_to_current_state()

            # Setup
            t = (args.num_steps_wait if loaded_fault_snapshot is not None else 0)
            replay_images = []
            inference_calls = 0
            chunk_id = -1
            action_index = -1
            previous_action = None
            disturbed_steps = 0
            rollback_executed_steps = 0
            rollback_attempt_executed_steps = 0
            rollback_terminal_reason = None
            last_policy_chunk = None
            failed_policy_chunk = None
            recovery_failure_context = None
            rollback_level = 0
            recovery_depth_attempts = 0
            fault_snapshot_written = False
            replay_temporary_escape_steps = 0
            replay_temporary_escape_active = False
            replay_temporary_escape_burst_steps = 0
            replay_temporary_escape_direction = None
            replay_temporary_escape_direction_baseline_m = None
            replay_temporary_escape_blocked_directions = set()
            taskspace_upward_escape_steps = 0
            taskspace_upward_escape_baseline_z = None
            taskspace_upward_escape_required_dz_m = None
            taskspace_progress_protection_released = False
            taskspace_escape_detour = AxisEscapeDetourController(
                sign=1,
                stagnation_window=(
                    args.supervisor_taskspace_escape_detour_stagnation_window),
                minimum_window_gain_m=(
                    args.supervisor_taskspace_escape_detour_minimum_window_gain_m),
                detour_steps=args.supervisor_taskspace_escape_detour_steps,
                probe_steps=args.supervisor_taskspace_escape_detour_probe_steps,
                minimum_probe_gain_m=(
                    args.supervisor_taskspace_escape_detour_minimum_probe_gain_m),
                direction_cooldown_steps=(
                    args.supervisor_taskspace_escape_detour_direction_cooldown_steps))
            # Policy-task budget is independent from bounded rollback motion.
            # A completed rollback establishes a new initial condition, so the
            # policy receives a fresh full task window instead of inheriting
            # the nearly exhausted pre-failure deadline.
            policy_budget = RecoveryAwarePolicyBudget(max_steps)
            if loaded_fault_snapshot is not None:
                snapshot_payload, snapshot_preparation = loaded_fault_snapshot
                snapshot_metadata = snapshot_payload.get("metadata", {})
                snapshot_depth_level = int(
                    snapshot_metadata.get("recovery_depth_attempt", 0))
                snapshot_minimum_steps = (
                    args.supervisor_rollback_minimum_replan_steps
                    + snapshot_depth_level
                    * args.supervisor_rollback_replan_step_increment)
                snapshot_minimum_history_depth = (
                    args.supervisor_rollback_minimum_history_depth
                    + snapshot_depth_level
                    * args.supervisor_rollback_history_depth_increment)
                snapshot_minimum_spatial_retreat_m = min(
                    args.supervisor_rollback_maximum_spatial_retreat_m,
                    args.supervisor_rollback_minimum_spatial_retreat_m
                    + snapshot_depth_level
                    * args.supervisor_rollback_spatial_retreat_increment_m)
                snapshot_action_index = int(snapshot_metadata.get("action_index", -1))
                action_index = snapshot_action_index
                previous_action = np.zeros(7, dtype=float)
                previous_action[6] = float(snapshot_payload["gripper_action"])
                supervisor.previous_action = previous_action.copy()
                requested = recovery_port.request_recovery(
                    reason="fault_snapshot_replay", confidence=1.0,
                    action_index=snapshot_action_index, history=())
                claimed = recovery_port.claim() if requested else None
                if (claimed is None or not rollback_cursor.begin(
                        snapshot_preparation,
                        minimum_replan_steps=snapshot_minimum_steps,
                        minimum_replan_history_depth=(
                            snapshot_minimum_history_depth),
                        minimum_replan_spatial_retreat_m=(
                            snapshot_minimum_spatial_retreat_m),
                        failure_eef_pos=snapshot_payload["failure_eef_pos"])):
                    raise RuntimeError("failed to activate rollback from fault snapshot")
                recovery_depth_attempts = snapshot_depth_level + 1
                rollback_attempt_executed_steps = 0
                fault_snapshot_written = True
            supervisor_alarm_counts = collections.Counter()
            supervisor_intervention_counts = collections.Counter()
            visual_action_indices = []
            visual_agent_images = []
            visual_wrist_images = []
            visual_agent_camera_to_world = []
            visual_wrist_camera_to_world = []
            visual_eef_positions = []
            visual_eef_quaternions = []
            visual_gripper_qpos = []
            latent_chunk_ids = []
            latent_action_start_indices = []
            latent_base = []
            latent_wrist = []
            sampling_noise_rng = (
                np.random.default_rng(np.random.SeedSequence([args.sampling_noise_seed, task_id, episode_idx]))
                if args.sampling_noise_seed is not None
                else None
            )
            scheduled_disturbance_start = args.disturbance_start_step
            disturbance_trigger_reason = None
            event_consecutive_steps = 0
            if args.disturbance_start_mode == "random":
                disturbance_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, task_id, episode_idx, 0xD157])
                )
                scheduled_disturbance_start = int(
                    disturbance_rng.integers(
                        args.disturbance_random_start_min,
                        args.disturbance_random_start_max + 1,
                    )
                )
                disturbance_trigger_reason = "deterministic_random_schedule"
            elif args.disturbance_start_mode == "translation_event":
                scheduled_disturbance_start = None

            if trace_file is not None:
                _write_trace(
                    trace_file,
                    {
                        "event": "episode_start",
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "seed": args.seed,
                        "sampling_noise_seed": args.sampling_noise_seed,
                        "disturbance_start_mode": args.disturbance_start_mode,
                        "scheduled_disturbance_start": scheduled_disturbance_start,
                        "disturbance_num_steps": args.disturbance_num_steps,
                        "translation_action_scale": args.translation_action_scale,
                    },
                )

            logging.info(f"Starting episode {task_episodes+1}...")
            rollback_total_step_budget = (
                args.supervisor_rollback_hard_step_budget
                + args.supervisor_rollback_hard_replay_step_budget
                if args.supervisor_rollback_adaptive_budget_enabled else
                args.supervisor_rollback_step_budget
                + args.supervisor_rollback_replay_step_budget)
            while (
                policy_budget.should_continue(
                    settling=t < args.num_steps_wait,
                    external_recovery_active=(
                        recovery_port is not None
                        and recovery_port.state is RecoveryHandoffState.ACTIVE),
                )
            ):
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # SAFE_STOPPED is terminal for the current episode.  Check
                    # it before image preprocessing or policy inference so a
                    # fresh chunk cannot accidentally bypass an exhausted
                    # recovery budget on the following loop iteration.
                    if (supervisor is not None and args.supervisor_intervention_enabled
                            and supervisor.recovery.state is RecoveryState.SAFE_STOPPED):
                        logging.warning(
                            "Supervisor recovery budget exhausted; safe stopping before action %d",
                            action_index + 1,
                        )
                        if trace_file is not None:
                            _write_trace(trace_file, {
                                "event": "supervisor_control", "task_id": task_id,
                                "episode_idx": episode_idx, "t": t - args.num_steps_wait,
                                "action_index": action_index + 1, "phase": "terminal_safe_stop",
                                "recovery_state": supervisor.recovery.state.value,
                            })
                        break

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Complex rollback owns the queue one guarded microstep at
                    # a time.  Every proposal is checked against the current
                    # MuJoCo swept geometry before it reaches the ordinary
                    # instruction guard.  This oracle is evaluation-only.
                    rollback_selected_audit = None
                    rollback_joint_servo_step = None
                    rollback_selected_joint_action = None
                    joint_replay_braking_step = False
                    rollback_video_target = None
                    if (rollback_cursor is not None and recovery_port is not None
                            and recovery_port.state is RecoveryHandoffState.ACTIVE
                            and not supervisor.chunk):
                        gripper_action = (
                            float(supervisor.previous_action[6])
                            if supervisor.previous_action is not None else -1.0)
                        proposal = rollback_cursor.propose(
                            eef_pos=obs["robot0_eef_pos"],
                            eef_rotation=obs["robot0_eef_quat"],
                            gripper_action=gripper_action,
                            joint_pos=obs["robot0_joint_pos"],
                            joint_position_mode=(
                                args.supervisor_direct_joint_replay_enabled))
                        rollback_video_target = proposal.target_action_index
                        progress_only_replan = (
                            args.supervisor_rollback_progress_only_replan_enabled
                            and progress_only_replan_ready(
                                rollback_state=proposal.state,
                                recovery_depth_attempts=recovery_depth_attempts,
                                executed_steps=rollback_attempt_executed_steps,
                                spatial_retreat_m=rollback_cursor.spatial_retreat_m,
                                minimum_recovery_depth=(
                                    args.supervisor_rollback_progress_only_replan_minimum_recovery_depth),
                                minimum_steps=(
                                    args.supervisor_rollback_progress_only_replan_minimum_steps),
                                minimum_spatial_retreat_m=(
                                    args.supervisor_rollback_progress_only_replan_minimum_spatial_retreat_m)))
                        if progress_only_replan:
                            if args.supervisor_joint_history_control_enabled:
                                env.robots[0].controller.update_initial_joints(
                                    np.asarray(obs["robot0_joint_pos"], dtype=float))
                            taskspace_escape_detour.reset()
                            recovery_port.recovery_completed(safe=True, needs_replan=True)
                            policy_budget.reset_after_completed_recovery()
                            if trace_file is not None:
                                _write_trace(trace_file, {
                                    "event": "policy_budget_reset",
                                    "task_id": task_id,
                                    "episode_idx": episode_idx,
                                    "action_index": action_index + 1,
                                    "reason": "progress_only_rollback_handoff",
                                    "actual_spatial_retreat_m": rollback_cursor.spatial_retreat_m,
                                    "rollback_attempt_executed_steps": rollback_attempt_executed_steps,
                                    "recovery_depth_attempts": recovery_depth_attempts,
                                    "new_policy_budget_steps": max_steps,
                                    "policy_budget_reset_count": policy_budget.reset_count,
                                })
                        elif (proposal.action is not None
                                and rollback_attempt_executed_steps >=
                                rollback_total_step_budget):
                            # Allow a read-only reached-state adjudication after
                            # the final action, but never execute action N+1.
                            rollback_cursor.reject(
                                "total_rollback_step_budget_exhausted")
                            rollback_terminal_reason = (
                                "total_rollback_step_budget_exhausted")
                            recovery_port.recovery_completed(safe=False)
                        elif proposal.state == RollbackState.REPLAN_PENDING.value:
                            if args.supervisor_joint_history_control_enabled:
                                env.robots[0].controller.update_initial_joints(
                                    np.asarray(obs["robot0_joint_pos"], dtype=float))
                            recovery_port.recovery_completed(safe=True, needs_replan=True)
                            policy_budget.reset_after_completed_recovery()
                            if trace_file is not None:
                                _write_trace(trace_file, {
                                    "event": "policy_budget_reset",
                                    "task_id": task_id,
                                    "episode_idx": episode_idx,
                                    "action_index": action_index + 1,
                                    "reason": "complex_rollback_completed",
                                    "new_policy_budget_steps": max_steps,
                                    "policy_budget_reset_count": policy_budget.reset_count,
                                })
                        elif proposal.state == RollbackState.SAFE_STOP.value:
                            if args.supervisor_joint_history_control_enabled:
                                env.robots[0].controller.update_initial_joints(
                                    np.asarray(obs["robot0_joint_pos"], dtype=float))
                            rollback_terminal_reason = proposal.reason
                            recovery_port.recovery_completed(safe=False)
                        elif proposal.action is not None:
                            if proposal.state != RollbackState.REPLAY.value:
                                if replay_temporary_escape_active:
                                    rollback_cursor.reset_tracking_controller()
                                replay_temporary_escape_steps = 0
                                replay_temporary_escape_active = False
                                replay_temporary_escape_burst_steps = 0
                                replay_temporary_escape_direction = None
                                replay_temporary_escape_direction_baseline_m = None
                                replay_temporary_escape_blocked_directions.clear()
                            escape_controls_phase_active = escape_controls_allowed(
                                proposal.state)
                            if escape_controls_phase_active:
                                escape_progress_mask.begin_decision()
                                escape_cycle_breaker.begin_decision()
                            else:
                                # Escape heuristics are temporary JOIN-only
                                # state. Reverse replay must never inherit a
                                # tabu direction, safety-weight mask or pulse.
                                escape_progress_mask.reset()
                                escape_cycle_breaker.reset()
                                if args.supervisor_taskspace_escape_detour_control_enabled:
                                    taskspace_escape_detour.reset()
                            rollback_target_eef = rollback_cursor.active_target_eef
                            rollback_target_joint = rollback_cursor.active_target_joint
                            reference_candidate = np.asarray(proposal.action, dtype=float)
                            current_joint = np.asarray(obs["robot0_joint_pos"], dtype=float)
                            if args.supervisor_direct_joint_replay_enabled:
                                if rollback_target_joint is None:
                                    raise RuntimeError(
                                        "direct joint replay has no historical joint target")
                                if joint_replay_active_attempt != recovery_depth_attempts:
                                    joint_replay_active_attempt = recovery_depth_attempts
                                    joint_replay_brake_steps = 0
                                    joint_replay_servo.reset()
                                if (joint_replay_target_action_index
                                        != proposal.target_action_index):
                                    joint_replay_servo.reset()
                                    joint_replay_target_action_index = (
                                        proposal.target_action_index)
                                maximum_joint_velocity = float(np.max(np.abs(
                                    np.asarray(obs["robot0_joint_vel"], dtype=float))))
                                joint_replay_braking_step = (
                                    joint_replay_brake_steps <
                                    args.supervisor_joint_replay_brake_minimum_steps
                                    or maximum_joint_velocity >
                                    args.supervisor_joint_replay_brake_velocity_rad_s)
                                if (joint_replay_braking_step
                                        and joint_replay_brake_steps >=
                                        args.supervisor_joint_replay_brake_maximum_steps):
                                    rollback_cursor.reject(
                                        "joint_replay_braking_budget_exhausted")
                                    rollback_terminal_reason = (
                                        "joint_replay_braking_budget_exhausted")
                                    recovery_port.recovery_completed(safe=False)
                                    continue
                                rollback_joint_servo_step = joint_replay_servo.next_step(
                                    current_joint,
                                    (current_joint if joint_replay_braking_step
                                     else rollback_target_joint),
                                    controller_output_rad=(
                                        joint_controller_switch.joint_output_rad))
                            # Keep a model-error buffer around clear states.
                            # Within that buffer (or in contact), only strict
                            # clearance improvement is admissible.
                            contact_zero_tolerance_m = 1e-5
                            clearance_buffer_m = .005
                            evaluated_candidates = []
                            for candidate_spec in generate_rollback_candidate_actions(
                                    reference_candidate):
                                candidate = np.asarray(candidate_spec.action, dtype=float)
                                # A direct candidate follows the previously
                                # observed historical corridor.  It may enter
                                # the conservative 5 mm ranking buffer while
                                # remaining above a 2 mm hard positive floor;
                                # escape-axis candidates receive no exception.
                                historical_corridor_floor_m = (
                                    (replay_corridor_floor_m()
                                     if proposal.state == RollbackState.REPLAY.value
                                     else .002)
                                    if candidate_spec.kind == "direct" else None)
                                translation = .05 * np.clip(candidate[:3], -1.0, 1.0)
                                rotation = .5 * np.clip(candidate[3:6], -1.0, 1.0)
                                if args.supervisor_direct_joint_replay_enabled:
                                    if joint_replay_braking_step:
                                        predicted_joint = current_joint.copy()
                                    elif candidate_spec.kind == "direct":
                                        full_target = np.asarray(
                                            rollback_joint_servo_step.target_joint,
                                            dtype=float)
                                        predicted_joint = current_joint + (
                                            candidate_spec.scale
                                            * (full_target - current_joint))
                                    else:
                                        predicted_joint = current_joint + predict_joint_delta_from_pose(
                                            env.sim, env.robots[0], translation, rotation,
                                            target_joint=None)
                                    candidate_joint_action = np.clip(
                                        (predicted_joint - current_joint)
                                        / joint_controller_switch.joint_output_rad,
                                        -1.0, 1.0)
                                else:
                                    predicted_joint = current_joint + predict_joint_delta_from_pose(
                                        env.sim, env.robots[0], translation, rotation,
                                        target_joint=(
                                            rollback_target_joint
                                            if args.supervisor_joint_history_control_enabled
                                            else None))
                                swept = checkpoint_collision_oracle.evaluate(predicted_joint)
                                task_space_trajectory = EndEffectorTrajectory(
                                    swept.eef_positions_m)
                                task_space_points = np.asarray(
                                    task_space_trajectory.positions_m, dtype=float)
                                task_space_displacement = (
                                    task_space_points[-1] - task_space_points[0])
                                task_space_target_progress_m = None
                                if rollback_target_eef is not None:
                                    task_space_target = np.asarray(
                                        rollback_target_eef, dtype=float)
                                    task_space_target_progress_m = float(
                                        np.linalg.norm(
                                            task_space_target - task_space_points[0])
                                        - np.linalg.norm(
                                            task_space_target - task_space_points[-1]))
                                clearance_pairs = (
                                    (swept.baseline_self_clearance_m,
                                     swept.final_self_clearance_m),
                                    (swept.baseline_environment_clearance_m,
                                     swept.final_environment_clearance_m),
                                )
                                # Self geometry and environment geometry have
                                # different semantics. A calibrated historical
                                # mesh tolerance must never weaken the current
                                # world / obstacle collision gate.
                                self_clearance_safe = rollback_clearance_safe(
                                    (clearance_pairs[0],),
                                    zero_tolerance_m=contact_zero_tolerance_m,
                                    clearance_buffer_m=clearance_buffer_m,
                                    historical_corridor_floor_m=(
                                        historical_corridor_floor_m))
                                environment_clearance_safe = rollback_clearance_safe(
                                    (clearance_pairs[1],),
                                    zero_tolerance_m=contact_zero_tolerance_m,
                                    clearance_buffer_m=clearance_buffer_m,
                                    historical_corridor_floor_m=None)
                                self_pairwise_clearance_safe = (
                                    rollback_pairwise_clearance_safe(
                                        swept.baseline_self_pair_clearances,
                                        swept.final_self_pair_clearances,
                                        zero_tolerance_m=contact_zero_tolerance_m,
                                        clearance_buffer_m=clearance_buffer_m,
                                        historical_corridor_floor_m=(
                                            historical_corridor_floor_m),
                                        escaping_existing_contact=(
                                            min(clearance_pairs[0][0],
                                                clearance_pairs[1][0])
                                            <= contact_zero_tolerance_m)))
                                environment_pairwise_clearance_safe = (
                                    rollback_pairwise_clearance_safe(
                                        swept.baseline_environment_pair_clearances,
                                        swept.final_environment_pair_clearances,
                                        zero_tolerance_m=contact_zero_tolerance_m,
                                        clearance_buffer_m=clearance_buffer_m,
                                        historical_corridor_floor_m=None,
                                        escaping_existing_contact=(
                                            min(clearance_pairs[0][0],
                                                clearance_pairs[1][0])
                                            <= contact_zero_tolerance_m)))
                                pairwise_clearance_safe = (
                                    self_pairwise_clearance_safe
                                    and environment_pairwise_clearance_safe)
                                clearance_safe = (
                                    self_clearance_safe
                                    and environment_clearance_safe
                                    and pairwise_clearance_safe)
                                collision_stage = rollback_collision_stage(
                                    rollback_state=proposal.state,
                                    candidate_kind=candidate_spec.kind,
                                    spatial_retreat_m=rollback_cursor.spatial_retreat_m,
                                    baseline_environment_clearance_m=(
                                        swept.baseline_environment_clearance_m),
                                    final_environment_clearance_m=(
                                        swept.final_environment_clearance_m))
                                if collision_stage == "open_history_replay":
                                    clearance_safe = open_history_replay_clearance_safe(
                                        baseline_self_clearance_m=(
                                            swept.baseline_self_clearance_m),
                                        final_self_clearance_m=(
                                            swept.final_self_clearance_m),
                                        baseline_environment_clearance_m=(
                                            swept.baseline_environment_clearance_m),
                                        final_environment_clearance_m=(
                                            swept.final_environment_clearance_m),
                                        final_self_pair_clearances=(
                                            swept.final_self_pair_clearances),
                                        final_environment_pair_clearances=(
                                            swept.final_environment_pair_clearances),
                                        historical_self_floor_m=(
                                            historical_corridor_floor_m))
                                if (args.supervisor_direct_joint_replay_enabled
                                        and joint_replay_braking_step
                                        and candidate_spec.kind == "direct"):
                                    # Zero-position-delta JOINT_POSITION applies
                                    # damping to inherited velocity. Geometry
                                    # prediction is unchanged; actual response
                                    # is audited immediately after the step.
                                    clearance_safe = True
                                historical_joint_contact_exit = False
                                if (args.supervisor_direct_joint_replay_enabled
                                        and candidate_spec.kind == "direct"):
                                    historical_joint_contact_exit = (
                                        historical_joint_contact_exit_safe(
                                            baseline_self_clearance_m=(
                                                swept.baseline_self_clearance_m),
                                            final_self_clearance_m=(
                                                swept.final_self_clearance_m),
                                            baseline_environment_clearance_m=(
                                                swept.baseline_environment_clearance_m),
                                            final_environment_clearance_m=(
                                                swept.final_environment_clearance_m),
                                            baseline_environment_pair_clearances=(
                                                swept.baseline_environment_pair_clearances),
                                            final_environment_pair_clearances=(
                                                swept.final_environment_pair_clearances)))
                                    clearance_safe = (
                                        clearance_safe
                                        or historical_joint_contact_exit)
                                gains = tuple(
                                    float(final - baseline)
                                    for baseline, final in clearance_pairs
                                    if baseline is not None and final is not None)
                                worst_gain = min(gains) if len(gains) == 2 else -math.inf
                                desired = reference_candidate[:3]
                                alignment = float(np.dot(candidate[:3], desired) / (
                                    np.linalg.norm(candidate[:3]) * np.linalg.norm(desired) + 1e-12))
                                desired_unit = desired / (np.linalg.norm(desired) + 1e-12)
                                projected_progress_m = float(
                                    .05 * np.dot(candidate[:3], desired_unit))
                                evaluated_candidates.append({
                                    "spec": candidate_spec, "candidate": candidate,
                                    "swept": swept, "clearance_safe": clearance_safe,
                                    "pairwise_clearance_safe": pairwise_clearance_safe,
                                    "historical_joint_contact_exit": (
                                        historical_joint_contact_exit),
                                    "collision_stage": collision_stage,
                                    "historical_corridor_floor_m": (
                                        historical_corridor_floor_m),
                                    "worst_clearance_gain_m": worst_gain,
                                    "join_direction_alignment": alignment,
                                    "projected_progress_m": projected_progress_m,
                                    "task_space_trajectory": task_space_trajectory,
                                    "task_space_displacement_m": task_space_displacement,
                                    "task_space_path_length_m": (
                                        task_space_trajectory.path_length_m),
                                    "task_space_target_progress_m": (
                                        task_space_target_progress_m),
                                    "predicted_joint": predicted_joint,
                                    "joint_normalized_action": (
                                        candidate_joint_action
                                        if args.supervisor_direct_joint_replay_enabled
                                        else None),
                                })
                            # Shadow-only task-space diversity audit.  This
                            # deliberately does not alter candidate selection
                            # until paired video/trace validation confirms that
                            # its ranking identifies useful escape directions.
                            for index, item in enumerate(evaluated_candidates):
                                alternatives = [
                                    other["task_space_trajectory"]
                                    for other_index, other in enumerate(evaluated_candidates)
                                    if other_index != index
                                ]
                                item["task_space_novelty_m"] = (
                                    min(trajectory_distance_m(
                                        item["task_space_trajectory"], other)
                                        for other in alternatives)
                                    if alternatives else math.inf)
                            safe_candidates = [
                                item for item in evaluated_candidates if item["clearance_safe"]]
                            candidate_summaries = [{
                                "kind": item["spec"].kind,
                                "candidate_id": item["spec"].candidate_id,
                                "scale": item["spec"].scale,
                                "clearance_safe": item["clearance_safe"],
                                "historical_joint_contact_exit": item[
                                    "historical_joint_contact_exit"],
                                "collision_stage": item["collision_stage"],
                                "worst_clearance_gain_m": item["worst_clearance_gain_m"],
                                "self_clearance_gain_m": float(
                                    item["swept"].final_self_clearance_m
                                    - item["swept"].baseline_self_clearance_m),
                                "join_direction_alignment": item["join_direction_alignment"],
                                "projected_progress_m": item["projected_progress_m"],
                                "task_space_target_progress_m": item[
                                    "task_space_target_progress_m"],
                                "task_space_path_length_m": item[
                                    "task_space_path_length_m"],
                                "task_space_novelty_m": item[
                                    "task_space_novelty_m"],
                                "task_space_displacement_m": item[
                                    "task_space_displacement_m"],
                                "baseline_clearance_m": min(
                                    item["swept"].baseline_self_clearance_m,
                                    item["swept"].baseline_environment_clearance_m),
                            } for item in evaluated_candidates]
                            raw_candidate_summaries = candidate_summaries
                            candidate_summaries = phase_scoped_candidate_summaries(
                                raw_candidate_summaries, proposal.state)
                            selected_index = select_rollback_candidate_index(
                                candidate_summaries,
                                safety_weight_multiplier=(
                                    escape_progress_mask.safety_weight_multiplier
                                    if (escape_controls_phase_active and
                                        args.supervisor_escape_progress_mask_control_enabled)
                                    else 1.0),
                                blocked_candidate_ids=(
                                    escape_progress_mask.blocked_candidate_ids
                                    if (escape_controls_phase_active and
                                        args.supervisor_escape_progress_mask_control_enabled)
                                    else ()))
                            if proposal.state == RollbackState.REPLAY.value:
                                selected_index = select_replay_direct_candidate_index(
                                    candidate_summaries)
                                if selected_index is not None:
                                    if replay_temporary_escape_active:
                                        rollback_cursor.reset_tracking_controller()
                                    replay_temporary_escape_steps = 0
                                    replay_temporary_escape_active = False
                                    replay_temporary_escape_burst_steps = 0
                                    replay_temporary_escape_direction = None
                                    replay_temporary_escape_direction_baseline_m = None
                                    replay_temporary_escape_blocked_directions.clear()
                            tabu_shadow_index = (
                                select_rollback_candidate_index(
                                    candidate_summaries,
                                    blocked_candidate_ids=(
                                        escape_cycle_breaker.blocked_candidate_ids))
                                if escape_controls_phase_active else None)
                            if (escape_controls_phase_active and
                                    args.supervisor_escape_cycle_breaker_control_enabled):
                                selected_index = tabu_shadow_index
                            pulse_shadow_index = (
                                select_safe_step_pulse_index(
                                    candidate_summaries, selected_index)
                                if (escape_controls_phase_active and
                                    escape_progress_mask.active) else None)
                            if (escape_controls_phase_active
                                    and args.supervisor_safe_step_pulse_control_enabled
                                    and pulse_shadow_index is not None):
                                selected_index = pulse_shadow_index
                            taskspace_upward_override_index = None
                            taskspace_lateral_detour_index = None
                            if (taskspace_upward_escape_baseline_z is None
                                    and proposal.state == RollbackState.JOIN.value
                                    and recovery_depth_attempts >=
                                    args.supervisor_taskspace_upward_escape_minimum_recovery_depth):
                                taskspace_upward_escape_baseline_z = float(
                                    obs["robot0_eef_pos"][2])
                                taskspace_upward_escape_required_dz_m = (
                                    required_axis_escape_displacement(
                                        taskspace_upward_escape_baseline_z,
                                        (None if rollback_target_eef is None else
                                         float(rollback_target_eef[2])),
                                        sign=1,
                                        minimum_displacement_m=(
                                            args.supervisor_taskspace_upward_escape_target_actual_dz_m),
                                        historical_margin_m=(
                                            args.supervisor_taskspace_upward_escape_history_height_margin_m)))
                            taskspace_upward_actual_dz_m = (
                                0.0 if taskspace_upward_escape_baseline_z is None else
                                max(0.0, float(obs["robot0_eef_pos"][2])
                                    - taskspace_upward_escape_baseline_z))
                            taskspace_upward_completion = min(
                                1.0,
                                taskspace_upward_actual_dz_m
                                / (taskspace_upward_escape_required_dz_m
                                   if taskspace_upward_escape_required_dz_m is not None
                                   else args.supervisor_taskspace_upward_escape_target_actual_dz_m))
                            taskspace_progress_protection_released = (
                                update_progress_protection_release(
                                    taskspace_progress_protection_released,
                                    taskspace_upward_completion,
                                    release_completion=(
                                        args.supervisor_taskspace_escape_progress_protection_release_completion),
                                    reactivate_completion=(
                                        args.supervisor_taskspace_escape_progress_protection_reactivate_completion)))
                            taskspace_progress_protection_weight = (
                                0.0 if taskspace_progress_protection_released else
                                protected_progress_weight(
                                    args.supervisor_taskspace_escape_protected_progress_regression_weight,
                                    taskspace_upward_completion,
                                    decay_start=(
                                        args.supervisor_taskspace_escape_progress_protection_decay_start),
                                    release_completion=(
                                        args.supervisor_taskspace_escape_progress_protection_release_completion)))
                            if (args.supervisor_taskspace_escape_detour_control_enabled
                                    and proposal.state == RollbackState.JOIN.value):
                                taskspace_escape_detour.begin_decision()
                                if taskspace_upward_completion >= 1.0:
                                    taskspace_escape_detour.resume_axis(
                                        float(obs["robot0_eef_pos"][2]),
                                        transition="geometric_escape_complete")
                            if (proposal.state == RollbackState.JOIN.value
                                    and args.supervisor_taskspace_upward_escape_control_enabled
                                    and recovery_depth_attempts >=
                                    args.supervisor_taskspace_upward_escape_minimum_recovery_depth
                                    and taskspace_upward_escape_steps <
                                    args.supervisor_taskspace_upward_escape_step_budget
                                    and taskspace_upward_completion < 1.0):
                                if (args.supervisor_taskspace_escape_detour_control_enabled
                                        and taskspace_escape_detour.mode ==
                                        AxisEscapeDetourController.DETOUR):
                                    taskspace_lateral_detour_index = (
                                        select_lateral_detour_candidate_index(
                                            candidate_summaries,
                                            primary_axis=2,
                                            blocked_direction_keys=(
                                                taskspace_escape_detour.blocked_directions),
                                            required_direction_key=(
                                                taskspace_escape_detour.locked_direction),
                                            protected_progress_axis=(
                                                args.supervisor_taskspace_escape_protected_progress_axis),
                                            protected_progress_desired_sign=(
                                                args.supervisor_taskspace_escape_protected_progress_desired_sign),
                                            protected_progress_regression_weight=(
                                                taskspace_progress_protection_weight),
                                            maximum_target_regression_m=(
                                                args.supervisor_taskspace_upward_escape_maximum_target_regression_m)))
                                    if (taskspace_lateral_detour_index is None
                                            and taskspace_escape_detour.locked_direction
                                            is not None):
                                        # A packet direction can become unsafe after
                                        # real motion. Cool it down and try another
                                        # direction now instead of rediscovering the
                                        # same one-step dead end after another probe.
                                        taskspace_escape_detour.abandon_locked_direction()
                                        taskspace_lateral_detour_index = (
                                            select_lateral_detour_candidate_index(
                                                candidate_summaries,
                                                primary_axis=2,
                                                blocked_direction_keys=(
                                                    taskspace_escape_detour.blocked_directions),
                                                protected_progress_axis=(
                                                    args.supervisor_taskspace_escape_protected_progress_axis),
                                                protected_progress_desired_sign=(
                                                    args.supervisor_taskspace_escape_protected_progress_desired_sign),
                                                protected_progress_regression_weight=(
                                                    taskspace_progress_protection_weight),
                                                maximum_target_regression_m=(
                                                    args.supervisor_taskspace_upward_escape_maximum_target_regression_m)))
                                    if taskspace_lateral_detour_index is None:
                                        taskspace_escape_detour.resume_axis(
                                            float(obs["robot0_eef_pos"][2]),
                                            transition="no_safe_lateral_candidate")
                                    else:
                                        direction_key = temporary_escape_direction_key(
                                            candidate_summaries[
                                                taskspace_lateral_detour_index]["candidate_id"])
                                        taskspace_escape_detour.lock_direction(direction_key)
                                        selected_index = taskspace_lateral_detour_index
                                if taskspace_lateral_detour_index is None:
                                    taskspace_upward_override_index = (
                                        select_axis_priority_taskspace_candidate_index(
                                        candidate_summaries, axis=2, sign=1,
                                        priority_completion=(
                                            taskspace_upward_completion),
                                        minimum_axis_displacement_m=(
                                            args.supervisor_taskspace_upward_escape_minimum_predicted_dz_m),
                                        minimum_axis_alignment=(
                                            args.supervisor_taskspace_upward_escape_minimum_axis_alignment),
                                        maximum_target_regression_m=(
                                            args.supervisor_taskspace_upward_escape_maximum_target_regression_m),
                                        protected_progress_axis=(
                                            args.supervisor_taskspace_escape_protected_progress_axis),
                                        protected_progress_desired_sign=(
                                            args.supervisor_taskspace_escape_protected_progress_desired_sign),
                                        protected_progress_regression_weight=(
                                            taskspace_progress_protection_weight),
                                        enforce_preferred_until_completion=(
                                            args.supervisor_taskspace_escape_progress_protection_release_completion)))
                                if (args.supervisor_taskspace_escape_detour_control_enabled
                                        and taskspace_lateral_detour_index is None
                                        and taskspace_upward_override_index is None):
                                    # Planning-level blockage is evidence too:
                                    # waiting for a physical stagnation window is
                                    # impossible when no safe axis action exists.
                                    taskspace_escape_detour.axis_candidate_unavailable()
                                    taskspace_lateral_detour_index = (
                                        select_lateral_detour_candidate_index(
                                            candidate_summaries,
                                            primary_axis=2,
                                            blocked_direction_keys=(
                                                taskspace_escape_detour.blocked_directions),
                                            protected_progress_axis=(
                                                args.supervisor_taskspace_escape_protected_progress_axis),
                                            protected_progress_desired_sign=(
                                                args.supervisor_taskspace_escape_protected_progress_desired_sign),
                                            protected_progress_regression_weight=(
                                                taskspace_progress_protection_weight),
                                            maximum_target_regression_m=(
                                                args.supervisor_taskspace_upward_escape_maximum_target_regression_m)))
                                    if taskspace_lateral_detour_index is not None:
                                        direction_key = temporary_escape_direction_key(
                                            candidate_summaries[
                                                taskspace_lateral_detour_index]["candidate_id"])
                                        taskspace_escape_detour.lock_direction(direction_key)
                                        selected_index = taskspace_lateral_detour_index
                                if taskspace_upward_override_index is not None:
                                    selected_index = taskspace_upward_override_index
                            selected = (None if selected_index is None
                                        else evaluated_candidates[selected_index])
                            skipped_blocked_reference = False
                            temporary_replay_escape_selected = False
                            if (selected is None
                                    and proposal.state == RollbackState.REPLAY.value):
                                skipped_blocked_reference = (
                                    rollback_cursor.skip_blocked_replay_reference(
                                        eef_pos=obs["robot0_eef_pos"],
                                        eef_rotation=obs["robot0_eef_quat"],
                                        position_tolerance_m=.012,
                                        orientation_tolerance_rad=.12))
                                if skipped_blocked_reference:
                                    replay_temporary_escape_steps = 0
                                    replay_temporary_escape_active = False
                                    replay_temporary_escape_burst_steps = 0
                                    replay_temporary_escape_direction = None
                                    replay_temporary_escape_direction_baseline_m = None
                                    replay_temporary_escape_blocked_directions.clear()
                                elif (replay_temporary_escape_steps
                                      <= args.supervisor_replay_temporary_escape_max_steps
                                      and rollback_cursor.temporary_replay_escape_steps
                                      < args.supervisor_replay_temporary_escape_step_budget):
                                    blocked_directions = set(
                                        replay_temporary_escape_blocked_directions)
                                    if (replay_temporary_escape_direction is not None
                                            and replay_temporary_escape_steps >=
                                            args.supervisor_replay_temporary_escape_max_steps):
                                        blocked_directions.add(
                                            replay_temporary_escape_direction)
                                    temporary_index = select_temporary_replay_escape_index(
                                        raw_candidate_summaries,
                                        blocked_direction_keys=blocked_directions,
                                        required_direction_key=(
                                            replay_temporary_escape_direction
                                            if (replay_temporary_escape_direction is not None
                                                and replay_temporary_escape_direction
                                                not in blocked_directions
                                                and replay_temporary_escape_steps <
                                                args.supervisor_replay_temporary_escape_max_steps)
                                            else None),
                                        progress_weight=(
                                            temporary_escape_progress_weight(
                                                replay_temporary_escape_burst_steps)))
                                    if temporary_index is not None:
                                        if not replay_temporary_escape_active:
                                            rollback_cursor.reset_tracking_controller()
                                        replay_temporary_escape_active = True
                                        selected_direction = temporary_escape_direction_key(
                                            raw_candidate_summaries[temporary_index][
                                                "candidate_id"])
                                        if selected_direction != replay_temporary_escape_direction:
                                            replay_temporary_escape_steps = 0
                                            replay_temporary_escape_direction_baseline_m = float(
                                                evaluated_candidates[temporary_index][
                                                    "swept"].baseline_self_clearance_m)
                                        replay_temporary_escape_direction = selected_direction
                                        replay_temporary_escape_steps += 1
                                        replay_temporary_escape_burst_steps += 1
                                        temporary_replay_escape_selected = True
                                        selected_index = temporary_index
                                        selected = evaluated_candidates[selected_index]
                            if trace_file is not None:
                                for item in evaluated_candidates:
                                    swept = item["swept"]
                                    _write_trace(trace_file, {
                                        "event": "complex_rollback_candidate",
                                        "task_id": task_id, "episode_idx": episode_idx,
                                        "action_index": action_index + 1,
                                        "rollback_state": proposal.state,
                                        "target_action_index": proposal.target_action_index,
                                        "candidate_id": item["spec"].candidate_id,
                                        "candidate_kind": item["spec"].kind,
                                        "candidate_scale": item["spec"].scale,
                                        "task_space_eef_positions_m": item[
                                            "task_space_trajectory"].positions_m,
                                        "task_space_displacement_m": item[
                                            "task_space_displacement_m"],
                                        "task_space_path_length_m": item[
                                            "task_space_path_length_m"],
                                        "task_space_target_progress_m": item[
                                            "task_space_target_progress_m"],
                                        "task_space_novelty_m": item[
                                            "task_space_novelty_m"],
                                        "current_joint_for_prediction": current_joint,
                                        "predicted_joint_for_prediction": item[
                                            "predicted_joint"],
                                        "observed_joint_velocity": obs["robot0_joint_vel"],
                                        "joint_replay_braking_step": (
                                            joint_replay_braking_step),
                                        "baseline_self_clearance_m":
                                            swept.baseline_self_clearance_m,
                                        "final_self_clearance_m": swept.final_self_clearance_m,
                                        "baseline_environment_clearance_m":
                                            swept.baseline_environment_clearance_m,
                                        "final_environment_clearance_m":
                                            swept.final_environment_clearance_m,
                                        "minimum_self_pair": swept.self_pair,
                                        "minimum_environment_pair": swept.environment_pair,
                                        "baseline_self_pair_clearances":
                                            swept.baseline_self_pair_clearances,
                                        "final_self_pair_clearances":
                                            swept.final_self_pair_clearances,
                                        "baseline_environment_pair_clearances":
                                            swept.baseline_environment_pair_clearances,
                                        "final_environment_pair_clearances":
                                            swept.final_environment_pair_clearances,
                                        "pairwise_clearance_safe":
                                            item["pairwise_clearance_safe"],
                                        "historical_joint_contact_exit":
                                            item["historical_joint_contact_exit"],
                                        "collision_stage": item["collision_stage"],
                                        "historical_corridor_floor_m":
                                            item["historical_corridor_floor_m"],
                                        "clearance_buffer_m": clearance_buffer_m,
                                        "worst_clearance_gain_m":
                                            item["worst_clearance_gain_m"],
                                        "self_clearance_gain_m": float(
                                            swept.final_self_clearance_m
                                            - swept.baseline_self_clearance_m),
                                        "join_direction_alignment":
                                            item["join_direction_alignment"],
                                        "projected_progress_m":
                                            item["projected_progress_m"],
                                        "clearance_safe": item["clearance_safe"],
                                        "selected": item is selected,
                                    })
                                _write_trace(trace_file, {
                                    "event": "complex_rollback_proposal",
                                    "task_id": task_id, "episode_idx": episode_idx,
                                    "action_index": action_index + 1,
                                    "rollback_state": proposal.state,
                                    "recovery_depth_attempt": recovery_depth_attempts,
                                    "minimum_replan_steps": (
                                        rollback_cursor.minimum_replan_steps),
                                    "minimum_replan_history_depth": (
                                        rollback_cursor.minimum_replan_history_depth),
                                    "minimum_replan_spatial_retreat_m": (
                                        rollback_cursor.minimum_replan_spatial_retreat_m),
                                    "actual_spatial_retreat_m": (
                                        rollback_cursor.spatial_retreat_m),
                                    "reached_history_depth": (
                                        rollback_cursor.reached_history_depth),
                                    "escape_controls_phase_active":
                                        escape_controls_phase_active,
                                    "target_action_index": proposal.target_action_index,
                                    "joint_history_control_enabled":
                                        args.supervisor_joint_history_control_enabled,
                                    "target_joint_error_rad": (
                                        None if rollback_target_joint is None else float(
                                            np.linalg.norm(
                                                current_joint - np.asarray(
                                                    rollback_target_joint, dtype=float)))),
                                    "candidate_count": len(evaluated_candidates),
                                    "safe_candidate_count": len(safe_candidates),
                                    "selected_candidate_id": (
                                        None if selected is None
                                        else selected["spec"].candidate_id),
                                    "clearance_buffer_m": clearance_buffer_m,
                                    "history_progress_weight": (
                                        None if selected is None else rollback_progress_weight(min(
                                            selected["swept"].baseline_self_clearance_m,
                                            selected["swept"].baseline_environment_clearance_m))),
                                    "escape_progress_mask_active":
                                        escape_progress_mask.active,
                                    "escape_progress_mask_control_enabled":
                                        args.supervisor_escape_progress_mask_control_enabled,
                                    "escape_safety_weight_multiplier":
                                        escape_progress_mask.safety_weight_multiplier,
                                    "escape_blocked_candidate_ids": sorted(
                                        escape_progress_mask.blocked_candidate_ids),
                                    "cycle_breaker_control_enabled":
                                        args.supervisor_escape_cycle_breaker_control_enabled,
                                    "cycle_breaker_transition":
                                        escape_cycle_breaker.last_transition,
                                    "cycle_breaker_blocked_candidate_ids": sorted(
                                        escape_cycle_breaker.blocked_candidate_ids),
                                    "cycle_breaker_shadow_candidate_id": (
                                        None if tabu_shadow_index is None else
                                        candidate_summaries[tabu_shadow_index]["candidate_id"]),
                                    "safe_step_pulse_control_enabled":
                                        args.supervisor_safe_step_pulse_control_enabled,
                                    "safe_step_pulse_shadow_candidate_id": (
                                        None if pulse_shadow_index is None else
                                        candidate_summaries[pulse_shadow_index]["candidate_id"]),
                                    "taskspace_upward_escape_control_enabled": (
                                        args.supervisor_taskspace_upward_escape_control_enabled),
                                    "taskspace_upward_escape_selected": (
                                        taskspace_upward_override_index is not None),
                                    "taskspace_upward_escape_completion": (
                                        taskspace_upward_completion),
                                    "taskspace_progress_protection_weight": (
                                        taskspace_progress_protection_weight),
                                    "taskspace_progress_protection_released": (
                                        taskspace_progress_protection_released),
                                    "taskspace_escape_detour_control_enabled": (
                                        args.supervisor_taskspace_escape_detour_control_enabled),
                                    "taskspace_escape_detour_mode": (
                                        taskspace_escape_detour.mode),
                                    "taskspace_escape_detour_selected": (
                                        taskspace_lateral_detour_index is not None),
                                    "taskspace_escape_detour_locked_direction": (
                                        taskspace_escape_detour.locked_direction),
                                    "taskspace_escape_detour_blocked_directions": sorted(
                                        taskspace_escape_detour.blocked_directions),
                                    "taskspace_escape_detour_transition": (
                                        taskspace_escape_detour.last_transition),
                                    "taskspace_upward_escape_actual_dz_m": (
                                        taskspace_upward_actual_dz_m),
                                    "taskspace_upward_escape_required_dz_m": (
                                        taskspace_upward_escape_required_dz_m),
                                    "taskspace_escape_protected_progress_axis": (
                                        args.supervisor_taskspace_escape_protected_progress_axis),
                                    "taskspace_escape_protected_progress_desired_sign": (
                                        args.supervisor_taskspace_escape_protected_progress_desired_sign),
                                    "taskspace_escape_protected_progress_regression_weight": (
                                        args.supervisor_taskspace_escape_protected_progress_regression_weight),
                                    "taskspace_upward_escape_steps": (
                                        taskspace_upward_escape_steps),
                                    "clearance_safe": selected is not None,
                                    "temporary_replay_escape_active":
                                        replay_temporary_escape_active,
                                    "temporary_replay_escape_steps":
                                        replay_temporary_escape_steps,
                                    "temporary_replay_escape_burst_steps":
                                        replay_temporary_escape_burst_steps,
                                    "temporary_replay_escape_progress_weight":
                                        temporary_escape_progress_weight(
                                            replay_temporary_escape_burst_steps),
                                    "temporary_replay_escape_total_steps":
                                        rollback_cursor.temporary_replay_escape_steps,
                                    "temporary_replay_escape_step_budget":
                                        args.supervisor_replay_temporary_escape_step_budget,
                                    "temporary_replay_escape_direction":
                                        replay_temporary_escape_direction,
                                    "temporary_replay_escape_direction_baseline_m":
                                        replay_temporary_escape_direction_baseline_m,
                                    "temporary_replay_escape_blocked_directions": sorted(
                                        replay_temporary_escape_blocked_directions),
                                    "temporary_replay_escape_selected":
                                        temporary_replay_escape_selected,
                                })
                            if skipped_blocked_reference:
                                if args.supervisor_direct_joint_replay_enabled:
                                    # The reference was advanced without motion.
                                    # Execute a true joint hold, never the stale
                                    # command toward the skipped waypoint.
                                    joint_replay_servo.reset()
                                    rollback_joint_servo_step = (
                                        joint_replay_servo.next_step(
                                            current_joint, current_joint,
                                            controller_output_rad=(
                                                joint_controller_switch.joint_output_rad)))
                                    rollback_selected_joint_action = np.asarray(
                                        rollback_joint_servo_step.normalized_action,
                                        dtype=float)
                                hold_action = np.zeros_like(reference_candidate)
                                hold_action[6] = gripper_action
                                supervisor.install_external_recovery_chunk(
                                    [hold_action])
                                if trace_file is not None:
                                    _write_trace(trace_file, {
                                        "event": "blocked_replay_reference_skipped",
                                        "task_id": task_id,
                                        "episode_idx": episode_idx,
                                        "action_index": action_index + 1,
                                        "target_action_index":
                                            proposal.target_action_index,
                                        "position_tolerance_m": .012,
                                        "orientation_tolerance_rad": .12,
                                    })
                            elif temporary_replay_escape_selected and trace_file is not None:
                                swept = selected["swept"]
                                _write_trace(trace_file, {
                                    "event": "temporary_replay_escape",
                                    "task_id": task_id,
                                    "episode_idx": episode_idx,
                                    "action_index": action_index + 1,
                                    "target_action_index": proposal.target_action_index,
                                    "escape_step": replay_temporary_escape_steps,
                                    "escape_burst_step":
                                        replay_temporary_escape_burst_steps,
                                    "progress_weight_used":
                                        temporary_escape_progress_weight(
                                            max(0, replay_temporary_escape_burst_steps - 1)),
                                    "maximum_escape_steps":
                                        args.supervisor_replay_temporary_escape_max_steps,
                                    "escape_total_steps_before_execution":
                                        rollback_cursor.temporary_replay_escape_steps,
                                    "escape_step_budget":
                                        args.supervisor_replay_temporary_escape_step_budget,
                                    "candidate_id": selected["spec"].candidate_id,
                                    "direction_key": replay_temporary_escape_direction,
                                    "self_clearance_gain_m": float(
                                        swept.final_self_clearance_m
                                        - swept.baseline_self_clearance_m),
                                })
                            if selected is None and not skipped_blocked_reference:
                                rollback_terminal_reason = (
                                    "temporary_replay_escape_exhausted"
                                    if (proposal.state == RollbackState.REPLAY.value
                                        and rollback_cursor.temporary_replay_escape_steps >=
                                        args.supervisor_replay_temporary_escape_step_budget)
                                    else "swept_collision_gate_rejected_all_candidates")
                                # Repeating the same cursor from the same state
                                # would deterministically regenerate the same
                                # rejected candidates.  Instead, consume one
                                # bounded recovery level and ask the existing
                                # preparer for an earlier / deeper history
                                # target.  This keeps the collision gate intact
                                # while giving the new attempt an independent
                                # motion budget and a genuinely different goal.
                                retry_preparation = None
                                retry_level = max(
                                    rollback_level, recovery_depth_attempts)
                                if (checkpoint_recorder is not None
                                        and recovery_failure_context is not None
                                        and retry_level <
                                        args.supervisor_replan_max_rollback_levels):
                                    retry_minimum_steps = (
                                        args.supervisor_rollback_minimum_replan_steps
                                        + retry_level
                                        * args.supervisor_rollback_replan_step_increment)
                                    retry_minimum_history_depth = (
                                        args.supervisor_rollback_minimum_history_depth
                                        + retry_level
                                        * args.supervisor_rollback_history_depth_increment)
                                    retry_minimum_spatial_retreat_m = min(
                                        args.supervisor_rollback_maximum_spatial_retreat_m,
                                        args.supervisor_rollback_minimum_spatial_retreat_m
                                        + retry_level
                                        * args.supervisor_rollback_spatial_retreat_increment_m)
                                    retry_preparation = recovery_preparer.prepare(
                                        checkpoint_recorder,
                                        failure_action_index=(
                                            recovery_failure_context["action_index"]),
                                        diagnosis=(
                                            recovery_failure_context["diagnosis"]),
                                        current_eef_pos=obs["robot0_eef_pos"],
                                        current_phase=(
                                            recovery_failure_context["phase"]),
                                        rollback_level=retry_level,
                                        minimum_history_depth_steps=(
                                            retry_minimum_history_depth),
                                        minimum_spatial_retreat_m=(
                                            retry_minimum_spatial_retreat_m),
                                    )
                                retry_started = (
                                    retry_preparation is not None
                                    and retry_preparation.ready
                                    and rollback_cursor.begin(
                                        retry_preparation,
                                        minimum_replan_steps=retry_minimum_steps,
                                        minimum_replan_history_depth=(
                                            retry_minimum_history_depth),
                                        minimum_replan_spatial_retreat_m=(
                                            retry_minimum_spatial_retreat_m),
                                        failure_eef_pos=obs["robot0_eef_pos"]))
                                if retry_started:
                                    rollback_level = retry_level
                                    recovery_depth_attempts += 1
                                    rollback_attempt_executed_steps = 0
                                    taskspace_upward_escape_steps = 0
                                    taskspace_upward_escape_baseline_z = None
                                    taskspace_upward_escape_required_dz_m = None
                                    taskspace_progress_protection_released = False
                                    taskspace_escape_detour.reset()
                                    replay_temporary_escape_steps = 0
                                    replay_temporary_escape_active = False
                                    replay_temporary_escape_burst_steps = 0
                                    replay_temporary_escape_direction = None
                                    replay_temporary_escape_direction_baseline_m = None
                                    replay_temporary_escape_blocked_directions.clear()
                                    _synchronize_robot_controller_to_current_state(env)
                                    if joint_controller_switch is not None:
                                        joint_controller_switch.synchronize_to_current_state()
                                    if trace_file is not None:
                                        _write_trace(trace_file, {
                                            "event": "rollback_retry",
                                            "task_id": task_id,
                                            "episode_idx": episode_idx,
                                            "action_index": action_index + 1,
                                            "trigger_reason": rollback_terminal_reason,
                                            "retry_level": retry_level,
                                            "join_action_index": (
                                                retry_preparation.join_state.action_index),
                                            "rollback_target_action_index": (
                                                retry_preparation.rollback_target_action_index),
                                            "minimum_replan_steps": retry_minimum_steps,
                                            "minimum_replan_history_depth": (
                                                retry_minimum_history_depth),
                                            "minimum_replan_spatial_retreat_m": (
                                                retry_minimum_spatial_retreat_m),
                                        })
                                    rollback_terminal_reason = None
                                else:
                                    rollback_cursor.reject(rollback_terminal_reason)
                                    if args.supervisor_joint_history_control_enabled:
                                        env.robots[0].controller.update_initial_joints(
                                            np.asarray(obs["robot0_joint_pos"], dtype=float))
                                    recovery_port.recovery_completed(safe=False)
                            elif selected is not None:
                                if args.supervisor_direct_joint_replay_enabled:
                                    rollback_selected_joint_action = np.asarray(
                                        selected["joint_normalized_action"], dtype=float)
                                    joint_replay_servo.commit_executed_delta(
                                        np.asarray(selected["predicted_joint"], dtype=float)
                                        - current_joint)
                                selected_swept = selected["swept"]
                                rollback_selected_audit = {
                                    "candidate_id": selected["spec"].candidate_id,
                                    "taskspace_upward_escape_selected": (
                                        taskspace_upward_override_index is not None),
                                    "taskspace_upward_escape_completion": (
                                        taskspace_upward_completion),
                                    "taskspace_escape_detour_selected": (
                                        taskspace_lateral_detour_index is not None),
                                    "taskspace_escape_detour_mode": (
                                        taskspace_escape_detour.mode),
                                    "taskspace_escape_detour_locked_direction": (
                                        taskspace_escape_detour.locked_direction),
                                    "temporary_replay_escape":
                                        temporary_replay_escape_selected,
                                    "temporary_escape_direction_key":
                                        (replay_temporary_escape_direction
                                         if temporary_replay_escape_selected else None),
                                    "temporary_escape_baseline_self_clearance_m":
                                        (selected_swept.baseline_self_clearance_m
                                         if temporary_replay_escape_selected else None),
                                    "rollback_phase": proposal.state,
                                    "rollback_target_action_index":
                                        proposal.target_action_index,
                                    "rollback_target_eef": rollback_target_eef,
                                    "rollback_target_joint": rollback_target_joint,
                                    "direct_joint_replay": (
                                        args.supervisor_direct_joint_replay_enabled),
                                    "joint_servo_target": (
                                        None if selected.get("predicted_joint") is None else
                                        selected["predicted_joint"]),
                                    "joint_servo_delta_rad": (
                                        None if rollback_joint_servo_step is None else
                                        rollback_joint_servo_step.commanded_delta_rad),
                                    "joint_servo_normalized_action": (
                                        None if rollback_joint_servo_step is None else
                                        rollback_joint_servo_step.normalized_action),
                                    "joint_replay_braking_step": (
                                        joint_replay_braking_step),
                                    "joint_replay_brake_step_index": (
                                        joint_replay_brake_steps),
                                    "predicted_final_self_clearance_m":
                                        selected_swept.final_self_clearance_m,
                                    "predicted_final_environment_clearance_m":
                                        selected_swept.final_environment_clearance_m,
                                    "predicted_minimum_self_pair": selected_swept.self_pair,
                                    "predicted_minimum_environment_pair":
                                        selected_swept.environment_pair,
                                }
                                supervisor.install_external_recovery_chunk(
                                    [selected["candidate"]])
                                if taskspace_upward_override_index is not None:
                                    taskspace_upward_escape_steps += 1
                                if (args.supervisor_joint_history_control_enabled
                                        and rollback_target_joint is not None):
                                    env.robots[0].controller.update_initial_joints(
                                        np.asarray(rollback_target_joint, dtype=float))

                    if (recovery_port is not None
                            and recovery_port.state is RecoveryHandoffState.SAFE_STOP):
                        logging.warning("Complex rollback reached safe stop")
                        break

                    if (supervisor is not None and args.supervisor_intervention_enabled
                            and not supervisor.chunk and not supervisor.recovery_bridge_active):
                        supervisor.stage_intervention_bridge()
                    supervisor_plan_empty = supervisor is not None and not supervisor.chunk
                    if supervisor_plan_empty or (supervisor is None and not action_plan):
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        policy_prompt = str(task_description)
                        recovery_prompt_active = False
                        recovery_prompt_mechanism = "none"
                        recovery_prompt_target = "request_more_evidence"
                        if supervisor is not None and args.supervisor_intervention_enabled:
                            policy_prompt, recovery_prompt_active, recovery_prompt_mechanism = (
                                supervisor.build_policy_prompt(
                                    policy_prompt,
                                    enabled=args.supervisor_recovery_prompt_enabled,
                                    fallback_suffix=args.supervisor_recovery_prompt,
                                )
                            )
                            if recovery_prompt_active and supervisor.last_intervention is not None:
                                recovery_prompt_target = supervisor.last_intervention.recovery_target
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": policy_prompt,
                        }

                        # Query model to get action. The request envelope is used only when fixed diffusion noise is
                        # explicitly enabled; default requests remain byte-for-byte compatible with the original client.
                        sampling_noise = None
                        request = element
                        if sampling_noise_rng is not None or args.visual_latent_out_path is not None:
                            if sampling_noise_rng is not None:
                                sampling_noise = sampling_noise_rng.standard_normal(
                                    (PI05_ACTION_HORIZON, PI05_ACTION_DIM)
                                ).astype(np.float32)
                            request = {
                                "__openpi_protocol_version__": FIXED_NOISE_PROTOCOL_VERSION,
                                "observation": element,
                                "sampling_noise": sampling_noise,
                                "return_visual_latents": args.visual_latent_out_path is not None,
                            }
                        response = client.infer(request)
                        action_chunk = response["actions"]
                        inference_calls += 1
                        chunk_id += 1
                        latent_sha256 = None
                        if args.visual_latent_out_path is not None:
                            visual_latents = response["visual_latents"]
                            base_latent = np.asarray(visual_latents["base_0_rgb"], dtype=np.float16)
                            wrist_latent = np.asarray(visual_latents["left_wrist_0_rgb"], dtype=np.float16)
                            latent_chunk_ids.append(chunk_id)
                            latent_action_start_indices.append(action_index + 1)
                            latent_base.append(base_latent)
                            latent_wrist.append(wrist_latent)
                            latent_sha256 = hashlib.sha256(
                                base_latent.tobytes() + wrist_latent.tobytes()
                            ).hexdigest()
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        selected_horizon = args.replan_steps
                        if supervisor is not None and args.supervisor_intervention_enabled:
                            if supervisor.novelty_only_steps_remaining <= 0:
                                selected_horizon = min(
                                    selected_horizon,
                                    supervisor.recommended_chunk_horizon)
                        selected_chunk = action_chunk[: selected_horizon]
                        if supervisor is not None:
                            if (supervisor.recovery.state is RecoveryState.REPLAN_PENDING
                                    and replan_novelty_gate is None):
                                supervisor.mark_replan_completed(selected_chunk)
                                last_policy_chunk = np.asarray(action_chunk).copy()
                            elif supervisor.recovery.state is RecoveryState.REPLAN_PENDING:
                                novelty = replan_novelty_gate.compare(
                                    failed_policy_chunk, action_chunk)
                                if trace_file is not None:
                                    _write_trace(trace_file, {
                                        "event": "replan_novelty",
                                        "task_id": task_id,
                                        "episode_idx": episode_idx,
                                        "action_index": action_index + 1,
                                        **dataclasses.asdict(novelty),
                                        "rollback_level": rollback_level,
                                    })
                                reject_repeated = (
                                    args.supervisor_replan_novelty_control_enabled
                                    and novelty.state == "repeated")
                                if reject_repeated:
                                    rollback_level += 1
                                    context = recovery_failure_context or {
                                        "action_index": action_index,
                                        "diagnosis": "repeated_replan",
                                        "phase": "observe",
                                    }
                                    requested = (
                                        recovery_port.request_recovery(
                                            reason="repeated_replan",
                                            confidence=1.0,
                                            action_index=context["action_index"],
                                            history=tuple(supervisor.history),
                                        ))
                                    depth_level = max(
                                        rollback_level, recovery_depth_attempts)
                                    minimum_steps = (
                                        args.supervisor_rollback_minimum_replan_steps
                                        + depth_level
                                        * args.supervisor_rollback_replan_step_increment)
                                    minimum_history_depth = (
                                        args.supervisor_rollback_minimum_history_depth
                                        + depth_level
                                        * args.supervisor_rollback_history_depth_increment)
                                    minimum_spatial_retreat_m = min(
                                        args.supervisor_rollback_maximum_spatial_retreat_m,
                                        args.supervisor_rollback_minimum_spatial_retreat_m
                                        + depth_level
                                        * args.supervisor_rollback_spatial_retreat_increment_m)
                                    preparation = recovery_preparer.prepare(
                                        checkpoint_recorder,
                                        failure_action_index=context["action_index"],
                                        diagnosis=context["diagnosis"],
                                        current_eef_pos=obs["robot0_eef_pos"],
                                        current_phase=context["phase"],
                                        rollback_level=depth_level,
                                        minimum_history_depth_steps=(
                                            minimum_history_depth),
                                        minimum_spatial_retreat_m=(
                                            minimum_spatial_retreat_m),
                                    ) if (requested and depth_level
                                          < args.supervisor_replan_max_rollback_levels) else None
                                    claimed = supervisor.claim_external_recovery() if requested else None
                                    if (preparation is None or not preparation.ready
                                            or claimed is None
                                            or not rollback_cursor.begin(
                                                preparation,
                                                minimum_replan_steps=minimum_steps,
                                                minimum_replan_history_depth=(
                                                    minimum_history_depth),
                                                minimum_replan_spatial_retreat_m=(
                                                    minimum_spatial_retreat_m),
                                                failure_eef_pos=obs["robot0_eef_pos"])):
                                        if claimed is not None:
                                            recovery_port.recovery_completed(safe=False)
                                        rollback_terminal_reason = (
                                            "repeated_replan_no_deeper_checkpoint")
                                    else:
                                        _synchronize_robot_controller_to_current_state(env)
                                        if joint_controller_switch is not None:
                                            joint_controller_switch.synchronize_to_current_state()
                                        if (depth_level == 1
                                                and args.supervisor_second_recovery_snapshot_path
                                                is not None):
                                            second_snapshot_path = (
                                                args.supervisor_second_recovery_snapshot_path.format(
                                                    task_id=task_id,
                                                    episode_idx=episode_idx,
                                                    action_index=action_index))
                                            if not pathlib.Path(second_snapshot_path).exists():
                                                write_fault_snapshot(
                                                    second_snapshot_path, sim=env.sim,
                                                    preparation=recovery_preparation,
                                                    failure_eef_pos=eef_pos_after,
                                                    gripper_action=float(intended_action[6]),
                                                    metadata={
                                                        "task_suite_name": args.task_suite_name,
                                                        "task_id": task_id,
                                                        "episode_idx": episode_idx,
                                                        "action_index": action_index,
                                                        "diagnosis": diagnosis,
                                                        "recovery_depth_attempt": depth_level,
                                                        "snapshot_role": (
                                                            "second_recovery_start"),
                                                    })
                                        if (args.supervisor_fault_snapshot_path is not None
                                                and not fault_snapshot_written):
                                            snapshot_path = args.supervisor_fault_snapshot_path.format(
                                                task_id=task_id,
                                                episode_idx=episode_idx,
                                                action_index=context["action_index"])
                                            write_fault_snapshot(
                                                snapshot_path, sim=env.sim,
                                                preparation=preparation,
                                                failure_eef_pos=obs["robot0_eef_pos"],
                                                gripper_action=float(
                                                    supervisor.previous_action[6]
                                                    if supervisor.previous_action is not None
                                                    else -1.0),
                                                metadata={
                                                    "task_suite_name": args.task_suite_name,
                                                    "task_id": task_id,
                                                    "episode_idx": episode_idx,
                                                    "action_index": context["action_index"],
                                                    "diagnosis": context["diagnosis"],
                                                    "recovery_depth_attempt": depth_level,
                                                })
                                            fault_snapshot_written = True
                                        recovery_depth_attempts += 1
                                        rollback_attempt_executed_steps = 0
                                        # A later recovery must not inherit the
                                        # previous attempt's geometric baseline,
                                        # exhausted axis budget, or detour state.
                                        taskspace_upward_escape_steps = 0
                                        taskspace_upward_escape_baseline_z = None
                                        taskspace_upward_escape_required_dz_m = None
                                        taskspace_progress_protection_released = False
                                        taskspace_escape_detour.reset()
                                else:
                                    accepted_chunk = action_chunk[: args.replan_steps]
                                    supervisor.mark_replan_completed(accepted_chunk)
                                    supervisor.begin_novelty_only_window(
                                        args.supervisor_post_replan_novelty_only_steps)
                                    last_policy_chunk = np.asarray(action_chunk).copy()
                                    failed_policy_chunk = None
                                    recovery_failure_context = None
                                    rollback_level = 0
                            else:
                                supervisor.install_chunk(selected_chunk)
                                last_policy_chunk = np.asarray(action_chunk).copy()
                        else:
                            action_plan.extend(selected_chunk)
                        if trace_file is not None:
                            _write_trace(
                                trace_file,
                                {
                                    "event": "inference",
                                    "task_id": task_id,
                                    "episode_idx": episode_idx,
                                    "t": t - args.num_steps_wait,
                                    "chunk_id": chunk_id,
                                    "eef_pos": obs["robot0_eef_pos"],
                                    "eef_quat": obs["robot0_eef_quat"],
                                    "joint_pos": obs["robot0_joint_pos"],
                                    "joint_vel": obs["robot0_joint_vel"],
                                    "gripper_qpos": obs["robot0_gripper_qpos"],
                                    "gripper_qvel": obs["robot0_gripper_qvel"],
                                    "action_chunk": action_chunk,
                                    "selected_chunk_horizon": selected_horizon,
                                    "recovery_prompt_active": recovery_prompt_active,
                                    "recovery_prompt_mechanism": recovery_prompt_mechanism,
                                    "recovery_prompt_target": recovery_prompt_target,
                                    "sampling_noise_seed": args.sampling_noise_seed,
                                    "sampling_noise_sha256": (
                                        hashlib.sha256(sampling_noise.tobytes()).hexdigest()
                                        if sampling_noise is not None
                                        else None
                                    ),
                                    "visual_latent_sha256": latent_sha256,
                                    "visual_latent_shape_per_camera": (
                                        list(base_latent.shape) if args.visual_latent_out_path is not None else None
                                    ),
                                },
                            )

                    if supervisor is not None:
                        pre_state = _supervisor_state(obs)
                        if (recovery_port is not None and
                                recovery_port.state is RecoveryHandoffState.ACTIVE):
                            supervisor_action_source = "complex_rollback"
                        else:
                            supervisor_action_source = (
                                "recovery_bridge" if supervisor.recovery_bridge_active else "policy_chunk"
                            )
                        intended_action, pre_decision = supervisor.next_action(pre_state, action_index + 1)
                        if intended_action is None:
                            if trace_file is not None:
                                _write_trace(trace_file, {
                                    "event": "supervisor_control", "task_id": task_id,
                                    "episode_idx": episode_idx, "t": t - args.num_steps_wait,
                                    "action_index": action_index + 1, "phase": "pre_action",
                                    "decision": pre_decision.to_dict(),
                                })
                            if pre_decision.action.value == "safe_stop":
                                logging.warning("Supervisor safe stop before action %d", action_index + 1)
                                break
                            continue
                        intended_action = np.asarray(intended_action).copy()
                        # Snapshot before this action executes.  Post-step alarms must not
                        # retroactively label their triggering policy action as controlled.
                        supervisor_control_epoch_before_action = supervisor.recovery.replans
                    else:
                        supervisor_action_source = None
                        supervisor_control_epoch_before_action = 0
                        intended_action = np.asarray(action_plan.popleft()).copy()
                    if supervisor_action_source == "complex_rollback":
                        replay_images[-1] = _annotate_recovery_frame(
                            replay_images[-1], action_index=action_index + 1,
                            source=supervisor_action_source,
                            state=rollback_cursor.state.value,
                            target_action_index=rollback_video_target,
                            candidate_id=(None if rollback_selected_audit is None
                                          else rollback_selected_audit.get("candidate_id")),
                            join_steps=rollback_cursor.join_steps,
                            replay_steps=rollback_cursor.replay_steps,
                            predicted_environment_clearance_m=(
                                None if rollback_selected_audit is None else
                                rollback_selected_audit.get(
                                    "predicted_final_environment_clearance_m")),
                        )
                    action_index += 1
                    intended_target_translation = 0.05 * np.clip(intended_action[:3], -1.0, 1.0)

                    # The event trigger is causal and deployable: it uses only the intended command available before
                    # execution, never simulator contact labels, object poses, rewards, or future observations.
                    if args.disturbance_start_mode == "translation_event" and scheduled_disturbance_start is None:
                        intended_translation_norm = float(np.linalg.norm(intended_target_translation))
                        if (
                            action_index >= args.disturbance_event_min_step
                            and intended_translation_norm >= args.disturbance_event_translation_norm
                        ):
                            event_consecutive_steps += 1
                        else:
                            event_consecutive_steps = 0
                        if event_consecutive_steps >= args.disturbance_event_consecutive_steps:
                            scheduled_disturbance_start = action_index
                            disturbance_trigger_reason = "intended_translation_event"

                    disturbance_active = (
                        scheduled_disturbance_start is not None
                        and scheduled_disturbance_start <= action_index
                        and action_index < scheduled_disturbance_start + args.disturbance_num_steps
                    )
                    executed_action = intended_action.copy()
                    if disturbance_active:
                        executed_action[:3] *= args.translation_action_scale
                        disturbed_steps += 1

                    executed_target_translation = 0.05 * np.clip(executed_action[:3], -1.0, 1.0)
                    visual_frame_index = None
                    if args.visual_out_path is not None and action_index % args.visual_stride == 0:
                        visual_frame_index = len(visual_action_indices)
                        visual_action_indices.append(action_index)
                        visual_agent_images.append(img.copy())
                        visual_wrist_images.append(wrist_img.copy())
                        # Camera calibration is public embodiment metadata. Save it at the
                        # exact pre-action image time because the wrist camera moves.
                        visual_agent_camera_to_world.append(
                            _camera_to_world(env.sim, "agentview").copy()
                        )
                        visual_wrist_camera_to_world.append(
                            _camera_to_world(env.sim, "robot0_eye_in_hand").copy()
                        )
                        visual_eef_positions.append(np.asarray(obs["robot0_eef_pos"]).copy())
                        visual_eef_quaternions.append(np.asarray(obs["robot0_eef_quat"]).copy())
                        visual_gripper_qpos.append(np.asarray(obs["robot0_gripper_qpos"]).copy())
                    eef_pos_before = np.asarray(obs["robot0_eef_pos"]).copy()
                    eef_quat_before = np.asarray(obs["robot0_eef_quat"]).copy()
                    joint_pos_before = np.asarray(obs["robot0_joint_pos"]).copy()
                    joint_vel_before = np.asarray(obs["robot0_joint_vel"]).copy()
                    gripper_qpos_before = np.asarray(obs["robot0_gripper_qpos"]).copy()
                    gripper_qvel_before = np.asarray(obs["robot0_gripper_qvel"]).copy()
                    recovery_kinematics_before = (
                        measure_public_kinematic_safety(env.sim, env.robots[0])
                        if supervisor is not None else {}
                    )

                    # Execute action in environment
                    if (args.supervisor_direct_joint_replay_enabled
                            and supervisor_action_source == "complex_rollback"):
                        if rollback_selected_joint_action is None:
                            raise RuntimeError(
                                "complex joint rollback has no joint servo command")
                        obs, reward, done, info = joint_controller_switch.step(
                            rollback_selected_joint_action,
                            gripper_action)
                    else:
                        obs, reward, done, info = env.step(executed_action.tolist())
                    supervisor_images_after = None
                    if supervisor is not None:
                        post_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        post_wrist_img = np.ascontiguousarray(
                            obs["robot0_eye_in_hand_image"][::-1, ::-1]
                        )
                        post_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(post_img, args.resize_size, args.resize_size)
                        )
                        post_wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(
                                post_wrist_img, args.resize_size, args.resize_size
                            )
                        )
                        supervisor_images_after = {
                            "agent": post_img, "wrist": post_wrist_img,
                        }
                    eef_pos_after = np.asarray(obs["robot0_eef_pos"]).copy()
                    eef_quat_after = np.asarray(obs["robot0_eef_quat"]).copy()
                    joint_pos_after = np.asarray(obs["robot0_joint_pos"]).copy()
                    joint_vel_after = np.asarray(obs["robot0_joint_vel"]).copy()
                    gripper_qpos_after = np.asarray(obs["robot0_gripper_qpos"]).copy()
                    gripper_qvel_after = np.asarray(obs["robot0_gripper_qvel"]).copy()
                    if (rollback_cursor is not None
                            and supervisor_action_source == "complex_rollback"):
                        temporary_escape_action = bool(
                            (rollback_selected_audit or {}).get(
                                "temporary_replay_escape"))
                        if temporary_escape_action:
                            rollback_cursor.accept_temporary_replay_escape()
                        else:
                            rollback_cursor.accept()
                        rollback_executed_steps += 1
                        rollback_attempt_executed_steps += 1
                        if (args.supervisor_taskspace_escape_detour_control_enabled
                                and ((rollback_selected_audit or {}).get(
                                    "taskspace_upward_escape_selected")
                                     or (rollback_selected_audit or {}).get(
                                         "taskspace_escape_detour_selected"))):
                            taskspace_escape_detour.observe(
                                float(eef_pos_after[2]),
                                (rollback_selected_audit or {}).get("candidate_id"))
                        if (args.supervisor_direct_joint_replay_enabled
                                and (rollback_selected_audit or {}).get(
                                    "joint_replay_braking_step")):
                            joint_replay_brake_steps += 1
                        if trace_file is not None and checkpoint_collision_oracle is not None:
                            actual_clearance = checkpoint_collision_oracle.current_clearances(
                                include_tracked_pairs=True)
                            actual_minimum_clearance = min(
                                actual_clearance.minimum_self_clearance_m,
                                actual_clearance.minimum_environment_clearance_m)
                            rollback_phase = (rollback_selected_audit or {}).get(
                                "rollback_phase")
                            if temporary_escape_action:
                                direction_key = (rollback_selected_audit or {}).get(
                                    "temporary_escape_direction_key")
                                cumulative_gain = (
                                    None if replay_temporary_escape_direction_baseline_m is None
                                    else actual_clearance.minimum_self_clearance_m
                                    - replay_temporary_escape_direction_baseline_m)
                                severe_worsening = (
                                    cumulative_gain is not None
                                    and cumulative_gain < -1e-4)
                                probe_complete = (
                                    replay_temporary_escape_steps >=
                                    args.supervisor_replay_temporary_escape_max_steps)
                                if (direction_key is not None and cumulative_gain is not None
                                        and (severe_worsening
                                             or (probe_complete and cumulative_gain < 1e-5))):
                                    replay_temporary_escape_blocked_directions.add(direction_key)
                                elif probe_complete and cumulative_gain is not None:
                                    # A successful three-step packet may continue, but
                                    # starts a fresh cumulative feedback measurement.
                                    replay_temporary_escape_steps = 0
                                    replay_temporary_escape_direction_baseline_m = (
                                        actual_clearance.minimum_self_clearance_m)
                            if rollback_phase == RollbackState.JOIN.value:
                                escape_progress_mask.observe(
                                    (rollback_selected_audit or {}).get("candidate_id"),
                                    actual_minimum_clearance)
                            target_eef = (rollback_selected_audit or {}).get(
                                "rollback_target_eef")
                            actual_join_error_m = (
                                None if target_eef is None else float(np.linalg.norm(
                                    eef_pos_after - np.asarray(target_eef, dtype=float))))
                            target_joint = (rollback_selected_audit or {}).get(
                                "rollback_target_joint")
                            actual_target_joint_error_rad = (
                                None if target_joint is None else float(np.linalg.norm(
                                    joint_pos_after - np.asarray(
                                        target_joint, dtype=float))))
                            actual_tracking_error = (
                                actual_target_joint_error_rad
                                if args.supervisor_direct_joint_replay_enabled
                                else actual_join_error_m)
                            if not temporary_escape_action:
                                rollback_cursor.observe_progress(
                                    actual_tracking_error,
                                    target_id=(rollback_selected_audit or {}).get(
                                        "rollback_target_action_index"),
                                    phase=(rollback_selected_audit or {}).get(
                                        "rollback_phase"))
                            if rollback_phase == RollbackState.JOIN.value:
                                escape_cycle_breaker.observe(
                                    (rollback_selected_audit or {}).get("candidate_id"),
                                    eef_pos=eef_pos_after,
                                    join_error_m=actual_join_error_m,
                                    clearance_m=actual_minimum_clearance)
                            _write_trace(trace_file, {
                                "event": "complex_rollback_response",
                                "task_id": task_id, "episode_idx": episode_idx,
                                "action_index": action_index,
                                **(rollback_selected_audit or {}),
                                "actual_self_clearance_m":
                                    actual_clearance.minimum_self_clearance_m,
                                "actual_environment_clearance_m":
                                    actual_clearance.minimum_environment_clearance_m,
                                "temporary_escape_direction_blocked_after_response": (
                                    temporary_escape_action
                                    and (rollback_selected_audit or {}).get(
                                        "temporary_escape_direction_key")
                                    in replay_temporary_escape_blocked_directions),
                                "temporary_escape_cumulative_self_clearance_gain_m": (
                                    cumulative_gain if temporary_escape_action else None),
                                "actual_self_pair": actual_clearance.self_pair,
                                "actual_environment_pair": actual_clearance.environment_pair,
                                "actual_self_pair_clearances":
                                    actual_clearance.baseline_self_pair_clearances,
                                "actual_environment_pair_clearances":
                                    actual_clearance.baseline_environment_pair_clearances,
                                "escape_progress_mask_active_after_response":
                                    escape_progress_mask.active,
                                "escape_safety_weight_multiplier_after_response":
                                    escape_progress_mask.safety_weight_multiplier,
                                "escape_progress_mask_transition":
                                    escape_progress_mask.last_transition,
                                "escape_actual_clearance_gain_m":
                                    escape_progress_mask.last_actual_gain_m,
                                "escape_blocked_candidate_ids": sorted(
                                    escape_progress_mask.blocked_candidate_ids),
                                "actual_join_error_m": actual_join_error_m,
                                "actual_target_joint_error_rad": (
                                    actual_target_joint_error_rad),
                                "maximum_joint_velocity_before_rad_s": float(
                                    np.max(np.abs(joint_vel_before))),
                                "maximum_joint_velocity_after_rad_s": float(
                                    np.max(np.abs(joint_vel_after))),
                                "cycle_breaker_transition_after_response":
                                    escape_cycle_breaker.last_transition,
                                "cycle_breaker_join_progress_m":
                                    escape_cycle_breaker.last_join_progress_m,
                                "cycle_breaker_clearance_progress_m":
                                    escape_cycle_breaker.last_clearance_progress_m,
                                "cycle_breaker_blocked_candidate_ids": sorted(
                                    escape_cycle_breaker.blocked_candidate_ids),
                            })
                    recovery_kinematics_after = (
                        measure_public_kinematic_safety(env.sim, env.robots[0])
                        if supervisor is not None else {}
                    )
                    supervisor_decision = None
                    checkpoint_decision = None
                    recovery_preparation = None
                    if supervisor is not None:
                        supervisor_state_before = {
                            "eef_pos": eef_pos_before, "eef_rotation": eef_quat_before,
                            "joint_pos": joint_pos_before, "joint_vel": joint_vel_before,
                            "gripper_qpos": gripper_qpos_before,
                            **recovery_kinematics_before,
                        }
                        supervisor_state_after = {
                            "eef_pos": eef_pos_after, "eef_rotation": eef_quat_after,
                            "joint_pos": joint_pos_after, "joint_vel": joint_vel_after,
                            "gripper_qpos": gripper_qpos_after,
                            **recovery_kinematics_after,
                        }
                        supervisor_decision = supervisor.observe_step(
                            intended_action=intended_action,
                            state_before=supervisor_state_before,
                            state_after=supervisor_state_after,
                            action_index=action_index,
                            images_before={"agent": img, "wrist": wrist_img},
                            images_after=supervisor_images_after,
                        )
                        if supervisor_decision.reason.value != "normal":
                            supervisor_alarm_counts[supervisor_decision.reason.value] += 1
                        if supervisor.last_intervention is not None:
                            supervisor_intervention_counts[supervisor.last_intervention.mode.value] += 1
                        if checkpoint_recorder is not None:
                            phase, phase_confidence = checkpoint_phase_tracker.update(
                                intended_action)
                            observability, observability_source = estimate_camera_health(
                                supervisor_images_after)
                            events = supervisor.history[-1]["events"]
                            reliability, response_source = normal_response_reliability(events)
                            contact_clear = None
                            contact_source = None
                            minimum_self_clearance_m = None
                            minimum_environment_clearance_m = None
                            self_clearance_pair = None
                            environment_clearance_pair = None
                            if checkpoint_collision_oracle is not None:
                                swept = checkpoint_collision_oracle.current_clearances()
                                minimum_self_clearance_m = swept.minimum_self_clearance_m
                                minimum_environment_clearance_m = swept.minimum_environment_clearance_m
                                self_clearance_pair = swept.self_pair
                                environment_clearance_pair = swept.environment_pair
                                contact_clear = contact_clear_from_signed_distances(
                                    minimum_self_clearance_m,
                                    minimum_environment_clearance_m,
                                    numerical_zero_band_m=(
                                        args.supervisor_checkpoint_contact_zero_band_m),
                                )
                                contact_source = "mujoco_geometry_oracle_evaluation_only"
                            checkpoint_decision = checkpoint_recorder.consider(
                                action_index=action_index,
                                state=supervisor_state_after,
                                evidence=CheckpointEvidence(
                                    phase=phase, phase_confidence=phase_confidence,
                                    observability=observability,
                                    observability_source=observability_source,
                                    response_reliability=reliability,
                                    response_source=response_source,
                                    contact_clear=contact_clear,
                                    contact_source=contact_source,
                                    minimum_self_clearance_m=minimum_self_clearance_m,
                                    minimum_environment_clearance_m=minimum_environment_clearance_m,
                                    self_clearance_pair=self_clearance_pair,
                                    environment_clearance_pair=environment_clearance_pair,
                                ),
                            )
                            if (supervisor_decision.request_replan
                                    and not supervisor.last_observation_novelty_only):
                                diagnosis = supervisor_decision.reason.value
                                for event in supervisor_decision.triggering_events:
                                    candidate = event.evidence.get("diagnostic_state")
                                    if candidate:
                                        diagnosis = str(candidate)
                                        break
                                if recovery_failure_context is None:
                                    failed_policy_chunk = (
                                        None if last_policy_chunk is None else
                                        np.asarray(last_policy_chunk).copy())
                                    recovery_failure_context = {
                                        "action_index": action_index,
                                        "diagnosis": diagnosis,
                                        "phase": phase or "observe",
                                    }
                                depth_level = recovery_depth_attempts
                                minimum_steps = (
                                    args.supervisor_rollback_minimum_replan_steps
                                    + depth_level
                                    * args.supervisor_rollback_replan_step_increment)
                                minimum_history_depth = (
                                    args.supervisor_rollback_minimum_history_depth
                                    + depth_level
                                    * args.supervisor_rollback_history_depth_increment)
                                minimum_spatial_retreat_m = min(
                                    args.supervisor_rollback_maximum_spatial_retreat_m,
                                    args.supervisor_rollback_minimum_spatial_retreat_m
                                    + depth_level
                                    * args.supervisor_rollback_spatial_retreat_increment_m)
                                recovery_preparation = recovery_preparer.prepare(
                                    checkpoint_recorder,
                                    failure_action_index=action_index,
                                    diagnosis=diagnosis,
                                    current_eef_pos=eef_pos_after,
                                    current_phase=phase or "observe",
                                    rollback_level=depth_level,
                                    minimum_history_depth_steps=(
                                        minimum_history_depth),
                                    minimum_spatial_retreat_m=(
                                        minimum_spatial_retreat_m),
                                ) if (depth_level
                                      < args.supervisor_replan_max_rollback_levels) else None
                                if (depth_level
                                        >= args.supervisor_replan_max_rollback_levels
                                        and recovery_port.state
                                        is RecoveryHandoffState.REQUESTED):
                                    claimed = supervisor.claim_external_recovery()
                                    if claimed is not None:
                                        recovery_port.recovery_completed(safe=False)
                                    rollback_terminal_reason = (
                                        "recovery_depth_limit_exhausted")
                                elif (args.supervisor_rollback_enabled
                                        and recovery_preparation is not None
                                        and recovery_preparation.ready
                                        and recovery_port.state
                                        is RecoveryHandoffState.REQUESTED):
                                    supervisor.claim_external_recovery()
                                    if not rollback_cursor.begin(
                                            recovery_preparation,
                                            minimum_replan_steps=minimum_steps,
                                            minimum_replan_history_depth=(
                                                minimum_history_depth),
                                            minimum_replan_spatial_retreat_m=(
                                                minimum_spatial_retreat_m),
                                            failure_eef_pos=eef_pos_after):
                                        recovery_port.recovery_completed(safe=False)
                                    else:
                                        _synchronize_robot_controller_to_current_state(env)
                                        if joint_controller_switch is not None:
                                            joint_controller_switch.synchronize_to_current_state()
                                        if (depth_level == 1
                                                and args.supervisor_second_recovery_snapshot_path
                                                is not None):
                                            second_snapshot_path = (
                                                args.supervisor_second_recovery_snapshot_path.format(
                                                    task_id=task_id,
                                                    episode_idx=episode_idx,
                                                    action_index=action_index))
                                            if not pathlib.Path(second_snapshot_path).exists():
                                                write_fault_snapshot(
                                                    second_snapshot_path, sim=env.sim,
                                                    preparation=recovery_preparation,
                                                    failure_eef_pos=eef_pos_after,
                                                    gripper_action=float(intended_action[6]),
                                                    metadata={
                                                        "task_suite_name": args.task_suite_name,
                                                        "task_id": task_id,
                                                        "episode_idx": episode_idx,
                                                        "action_index": action_index,
                                                        "diagnosis": diagnosis,
                                                        "recovery_depth_attempt": depth_level,
                                                        "snapshot_role": (
                                                            "second_recovery_start"),
                                                    })
                                        if (args.supervisor_fault_snapshot_path is not None
                                                and not fault_snapshot_written):
                                            snapshot_path = args.supervisor_fault_snapshot_path.format(
                                                task_id=task_id,
                                                episode_idx=episode_idx,
                                                action_index=action_index)
                                            write_fault_snapshot(
                                                snapshot_path, sim=env.sim,
                                                preparation=recovery_preparation,
                                                failure_eef_pos=eef_pos_after,
                                                gripper_action=float(intended_action[6]),
                                                metadata={
                                                    "task_suite_name": args.task_suite_name,
                                                    "task_id": task_id,
                                                    "episode_idx": episode_idx,
                                                    "action_index": action_index,
                                                    "diagnosis": diagnosis,
                                                    "recovery_depth_attempt": depth_level,
                                                })
                                            fault_snapshot_written = True
                                        recovery_depth_attempts += 1
                                        rollback_attempt_executed_steps = 0
                                        # The ordinary alarm path needs the same
                                        # per-attempt reset as repeated replanning.
                                        taskspace_upward_escape_steps = 0
                                        taskspace_upward_escape_baseline_z = None
                                        taskspace_upward_escape_required_dz_m = None
                                        taskspace_progress_protection_released = False
                                        taskspace_escape_detour.reset()
                    if trace_file is not None:
                        previous = np.zeros_like(intended_action) if previous_action is None else previous_action
                        actual_translation = eef_pos_after - eef_pos_before
                        _write_trace(
                            trace_file,
                            {
                                "event": "step",
                                "task_id": task_id,
                                "episode_idx": episode_idx,
                                "t": t - args.num_steps_wait,
                                "chunk_id": chunk_id,
                                "action_index": action_index,
                                # Keep "action" as an alias for the original policy output for backward compatibility.
                                "action": intended_action,
                                "intended_action": intended_action,
                                "executed_action": executed_action,
                                "disturbance_active": disturbance_active,
                                "disturbance_start_mode": args.disturbance_start_mode,
                                "scheduled_disturbance_start": scheduled_disturbance_start,
                                "disturbance_trigger_reason": disturbance_trigger_reason,
                                "translation_action_scale": (
                                    args.translation_action_scale if disturbance_active else 1.0
                                ),
                                "intended_target_translation": intended_target_translation,
                                "executed_target_translation": executed_target_translation,
                                "action_delta_norm": np.linalg.norm(intended_action - previous),
                                "eef_pos_before": eef_pos_before,
                                "eef_pos_after": eef_pos_after,
                                "actual_translation": actual_translation,
                                "eef_translation_norm": np.linalg.norm(actual_translation),
                                "eef_quat_before": eef_quat_before,
                                "eef_quat_after": eef_quat_after,
                                "joint_pos_before": joint_pos_before,
                                "joint_pos_after": joint_pos_after,
                                "joint_vel_before": joint_vel_before,
                                "joint_vel_after": joint_vel_after,
                                "joint_margin": recovery_kinematics_after.get("joint_margin"),
                                "singularity_sigma": recovery_kinematics_after.get("singularity_sigma"),
                                "kinematic_measurement_status": recovery_kinematics_after.get(
                                    "kinematic_measurement_status"
                                ),
                                # These cannot be inferred from absence of an alarm.
                                # Future calibrated providers must fill them explicitly.
                                "contact_clear": None,
                                "recovery_observability": None,
                                "task_phase": None,
                                "gripper_qpos_before": gripper_qpos_before,
                                "gripper_qpos_after": gripper_qpos_after,
                                "gripper_qvel_before": gripper_qvel_before,
                                "gripper_qvel_after": gripper_qvel_after,
                                "reward": reward,
                                "done": done,
                                "visual_frame_index": visual_frame_index,
                                "supervisor_decision": (
                                    supervisor_decision.to_dict() if supervisor_decision is not None else None
                                ),
                                "supervisor_intervention": (
                                    supervisor.last_intervention.to_dict()
                                    if supervisor is not None and supervisor.last_intervention is not None else None
                                ),
                                "supervisor_action_source": supervisor_action_source,
                                "recovery_checkpoint_admitted": (
                                    checkpoint_decision.admitted
                                    if checkpoint_decision is not None else None
                                ),
                                "recovery_checkpoint_rejection_reasons": (
                                    checkpoint_decision.reasons
                                    if checkpoint_decision is not None else None
                                ),
                                "recovery_checkpoint_evidence": (
                                    dict(checkpoint_decision.history_row)
                                    if checkpoint_decision is not None else None
                                ),
                                "recovery_preparation_ready": (
                                    recovery_preparation.ready
                                    if recovery_preparation is not None else None
                                ),
                                "recovery_preparation_reason": (
                                    recovery_preparation.reason
                                    if recovery_preparation is not None else None
                                ),
                                "recovery_join_action_index": (
                                    recovery_preparation.join_state.action_index
                                    if (recovery_preparation is not None
                                        and recovery_preparation.join_state is not None)
                                    else None
                                ),
                                "recovery_target_action_index": (
                                    recovery_preparation.rollback_target_action_index
                                    if recovery_preparation is not None else None
                                ),
                                "recovery_reference_count": (
                                    len(recovery_preparation.replay_references)
                                    if recovery_preparation is not None else None
                                ),
                                "supervisor_control_epoch_before_action": (
                                    supervisor_control_epoch_before_action
                                ),
                            },
                        )
                    previous_action = intended_action.copy()
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    if supervisor_action_source != "complex_rollback":
                        policy_budget.consume_policy_action()
                    t += 1

                except Exception:
                    # Preserve the full stack in experimental logs.  A one-line
                    # message hid integration failures inside optional online
                    # supervisor components and made them look like task failures.
                    logging.exception("Caught exception during LIBERO episode")
                    break

            task_episodes += 1
            total_episodes += 1
            if trace_file is not None:
                _write_trace(
                    trace_file,
                    {
                        "event": "episode_end",
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "success": bool(done),
                        "inference_calls": inference_calls,
                        "executed_actions": action_index + 1,
                        "disturbed_steps": disturbed_steps,
                        "disturbance_start_mode": args.disturbance_start_mode,
                        "actual_disturbance_start": scheduled_disturbance_start,
                        "disturbance_trigger_reason": disturbance_trigger_reason,
                        "sampling_noise_seed": args.sampling_noise_seed,
                        "supervisor_enabled": args.supervisor_enabled,
                        "supervisor_test_event": args.supervisor_test_event,
                        "supervisor_replans": (supervisor.recovery.replans if supervisor is not None else 0),
                        "supervisor_alarm_counts": dict(supervisor_alarm_counts),
                        "supervisor_intervention_enabled": args.supervisor_intervention_enabled,
                        "rollback_executed_steps": rollback_executed_steps,
                        "rollback_attempt_executed_steps": (
                            rollback_attempt_executed_steps),
                        "recovery_depth_attempts": recovery_depth_attempts,
                        "rollback_join_steps": (
                            rollback_cursor.join_steps if rollback_cursor is not None else 0),
                        "rollback_replay_steps": (
                            rollback_cursor.replay_steps if rollback_cursor is not None else 0),
                        "rollback_final_state": (
                            rollback_cursor.state.value if rollback_cursor is not None else None),
                        "rollback_terminal_reason": rollback_terminal_reason,
                        "policy_actions_since_budget_reset": (
                            policy_budget.actions_since_reset),
                        "policy_budget_reset_count": policy_budget.reset_count,
                        "supervisor_intervention_counts": dict(supervisor_intervention_counts),
                    },
                )

            # Include every experimental identity field that can differ across rollouts. This prevents silent
            # overwrites and makes videos / visual arrays joinable to JSONL records without guessing.
            suffix = "success" if done else "failure"
            artifact_stem = _episode_artifact_stem(
                args=args,
                task_id=task_id,
                episode_idx=episode_idx,
                disturbance_start=scheduled_disturbance_start,
                suffix=suffix,
            )
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"{artifact_stem}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            if args.visual_out_path is not None:
                np.savez_compressed(
                    pathlib.Path(args.visual_out_path) / f"{artifact_stem}.npz",
                    action_indices=np.asarray(visual_action_indices, dtype=np.int32),
                    agent_images=np.asarray(visual_agent_images, dtype=np.uint8),
                    wrist_images=np.asarray(visual_wrist_images, dtype=np.uint8),
                    agent_camera_intrinsic=_camera_intrinsic(
                        env.sim, "agentview", args.resize_size, args.resize_size
                    ),
                    wrist_camera_intrinsic=_camera_intrinsic(
                        env.sim, "robot0_eye_in_hand", args.resize_size, args.resize_size
                    ),
                    agent_camera_to_world=np.asarray(visual_agent_camera_to_world, dtype=np.float64),
                    wrist_camera_to_world=np.asarray(visual_wrist_camera_to_world, dtype=np.float64),
                    eef_positions=np.asarray(visual_eef_positions, dtype=np.float64),
                    eef_quaternions=np.asarray(visual_eef_quaternions, dtype=np.float64),
                    gripper_qpos=np.asarray(visual_gripper_qpos, dtype=np.float64),
                    camera_names=np.asarray(["agentview", "robot0_eye_in_hand"]),
                    image_transform=np.asarray("rotate180_then_resize_with_pad"),
                    saved_image_shape=np.asarray([args.resize_size, args.resize_size], dtype=np.int32),
                    geometry_schema_version=np.asarray(3, dtype=np.int32),
                    observation_time=np.asarray("pre_action"),
                    eef_reference=np.asarray("robot0_grip_site"),
                    eef_quaternion_convention=np.asarray("xyzw"),
                    camera_extrinsic_convention=np.asarray("opencv_camera_to_world"),
                    projection_flip_x=np.asarray(True),
                    projection_flip_y=np.asarray(False),
                )
            if args.visual_latent_out_path is not None:
                np.savez_compressed(
                    pathlib.Path(args.visual_latent_out_path) / f"{artifact_stem}.npz",
                    task_id=np.asarray(task_id, dtype=np.int32),
                    episode_idx=np.asarray(episode_idx, dtype=np.int32),
                    chunk_ids=np.asarray(latent_chunk_ids, dtype=np.int32),
                    action_start_indices=np.asarray(latent_action_start_indices, dtype=np.int32),
                    base_0_rgb=np.asarray(latent_base, dtype=np.float16),
                    left_wrist_0_rgb=np.asarray(latent_wrist, dtype=np.float16),
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if trace_file is not None:
        trace_file.close()
    if supervisor is not None and supervisor.logger is not None:
        supervisor.logger.close()


def _write_trace(trace_file, record):
    """Write one self-contained record without changing the evaluation behavior."""
    trace_file.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
    trace_file.flush()


def _episode_artifact_stem(args, task_id, episode_idx, disturbance_start, suffix):
    start_segment = "none" if disturbance_start is None else f"{disturbance_start:03d}"
    scale_segment = f"{args.translation_action_scale:.3f}".replace(".", "p")
    noise_segment = "none" if args.sampling_noise_seed is None else str(args.sampling_noise_seed)
    # task_id is only unique inside a LIBERO suite.  Include the suite so that
    # e.g. Spatial task 9 and LIBERO-90 task 9 cannot overwrite each other.
    suite_segment = str(args.task_suite_name).replace("/", "_").replace("\\", "_")
    return (
        f"rollout_{suite_segment}_seed{args.seed}_noise{noise_segment}_"
        f"task{task_id:02d}_episode{episode_idx:03d}_"
        f"{args.disturbance_start_mode}_start{start_segment}_n{args.disturbance_num_steps:03d}_"
        f"scale{scale_segment}_{suffix}"
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON-serialize {type(value)}")


def _supervisor_state(obs):
    """Public-sensor state used by the pre-action instruction guard."""
    return {
        "eef_pos": np.asarray(obs["robot0_eef_pos"]).copy(),
        "eef_rotation": np.asarray(obs["robot0_eef_quat"]).copy(),
        "joint_pos": np.asarray(obs["robot0_joint_pos"]).copy(),
        "joint_vel": np.asarray(obs["robot0_joint_vel"]).copy(),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).copy(),
    }


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
