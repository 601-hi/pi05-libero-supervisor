import collections
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from gripper_intervention import VALID_MODES as VALID_GRIPPER_DISTURBANCE_MODES
from gripper_intervention import apply_gripper_intervention
from kinematic_safety import from_robosuite_robot
from object_slip_intervention import ObjectSlipIntervention, OracleLiftTrigger

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
PI05_ACTION_HORIZON = 10
PI05_ACTION_DIM = 32
FIXED_NOISE_PROTOCOL_VERSION = 1


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
    gripper_event_close_threshold: float = 0.5
    gripper_event_consecutive_steps: int = 2
    # Optional gripper-channel intervention, sharing the same scheduled disturbance window.
    # The default "none" is bitwise behavior-preserving. force_open can model late closure / grasp miss;
    # force_closed can model premature closure; delay causally replays a past intended gripper command.
    gripper_disturbance_mode: str = "none"
    gripper_delay_steps: int = 0
    # Diagnostic sidecars. Oracle object state is privileged and must never be a deployed monitor input.
    record_oracle_object_state: bool = False
    record_kinematic_safety: bool = False

    # Optional object-slip fault generator. These simulator-privileged settings create a controlled fault only;
    # every resulting trace label is prefixed oracle_only_ and must be excluded from deployed monitor inputs.
    object_slip_target_name: Optional[str] = None
    object_slip_num_steps: int = 0
    object_slip_friction_scale: float = 0.0
    object_slip_downward_force_weight_ratio: float = 0.0
    object_slip_rise_threshold_m: float = 0.03
    object_slip_lift_consecutive_steps: int = 2

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

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    valid_disturbance_modes = {"fixed", "random", "translation_event", "gripper_close_event"}
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
    if not -1.0 <= args.gripper_event_close_threshold <= 1.0:
        raise ValueError("gripper_event_close_threshold must be between -1.0 and 1.0")
    if args.gripper_event_consecutive_steps <= 0:
        raise ValueError("gripper_event_consecutive_steps must be positive")
    if args.gripper_disturbance_mode not in VALID_GRIPPER_DISTURBANCE_MODES:
        raise ValueError(
            f"gripper_disturbance_mode must be one of {sorted(VALID_GRIPPER_DISTURBANCE_MODES)}, "
            f"got {args.gripper_disturbance_mode!r}"
        )
    if args.gripper_delay_steps < 0:
        raise ValueError("gripper_delay_steps must be non-negative")
    if args.object_slip_num_steps < 0:
        raise ValueError("object_slip_num_steps must be non-negative")
    if not 0.0 <= args.object_slip_friction_scale <= 1.0:
        raise ValueError("object_slip_friction_scale must be between 0.0 and 1.0")
    if args.object_slip_downward_force_weight_ratio < 0.0:
        raise ValueError("object_slip_downward_force_weight_ratio must be non-negative")
    if args.object_slip_rise_threshold_m <= 0.0:
        raise ValueError("object_slip_rise_threshold_m must be positive")
    if args.object_slip_lift_consecutive_steps <= 0:
        raise ValueError("object_slip_lift_consecutive_steps must be positive")
    if args.visual_stride <= 0:
        raise ValueError("visual_stride must be positive")

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
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            inference_calls = 0
            chunk_id = -1
            action_index = -1
            previous_action = None
            disturbed_steps = 0
            gripper_disturbed_steps = 0
            object_slip_intervention = None
            object_slip_trigger = None
            object_slip_start = None
            object_slip_steps = 0
            intended_gripper_history = []
            visual_action_indices = []
            visual_agent_images = []
            visual_wrist_images = []
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
            gripper_event_consecutive_steps = 0
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
            elif args.disturbance_start_mode in {"translation_event", "gripper_close_event"}:
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
                        "gripper_disturbance_mode": args.gripper_disturbance_mode,
                        "gripper_delay_steps": args.gripper_delay_steps,
                        "gripper_event_close_threshold": args.gripper_event_close_threshold,
                        "gripper_event_consecutive_steps": args.gripper_event_consecutive_steps,
                        "record_oracle_object_state": args.record_oracle_object_state,
                        "record_kinematic_safety": args.record_kinematic_safety,
                        "oracle_only_object_slip_target_name": args.object_slip_target_name,
                        "oracle_only_object_slip_num_steps": args.object_slip_num_steps,
                        "oracle_only_object_slip_friction_scale": args.object_slip_friction_scale,
                        "oracle_only_object_slip_downward_force_weight_ratio": (
                            args.object_slip_downward_force_weight_ratio
                        ),
                        "oracle_only_object_slip_rise_threshold_m": args.object_slip_rise_threshold_m,
                        "oracle_only_object_slip_lift_consecutive_steps": (
                            args.object_slip_lift_consecutive_steps
                        ),
                    },
                )

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

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

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
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
                            "prompt": str(task_description),
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
                        action_plan.extend(action_chunk[: args.replan_steps])
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

                    intended_action = np.asarray(action_plan.popleft()).copy()
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

                    # Causal grasp-intent trigger. It uses only the policy's current gripper command and does not
                    # inspect simulator object poses, contacts, rewards, images, or future actions.
                    if args.disturbance_start_mode == "gripper_close_event" and scheduled_disturbance_start is None:
                        if intended_action[-1] >= args.gripper_event_close_threshold:
                            gripper_event_consecutive_steps += 1
                        else:
                            gripper_event_consecutive_steps = 0
                        if gripper_event_consecutive_steps >= args.gripper_event_consecutive_steps:
                            scheduled_disturbance_start = action_index
                            disturbance_trigger_reason = "intended_gripper_close_event"

                    disturbance_active = (
                        scheduled_disturbance_start is not None
                        and scheduled_disturbance_start <= action_index
                        and action_index < scheduled_disturbance_start + args.disturbance_num_steps
                    )
                    executed_action = intended_action.copy()
                    if disturbance_active:
                        executed_action[:3] *= args.translation_action_scale
                        disturbed_steps += 1

                    gripper_result = apply_gripper_intervention(
                        executed_action,
                        active=disturbance_active,
                        mode=args.gripper_disturbance_mode,
                        intended_gripper_history=intended_gripper_history,
                        delay_steps=args.gripper_delay_steps,
                    )
                    executed_action = gripper_result.executed_action
                    if gripper_result.applied:
                        gripper_disturbed_steps += 1

                    executed_target_translation = 0.05 * np.clip(executed_action[:3], -1.0, 1.0)
                    visual_frame_index = None
                    if args.visual_out_path is not None and action_index % args.visual_stride == 0:
                        visual_frame_index = len(visual_action_indices)
                        visual_action_indices.append(action_index)
                        visual_agent_images.append(img.copy())
                        visual_wrist_images.append(wrist_img.copy())
                    eef_pos_before = np.asarray(obs["robot0_eef_pos"]).copy()
                    eef_quat_before = np.asarray(obs["robot0_eef_quat"]).copy()
                    joint_pos_before = np.asarray(obs["robot0_joint_pos"]).copy()
                    joint_vel_before = np.asarray(obs["robot0_joint_vel"]).copy()
                    gripper_qpos_before = np.asarray(obs["robot0_gripper_qpos"]).copy()
                    gripper_qvel_before = np.asarray(obs["robot0_gripper_qvel"]).copy()
                    if args.object_slip_target_name is not None and object_slip_intervention is None:
                        target_object = env.env.get_object(args.object_slip_target_name)
                        object_slip_intervention = ObjectSlipIntervention(
                            env.sim,
                            root_body_name=target_object.root_body,
                            contact_geom_names=target_object.contact_geoms,
                            friction_scale=args.object_slip_friction_scale,
                            downward_force_weight_ratio=args.object_slip_downward_force_weight_ratio,
                        )
                        initial_target_height = float(
                            env.sim.data.body_xpos[object_slip_intervention.body_id, 2]
                        )
                        object_slip_trigger = OracleLiftTrigger(
                            initial_target_height,
                            rise_threshold_m=args.object_slip_rise_threshold_m,
                            consecutive=args.object_slip_lift_consecutive_steps,
                        )
                    object_slip_triggered = False
                    target_height_before = None
                    if object_slip_trigger is not None:
                        target_height_before = float(
                            env.sim.data.body_xpos[object_slip_intervention.body_id, 2]
                        )
                        object_slip_triggered = object_slip_trigger.update(
                            target_height_before,
                            bool(intended_action[-1] >= args.gripper_event_close_threshold),
                        )
                        if object_slip_triggered and object_slip_start is None:
                            object_slip_start = action_index
                    object_slip_active = (
                        object_slip_start is not None
                        and object_slip_start <= action_index
                        and action_index < object_slip_start + args.object_slip_num_steps
                    )
                    object_slip_result = (
                        object_slip_intervention.apply(object_slip_active)
                        if object_slip_intervention is not None
                        else None
                    )
                    if object_slip_active:
                        object_slip_steps += 1
                    oracle_object_state_before = (
                        np.asarray(obs["object-state"]).copy()
                        if args.record_oracle_object_state and "object-state" in obs
                        else None
                    )
                    kinematic_safety_before = (
                        from_robosuite_robot(env.env.robots[0]).to_dict()
                        if args.record_kinematic_safety
                        else None
                    )

                    # Execute action in environment
                    obs, reward, done, info = env.step(executed_action.tolist())
                    eef_pos_after = np.asarray(obs["robot0_eef_pos"]).copy()
                    eef_quat_after = np.asarray(obs["robot0_eef_quat"]).copy()
                    joint_pos_after = np.asarray(obs["robot0_joint_pos"]).copy()
                    joint_vel_after = np.asarray(obs["robot0_joint_vel"]).copy()
                    gripper_qpos_after = np.asarray(obs["robot0_gripper_qpos"]).copy()
                    gripper_qvel_after = np.asarray(obs["robot0_gripper_qvel"]).copy()
                    target_height_after = (
                        float(env.sim.data.body_xpos[object_slip_intervention.body_id, 2])
                        if object_slip_intervention is not None
                        else None
                    )
                    oracle_object_state_after = (
                        np.asarray(obs["object-state"]).copy()
                        if args.record_oracle_object_state and "object-state" in obs
                        else None
                    )
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
                                "gripper_disturbance_mode": args.gripper_disturbance_mode,
                                "gripper_disturbance_applied": gripper_result.applied,
                                "gripper_delay_steps": args.gripper_delay_steps,
                                "gripper_source_history_offset": gripper_result.source_history_offset,
                                "intended_gripper_action": intended_action[-1],
                                "executed_gripper_action": executed_action[-1],
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
                                "gripper_qpos_before": gripper_qpos_before,
                                "gripper_qpos_after": gripper_qpos_after,
                                "gripper_qvel_before": gripper_qvel_before,
                                "gripper_qvel_after": gripper_qvel_after,
                                # These privileged fields are ground-truth diagnostics only. Production feature
                                # exporters must reject any key with the oracle_only_ prefix.
                                "oracle_only_object_state_before": oracle_object_state_before,
                                "oracle_only_object_state_after": oracle_object_state_after,
                                "kinematic_safety_before": kinematic_safety_before,
                                "oracle_only_object_slip_active": object_slip_active,
                                "oracle_only_object_slip_triggered": object_slip_triggered,
                                "oracle_only_object_slip_start": object_slip_start,
                                "oracle_only_object_slip_target_height_before_m": target_height_before,
                                "oracle_only_object_slip_target_height_after_m": target_height_after,
                                "oracle_only_object_slip_friction_scale": (
                                    object_slip_result.friction_scale if object_slip_result is not None else None
                                ),
                                "oracle_only_object_slip_force_world_n": (
                                    object_slip_result.applied_force_world_n
                                    if object_slip_result is not None
                                    else None
                                ),
                                "reward": reward,
                                "done": done,
                                "visual_frame_index": visual_frame_index,
                            },
                        )
                    previous_action = intended_action.copy()
                    intended_gripper_history.append(float(intended_action[-1]))
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            if object_slip_intervention is not None:
                object_slip_intervention.restore()
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
                        "gripper_disturbed_steps": gripper_disturbed_steps,
                        "gripper_disturbance_mode": args.gripper_disturbance_mode,
                        "gripper_delay_steps": args.gripper_delay_steps,
                        "gripper_event_close_threshold": args.gripper_event_close_threshold,
                        "gripper_event_consecutive_steps": args.gripper_event_consecutive_steps,
                        "disturbance_start_mode": args.disturbance_start_mode,
                        "actual_disturbance_start": scheduled_disturbance_start,
                        "disturbance_trigger_reason": disturbance_trigger_reason,
                        "sampling_noise_seed": args.sampling_noise_seed,
                        "oracle_only_object_slip_target_name": args.object_slip_target_name,
                        "oracle_only_object_slip_start": object_slip_start,
                        "oracle_only_object_slip_steps": object_slip_steps,
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


def _write_trace(trace_file, record):
    """Write one self-contained record without changing the evaluation behavior."""
    trace_file.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
    trace_file.flush()


def _episode_artifact_stem(args, task_id, episode_idx, disturbance_start, suffix):
    start_segment = "none" if disturbance_start is None else f"{disturbance_start:03d}"
    scale_segment = f"{args.translation_action_scale:.3f}".replace(".", "p")
    noise_segment = "none" if args.sampling_noise_seed is None else str(args.sampling_noise_seed)
    gripper_segment = (
        args.gripper_disturbance_mode
        if args.gripper_disturbance_mode != "delay"
        else f"delay{args.gripper_delay_steps:03d}"
    )
    slip_segment = ""
    if args.object_slip_target_name is not None and args.object_slip_num_steps > 0:
        slip_target = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in args.object_slip_target_name
        )
        slip_friction = f"{args.object_slip_friction_scale:.3f}".replace(".", "p")
        slip_force = f"{args.object_slip_downward_force_weight_ratio:.3f}".replace(".", "p")
        slip_segment = (
            f"object-slip-{slip_target}-n{args.object_slip_num_steps:03d}-"
            f"friction{slip_friction}-force{slip_force}_"
        )
    return (
        f"rollout_seed{args.seed}_noise{noise_segment}_task{task_id:02d}_episode{episode_idx:03d}_"
        f"{args.disturbance_start_mode}_start{start_segment}_n{args.disturbance_num_steps:03d}_"
        f"gripper-{gripper_segment}_{slip_segment}"
        f"scale{scale_segment}_{suffix}"
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON-serialize {type(value)}")


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
