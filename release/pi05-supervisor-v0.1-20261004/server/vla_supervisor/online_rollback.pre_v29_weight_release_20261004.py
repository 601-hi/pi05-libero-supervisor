from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import collections
import math

import numpy as np

from .adaptive_budget import AdaptivePhaseBudget, BudgetDecision
from .checkpoint_recovery import PoseRollbackController, quaternion_error_axis_angle
from .checkpoint_runtime import RecoveryPreparationDecision


class RollbackState(str, Enum):
    IDLE = "idle"
    JOIN = "join"
    REPLAY = "replay"
    REPLAN_PENDING = "replan_pending"
    SAFE_STOP = "safe_stop"


def escape_controls_allowed(rollback_state: str) -> bool:
    """Escape-only heuristics must never leak into reverse replay."""
    return str(rollback_state) == RollbackState.JOIN.value


def phase_scoped_candidate_summaries(candidate_summaries, rollback_state: str):
    """Return selection summaries valid for the active rollback phase.

    JOIN may search escape axes. REPLAY may only advance along a direct
    historical-corridor candidate; if that is unsafe, the caller must stop.
    """
    scoped = [dict(item) for item in candidate_summaries]
    if not escape_controls_allowed(rollback_state):
        for item in scoped:
            if item.get("kind") != "direct":
                item["clearance_safe"] = False
    return scoped


def replay_corridor_floor_m(*, nominal_floor_m: float = .002,
                            model_hysteresis_m: float = .0025) -> float:
    """Historical-corridor floor with bounded collision-geometry tolerance.

    MuJoCo's simplified robot collision meshes can overlap fractionally in
    nominal poses that were already executed safely.  This exception is used
    only for direct reverse replay of recorded joint states; free escape and
    environment-contact candidates retain the ordinary collision gates.
    """
    if nominal_floor_m <= 0 or not 0 <= model_hysteresis_m <= nominal_floor_m + .0005:
        raise ValueError("replay hysteresis exceeds the bounded 0.5 mm mesh tolerance")
    return float(nominal_floor_m - model_hysteresis_m)


def rollback_collision_stage(*, rollback_state: str, candidate_kind: str,
                             spatial_retreat_m: float,
                             baseline_environment_clearance_m: float,
                             final_environment_clearance_m: float,
                             contact_tolerance_m: float = 1e-5,
                             open_space_clearance_m: float = .02,
                             open_space_retreat_m: float = .05) -> str:
    """Classify collision scrutiny from measured geometry, not elapsed time."""
    baseline = float(baseline_environment_clearance_m)
    final = float(final_environment_clearance_m)
    if not np.isfinite(baseline) or not np.isfinite(final):
        return "invalid"
    if baseline <= contact_tolerance_m:
        return "contact_exit"
    if (str(rollback_state) == RollbackState.REPLAY.value
            and str(candidate_kind) == "direct"
            and float(spatial_retreat_m) >= open_space_retreat_m
            and min(baseline, final) >= open_space_clearance_m):
        return "open_history_replay"
    return "constrained_corridor"


def open_history_replay_clearance_safe(
        *, baseline_self_clearance_m: float,
        final_self_clearance_m: float,
        baseline_environment_clearance_m: float,
        final_environment_clearance_m: float,
        final_self_pair_clearances,
        final_environment_pair_clearances,
        historical_self_floor_m: float = -.0005,
        open_space_clearance_m: float = .02) -> bool:
    """Admit only recorded-joint replay in already open external space.

    The conservative requirement that every nearby pair improve is removed,
    but external clearance must remain large.  Only the robot self geometry
    receives the calibrated historical mesh tolerance.
    """
    scalars = (
        baseline_self_clearance_m, final_self_clearance_m,
        baseline_environment_clearance_m, final_environment_clearance_m)
    if not all(np.isfinite(float(value)) for value in scalars):
        return False
    if min(float(baseline_environment_clearance_m),
           float(final_environment_clearance_m)) < open_space_clearance_m:
        return False
    if float(final_self_clearance_m) < historical_self_floor_m:
        return False
    for _, _, distance in final_self_pair_clearances:
        if float(distance) < historical_self_floor_m:
            return False
    # This is intentionally stricter than the self-geometry rule: a recorded
    # robot pose does not certify the current world / object configuration.
    for _, _, distance in final_environment_pair_clearances:
        if float(distance) < 0.0:
            return False
    return True


def select_replay_direct_candidate_index(candidate_summaries) -> int | None:
    """Choose the largest swept-safe direct step during reverse replay.

    Every returned candidate has already passed the same pairwise clearance
    gate.  Selecting the smallest scale caused proportional replay commands
    to decay into sub-millimetre asymptotic motion near each dense history
    point.  The largest admitted direct scale advances through that certified
    corridor without weakening any collision threshold.
    """
    safe_direct = [
        index for index, item in enumerate(candidate_summaries)
        if item.get("kind") == "direct" and bool(item.get("clearance_safe"))
    ]
    if not safe_direct:
        return None
    return max(safe_direct, key=lambda index: float(candidate_summaries[index]["scale"]))


def select_temporary_replay_escape_index(
        candidate_summaries, *, minimum_self_clearance_gain_m: float = 1e-5,
        blocked_direction_keys=(), required_direction_key: str | None = None,
        progress_weight: float = 0.0
) -> int | None:
    """Select a short collision-safe escape when REPLAY's direct corridor is blocked.

    This is deliberately separate from normal REPLAY selection: it admits only
    escape-axis candidates that passed the full swept collision gate and that
    measurably improve self-collision clearance.  Maximum clearance gain wins;
    a smaller scale breaks ties so the exceptional motion stays local.
    """
    blocked_direction_keys = set(blocked_direction_keys)
    eligible = [
        index for index, item in enumerate(candidate_summaries)
        if item.get("kind") != "direct"
        and bool(item.get("clearance_safe"))
        and temporary_escape_direction_key(item.get("candidate_id"))
        not in blocked_direction_keys
        and (required_direction_key is None
             or temporary_escape_direction_key(item.get("candidate_id"))
             == required_direction_key)
        and float(item.get("self_clearance_gain_m", -math.inf))
        >= float(minimum_self_clearance_gain_m)
    ]
    if not eligible:
        return None
    progress_weight = float(np.clip(progress_weight, 0.0, 1.0))
    safety_values = np.asarray([
        float(candidate_summaries[index]["self_clearance_gain_m"])
        for index in eligible], dtype=float)
    progress_values = np.asarray([
        float(candidate_summaries[index].get("projected_progress_m", 0.0))
        for index in eligible], dtype=float)

    def normalized(values):
        span = float(np.max(values) - np.min(values))
        if span <= 1e-12:
            return np.ones_like(values)
        return (values - np.min(values)) / span

    safety_score = normalized(safety_values)
    progress_score = normalized(progress_values)
    combined = ((1.0 - progress_weight) * safety_score
                + progress_weight * progress_score)
    score_by_index = {
        index: float(combined[offset]) for offset, index in enumerate(eligible)}
    return max(
        eligible,
        key=lambda index: (
            score_by_index[index],
            float(candidate_summaries[index]["self_clearance_gain_m"]),
            float(candidate_summaries[index].get(
                "worst_clearance_gain_m", -math.inf)),
            -float(candidate_summaries[index]["scale"]),
        ),
    )


def temporary_escape_progress_weight(burst_steps: int, *,
                                     initial: float = .20,
                                     increment: float = .25,
                                     maximum: float = .80) -> float:
    """Fast soft-weight transfer after each certified escape microstep."""
    if burst_steps < 0:
        raise ValueError("burst_steps must be non-negative")
    return float(min(maximum, initial + increment * int(burst_steps)))


def temporary_escape_direction_key(candidate_id) -> str:
    """Collapse escape scales onto one axis/direction feedback identity."""
    text = str(candidate_id)
    if text.startswith("escape_axis_") and "_" in text:
        return text.rsplit("_", 1)[0]
    return text


def select_axis_priority_taskspace_candidate_index(
        candidate_summaries, *, axis: int, sign: int,
        priority_completion: float,
        minimum_axis_displacement_m: float = 0.0,
        minimum_axis_alignment: float = .20,
        maximum_target_regression_m: float = .001,
        protected_progress_axis: int | None = None,
        protected_progress_desired_sign: int = 1,
        protected_progress_regression_weight: float = 1.25,
) -> int | None:
    """Blend a temporary axis preference back into historical convergence.

    This helper exists for controlled fault-specific ablations.  It never
    bypasses ``clearance_safe`` and it does not encode positive Z as a generic
    robot rule.  ``priority_completion`` rises from zero to one during the
    bounded escape packet: axis priority decays while target progress gains
    weight, preventing an escape direction from drifting indefinitely away
    from the recorded historical corridor.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    if not 0.0 <= minimum_axis_alignment <= 1.0:
        raise ValueError("minimum_axis_alignment must be between zero and one")
    if protected_progress_axis is not None and protected_progress_axis not in (0, 1, 2):
        raise ValueError("protected progress axis must be 0, 1, 2, or None")
    if protected_progress_desired_sign not in (-1, 1):
        raise ValueError("protected progress desired sign must be -1 or 1")
    if protected_progress_regression_weight < 0:
        raise ValueError("protected progress regression weight must be non-negative")
    completion = float(np.clip(priority_completion, 0.0, 1.0))
    eligible = []
    for index, item in enumerate(candidate_summaries):
        displacement = np.asarray(
            item.get("task_space_displacement_m", ()), dtype=float)
        progress = item.get("task_space_target_progress_m")
        if (not bool(item.get("clearance_safe"))
                or displacement.shape != (3,)
                or progress is None
                or not np.isfinite(float(progress))
                or float(progress) < -float(maximum_target_regression_m)):
            continue
        eligible.append(index)
    if not eligible:
        return None

    def normalized(field):
        values = np.asarray([float(field(candidate_summaries[index]))
                             for index in eligible], dtype=float)
        span = float(np.max(values) - np.min(values))
        return (np.full_like(values, .5) if span <= 1e-12
                else (values - np.min(values)) / span)

    clearance = normalized(lambda item: item.get(
        "worst_clearance_gain_m", -math.inf))
    target = normalized(lambda item: item["task_space_target_progress_m"])
    axis_motion = normalized(lambda item: sign * float(
        item["task_space_displacement_m"][axis]))
    protected_regression = np.zeros(len(eligible), dtype=float)
    if protected_progress_axis is not None:
        protected_regression = normalized(lambda item: max(
            0.0,
            -protected_progress_desired_sign * float(
                item["task_space_displacement_m"][protected_progress_axis])))

    # Escape direction dominates only at the start.  Historical convergence
    # grows continuously and is the majority term by the packet midpoint.
    axis_weight = .60 * (1.0 - completion)
    target_weight = .20 + .60 * completion
    clearance_weight = 1.0 - axis_weight - target_weight
    scores = (clearance_weight * clearance
              + axis_weight * axis_motion
              + target_weight * target
              - float(protected_progress_regression_weight) * protected_regression)
    # Require a real motion along the preferred direction while it still has
    # meaningful authority.  Near completion all safe candidates may compete
    # so the controller can bend back toward history instead of continuing up.
    if axis_weight >= .15:
        preferred = []
        for offset, index in enumerate(eligible):
            displacement = np.asarray(candidate_summaries[index][
                "task_space_displacement_m"], dtype=float)
            axis_displacement = sign * float(displacement[axis])
            alignment = axis_displacement / (float(np.linalg.norm(displacement)) + 1e-12)
            if (axis_displacement >= float(minimum_axis_displacement_m)
                    and alignment >= float(minimum_axis_alignment)):
                preferred.append(offset)
        if preferred:
            best_offset = max(preferred, key=lambda offset: float(scores[offset]))
            return eligible[best_offset]
    return eligible[int(np.argmax(scores))]


def required_axis_escape_displacement(current_axis_position_m: float,
                                      historical_axis_position_m: float | None,
                                      *, sign: int,
                                      minimum_displacement_m: float,
                                      historical_margin_m: float = 0.0) -> float:
    """Compute physical escape distance from a historical safe coordinate."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    if minimum_displacement_m <= 0 or historical_margin_m < 0:
        raise ValueError("escape distance must be positive and margin non-negative")
    required = float(minimum_displacement_m)
    if historical_axis_position_m is not None:
        historical_progress = sign * (
            float(historical_axis_position_m) - float(current_axis_position_m))
        if np.isfinite(historical_progress):
            required = max(required, historical_progress + float(historical_margin_m))
    return required


def rollback_clearance_safe(clearance_pairs, *, zero_tolerance_m: float = 1e-5,
                            clearance_buffer_m: float = .005,
                            minimum_improvement_m: float = 1e-5,
                            historical_corridor_floor_m: float | None = None) -> bool:
    """Conservatively admit a rollback candidate from signed clearances.

    A clear state may approach an obstacle only while the predicted endpoint
    retains the model-error buffer.  Inside that buffer, motion must increase
    clearance.  Existing penetration is allowed to escape incrementally, but
    never to deepen.  This prevents a nominally positive sub-millimetre
    prediction from being treated as robustly collision-free.
    """
    for baseline, final in clearance_pairs:
        if baseline is None or final is None:
            return False
        baseline, final = float(baseline), float(final)
        if not np.isfinite(baseline) or not np.isfinite(final):
            return False
        if baseline > zero_tolerance_m:
            if final >= clearance_buffer_m:
                continue
            if (historical_corridor_floor_m is not None
                    and final >= float(historical_corridor_floor_m)):
                continue
            if final > zero_tolerance_m and final >= baseline + minimum_improvement_m:
                continue
            return False
        if final >= baseline + minimum_improvement_m:
            continue
        return False
    return True


def rollback_pairwise_clearance_safe(baseline_rows, final_rows, *,
                                     zero_tolerance_m: float = 1e-5,
                                     clearance_buffer_m: float = .005,
                                     minimum_improvement_m: float = 1e-5,
                                     historical_corridor_floor_m: float | None = None,
                                     escaping_existing_contact: bool = False) -> bool:
    """Compare like-for-like geom pairs and reject novel penetration.

    Rows are ``(geom_a, geom_b, signed_distance_m)``.  The oracle records all
    pairs within a fixed positive tracking radius, so an absent final pair has
    moved away.  A pair absent at baseline but penetrating at the endpoint is
    a new collision and is rejected.
    """
    def mapping(rows):
        result = {}
        for geom_a, geom_b, distance in rows:
            value = float(distance)
            if not np.isfinite(value):
                return None
            result[(str(geom_a), str(geom_b))] = value
        return result

    baseline = mapping(baseline_rows)
    final = mapping(final_rows)
    if baseline is None or final is None:
        return False
    for pair, final_distance in final.items():
        baseline_distance = baseline.get(pair)
        if baseline_distance is None:
            if final_distance < -zero_tolerance_m:
                return False
            continue
        if baseline_distance > zero_tolerance_m:
            # While another pair is already penetrating, collision-free
            # geometry pairs may trade clearance as the mechanism separates.
            # They must remain strictly positive: this never licenses a new
            # contact, while avoiding the impossible requirement that every
            # nearby finger/cabinet distance improve simultaneously.
            if (escaping_existing_contact
                    and final_distance > zero_tolerance_m):
                continue
            if final_distance >= clearance_buffer_m:
                continue
            if (historical_corridor_floor_m is not None
                    and final_distance >= float(historical_corridor_floor_m)):
                continue
            if (final_distance > zero_tolerance_m
                    and final_distance >= baseline_distance + minimum_improvement_m):
                continue
            return False
        if baseline_distance >= -zero_tolerance_m:
            if final_distance >= -zero_tolerance_m:
                continue
            return False
        if final_distance >= baseline_distance + minimum_improvement_m:
            continue
        return False
    return True


@dataclass(frozen=True)
class RollbackProposal:
    state: str
    action: tuple[float, ...] | None
    target_action_index: int | None
    reason: str


@dataclass(frozen=True)
class RollbackCandidateAction:
    candidate_id: str
    action: tuple[float, ...]
    kind: str
    scale: float


def generate_rollback_candidate_actions(reference_action: Sequence[float], *,
                                        scales=(.25, .5, 1.0)):
    """Generate direct and translation-only escape microsteps.

    Direct candidates preserve the controller's 6D request.  Escape-axis
    candidates deliberately remove rotation so a collision response can be
    attributed to one small translational direction.  The gripper command is
    always preserved.
    """
    reference = np.asarray(reference_action, dtype=float)
    if reference.shape != (7,):
        raise ValueError("rollback reference action must contain seven values")
    clean_scales = tuple(sorted({float(x) for x in scales if 0.0 < float(x) <= 1.0}))
    if not clean_scales:
        raise ValueError("at least one candidate scale in (0, 1] is required")
    candidates = []
    for scale in clean_scales:
        action = reference.copy()
        action[:6] *= scale
        candidates.append(RollbackCandidateAction(
            f"direct_{scale:g}", tuple(float(x) for x in action), "direct", scale))
    translation_norm = float(np.linalg.norm(reference[:3]))
    if translation_norm > 1e-12:
        for scale in clean_scales:
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    action = np.zeros(7, dtype=float)
                    action[axis] = sign * translation_norm * scale
                    action[6] = reference[6]
                    candidates.append(RollbackCandidateAction(
                        f"escape_axis_{axis}_{'pos' if sign > 0 else 'neg'}_{scale:g}",
                        tuple(float(x) for x in action), "escape_axis", scale))
    return tuple(candidates)


def rollback_progress_weight(baseline_clearance_m: float | None, *,
                             transition_center_m: float = .0045,
                             transition_width_m: float = .0005) -> float:
    """Continuously transfer priority from escape safety to history progress.

    The hard collision gates run before this weight is used.  Near contact the
    score favours the candidate that opens clearance fastest.  Once clearance
    approaches the 5 mm model-error buffer, direction toward the historical
    join becomes dominant, preventing an escape axis from running forever.
    """
    if transition_width_m <= 0:
        raise ValueError("transition_width_m must be positive")
    if baseline_clearance_m is None or not np.isfinite(float(baseline_clearance_m)):
        return 0.0
    value = (float(baseline_clearance_m) - transition_center_m) / transition_width_m
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-value))


def select_rollback_candidate_index(
        candidate_summaries, *, safety_weight_multiplier: float = 1.0,
        blocked_candidate_ids=(), fallback_when_all_blocked: bool = True) -> int | None:
    """Select among pre-gated candidate summaries without weakening safety.

    Each summary supplies ``kind``, ``scale``, ``clearance_safe``,
    ``worst_clearance_gain_m`` and ``join_direction_alignment``.  Direct
    joining remains preferred when safe.  Escape candidates are considered
    only when every direct candidate fails, and the smallest safe scale wins
    before clearance gain and historical alignment are compared.
    """
    blocked_candidate_ids = set(blocked_candidate_ids)
    all_safe = [
        index for index, item in enumerate(candidate_summaries)
        if bool(item["clearance_safe"])]
    safe = [index for index in all_safe
            if candidate_summaries[index].get("candidate_id") not in blocked_candidate_ids]
    if not safe and all_safe and fallback_when_all_blocked:
        safe = all_safe
    direct = [
        index for index in safe
        if candidate_summaries[index]["kind"] == "direct"]
    if direct:
        return max(direct, key=lambda index: float(candidate_summaries[index]["scale"]))
    if not safe:
        return None
    masked = float(safety_weight_multiplier) < 1.0
    if masked:
        pool = safe
    else:
        minimum_scale = min(float(candidate_summaries[index]["scale"])
                            for index in safe)
        pool = [index for index in safe
                if float(candidate_summaries[index]["scale"]) == minimum_scale]
    gains = [float(candidate_summaries[index]["worst_clearance_gain_m"])
             for index in pool]
    gain_low, gain_high = min(gains), max(gains)
    projected = [float(candidate_summaries[index].get(
        "projected_progress_m",
        candidate_summaries[index]["join_direction_alignment"])) for index in pool]
    progress_low, progress_high = min(projected), max(projected)
    baseline_clearance = min(
        (float(candidate_summaries[index]["baseline_clearance_m"])
         for index in pool
         if candidate_summaries[index].get("baseline_clearance_m") is not None),
        default=None)
    progress_weight = rollback_progress_weight(baseline_clearance)
    safety_weight = (1.0 - progress_weight) * float(np.clip(
        safety_weight_multiplier, 0.0, 1.0))
    progress_weight = 1.0 - safety_weight

    def blended_score(index):
        item = candidate_summaries[index]
        if gain_high - gain_low <= 1e-12:
            safety_score = .5
        else:
            safety_score = (
                float(item["worst_clearance_gain_m"]) - gain_low
            ) / (gain_high - gain_low)
        raw_progress = float(item.get(
            "projected_progress_m", item["join_direction_alignment"]))
        if progress_high - progress_low <= 1e-12:
            progress_score = .5
        else:
            progress_score = (raw_progress - progress_low) / (
                progress_high - progress_low)
        return (safety_weight * safety_score
                + progress_weight * progress_score,
                progress_score, safety_score)

    return max(pool, key=blended_score)


def _candidate_direction_key(candidate_id: str | None) -> str | None:
    """Return a scale-independent key for an axis escape candidate."""
    if candidate_id is None:
        return None
    parts = str(candidate_id).split("_")
    if len(parts) == 5 and parts[:2] == ["escape", "axis"]:
        return "_".join(parts[:4])
    return None


class EscapeCycleBreaker:
    """Temporarily tabu candidates proven to participate in a real cycle."""

    def __init__(self, *, window: int = 6, cooldown_steps: int = 8,
                 minimum_join_progress_m: float = .00020,
                 minimum_clearance_progress_m: float = .00005,
                 repeated_candidate_count: int = 3,
                 return_tolerance_m: float = .00035):
        if window < 3 or cooldown_steps < 1 or repeated_candidate_count < 2:
            raise ValueError("cycle breaker windows and counters must be positive")
        self.window = int(window)
        self.cooldown_steps = int(cooldown_steps)
        self.minimum_join_progress_m = float(minimum_join_progress_m)
        self.minimum_clearance_progress_m = float(minimum_clearance_progress_m)
        self.repeated_candidate_count = int(repeated_candidate_count)
        self.return_tolerance_m = float(return_tolerance_m)
        self.reset()

    def reset(self):
        self.samples = collections.deque(maxlen=self.window)
        self.cooldowns = {}
        self.last_transition = "reset"
        self.last_join_progress_m = None
        self.last_clearance_progress_m = None

    @property
    def blocked_candidate_ids(self):
        return frozenset(self.cooldowns)

    def begin_decision(self):
        self.cooldowns = {
            candidate_id: remaining - 1
            for candidate_id, remaining in self.cooldowns.items()
            if remaining > 1
        }

    def observe(self, candidate_id: str | None, *, eef_pos,
                join_error_m: float | None, clearance_m: float | None) -> bool:
        key = _candidate_direction_key(candidate_id)
        if key is None or join_error_m is None or clearance_m is None:
            self.samples.clear()
            self.last_transition = "insufficient_escape_evidence"
            return False
        position = np.asarray(eef_pos, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("eef_pos must contain three finite coordinates")
        join_error_m, clearance_m = float(join_error_m), float(clearance_m)
        if not np.isfinite(join_error_m) or not np.isfinite(clearance_m):
            self.samples.clear()
            self.last_transition = "insufficient_escape_evidence"
            return False
        sample = (str(candidate_id), key, position.copy(), join_error_m, clearance_m)
        self.samples.append(sample)
        if len(self.samples) < self.window:
            self.last_transition = "warming_up"
            return False

        first, last = self.samples[0], self.samples[-1]
        join_progress = first[3] - last[3]
        clearance_progress = last[4] - first[4]
        self.last_join_progress_m = join_progress
        self.last_clearance_progress_m = clearance_progress
        repeated = sum(row[0] == last[0] for row in self.samples)
        returned = any(
            np.linalg.norm(last[2] - row[2]) <= self.return_tolerance_m
            for row in tuple(self.samples)[:-2])
        no_progress = (
            join_progress < self.minimum_join_progress_m
            and clearance_progress < self.minimum_clearance_progress_m)
        cycling = repeated >= self.repeated_candidate_count or returned
        if no_progress and cycling and last[0] not in self.cooldowns:
            self.cooldowns[last[0]] = self.cooldown_steps
            self.last_transition = (
                "return_cycle_tabu" if returned else "repeated_no_progress_tabu")
            self.samples.clear()
            return True
        self.last_transition = "progressing" if not no_progress else "not_yet_cycling"
        return False


def select_safe_step_pulse_index(candidate_summaries, base_index: int | None, *,
                                 minimum_progress_gain_m: float = .00010,
                                 minimum_clearance_gain_m: float = 0.0) -> int | None:
    """Choose the next larger hard-safe candidate in the same direction."""
    if base_index is None:
        return None
    base = candidate_summaries[int(base_index)]
    direction = _candidate_direction_key(base.get("candidate_id"))
    if direction is None:
        return None
    base_scale = float(base["scale"])
    base_progress = float(base.get(
        "projected_progress_m", base["join_direction_alignment"]))
    eligible = []
    for index, item in enumerate(candidate_summaries):
        if (_candidate_direction_key(item.get("candidate_id")) != direction
                or float(item["scale"]) <= base_scale
                or not bool(item["clearance_safe"])
                or float(item["worst_clearance_gain_m"]) < minimum_clearance_gain_m):
            continue
        progress = float(item.get(
            "projected_progress_m", item["join_direction_alignment"]))
        if progress - base_progress >= minimum_progress_gain_m:
            eligible.append(index)
    return min(eligible, key=lambda index: float(candidate_summaries[index]["scale"]),
               default=None)


class EscapeProgressMask:
    """Temporarily reduce soft safety weighting after real escape stagnation.

    This state never changes candidate admissibility.  It only affects ranking
    among candidates that already passed every hard collision gate.
    """

    def __init__(self, *, window: int = 6,
                 minimum_clearance_gain_m: float = .00015,
                 recovery_clearance_gain_m: float = .00030,
                 validation_gain_m: float = .00005,
                 deterioration_tolerance_m: float = .00001,
                 neutral_step_limit: int = 2,
                 cooldown_steps: int = 8,
                 multiplier_levels=(.75, .5, .25)):
        if window < 2 or neutral_step_limit < 1 or cooldown_steps < 1:
            raise ValueError("escape mask windows and counters must be positive")
        self.window = int(window)
        self.minimum_clearance_gain_m = float(minimum_clearance_gain_m)
        self.recovery_clearance_gain_m = float(recovery_clearance_gain_m)
        self.validation_gain_m = float(validation_gain_m)
        self.deterioration_tolerance_m = float(deterioration_tolerance_m)
        self.neutral_step_limit = int(neutral_step_limit)
        self.cooldown_steps = int(cooldown_steps)
        self.multiplier_levels = tuple(float(x) for x in multiplier_levels)
        if not self.multiplier_levels or any(
                not 0.0 < x <= 1.0 for x in self.multiplier_levels):
            raise ValueError("mask multiplier levels must lie in (0, 1]")
        self.reset()

    def reset(self):
        self.samples = collections.deque(maxlen=self.window)
        self.active = False
        self.level_index = 0
        self.neutral_steps = 0
        self.previous_clearance_m = None
        self.cooldowns = {}
        self.last_transition = "reset"
        self.last_actual_gain_m = None

    @property
    def safety_weight_multiplier(self) -> float:
        return self.multiplier_levels[self.level_index] if self.active else 1.0

    @property
    def blocked_candidate_ids(self):
        return frozenset(self.cooldowns)

    def begin_decision(self):
        self.cooldowns = {
            candidate_id: remaining - 1
            for candidate_id, remaining in self.cooldowns.items()
            if remaining > 1
        }

    def _deactivate(self, transition, *, clear_samples=False):
        self.active = False
        self.level_index = 0
        self.neutral_steps = 0
        self.last_transition = transition
        if clear_samples:
            self.samples.clear()

    def observe(self, candidate_id: str | None, clearance_m: float | None) -> bool:
        if candidate_id is None or not str(candidate_id).startswith("escape_axis_"):
            self.samples.clear()
            self._deactivate("non_escape")
            self.previous_clearance_m = (
                None if clearance_m is None else float(clearance_m))
            return self.active
        if clearance_m is None or not np.isfinite(float(clearance_m)):
            return self.active
        candidate_id, clearance_m = str(candidate_id), float(clearance_m)
        actual_gain = (None if self.previous_clearance_m is None
                       else clearance_m - self.previous_clearance_m)
        self.previous_clearance_m = clearance_m
        self.last_actual_gain_m = actual_gain

        if self.active and actual_gain is not None:
            if actual_gain < -self.deterioration_tolerance_m:
                self.cooldowns[candidate_id] = self.cooldown_steps
                self._deactivate("deteriorated_and_cooled", clear_samples=True)
            elif actual_gain >= self.validation_gain_m:
                self.level_index = min(
                    self.level_index + 1, len(self.multiplier_levels) - 1)
                self.neutral_steps = 0
                self.last_transition = "validated_and_strengthened"
            else:
                self.neutral_steps += 1
                self.last_transition = "awaiting_validation"
                if self.neutral_steps >= self.neutral_step_limit:
                    self.cooldowns[candidate_id] = self.cooldown_steps
                    self._deactivate("no_gain_and_cooled", clear_samples=True)

        self.samples.append((candidate_id, clearance_m))
        if len(self.samples) < self.window:
            return self.active
        ids = [item[0] for item in self.samples]
        net_gain = self.samples[-1][1] - self.samples[0][1]
        repeated = len(set(ids)) == 1
        if (not self.active and candidate_id not in self.cooldowns
                and repeated and net_gain < self.minimum_clearance_gain_m):
            self.active = True
            self.level_index = 0
            self.neutral_steps = 0
            self.last_transition = "stagnation_activated"
        elif self.active and net_gain >= self.recovery_clearance_gain_m:
            self._deactivate("recovered")
        return self.active


class OnlineRollbackCursor:
    """Minimal online cursor over a prepared 6D join and reverse replay.

    Collision and kinematic approval deliberately remain outside this class.
    The caller must gate every proposed action and call ``reject`` on failure.
    """

    def __init__(self, *, maximum_steps: int = 320,
                 maximum_replay_steps: int | None = None,
                 adaptive_budget_enabled: bool = False,
                 maximum_hard_steps: int | None = None,
                 maximum_hard_replay_steps: int | None = None,
                 budget_extension_chunk: int = 80,
                 replan_when_join_reached: bool = False,
                 position_tolerance_m: float = .004,
                 orientation_tolerance_rad: float = .08,
                 joint_tolerance_rad: float = .025,
                 minimum_replan_steps: int = 0,
                 minimum_replan_history_depth: int = 0,
                 minimum_replan_spatial_retreat_m: float = 0.0):
        self.maximum_steps = int(maximum_steps)
        self.maximum_replay_steps = int(
            maximum_steps if maximum_replay_steps is None
            else maximum_replay_steps)
        if self.maximum_steps <= 0 or self.maximum_replay_steps <= 0:
            raise ValueError("rollback phase budgets must be positive")
        self.adaptive_budget_enabled = bool(adaptive_budget_enabled)
        self.replan_when_join_reached = bool(replan_when_join_reached)
        self.maximum_hard_steps = int(
            self.maximum_steps if maximum_hard_steps is None
            else maximum_hard_steps)
        self.maximum_hard_replay_steps = int(
            self.maximum_replay_steps if maximum_hard_replay_steps is None
            else maximum_hard_replay_steps)
        self.join_budget = AdaptivePhaseBudget(
            soft_limit=self.maximum_steps, hard_limit=self.maximum_hard_steps,
            extension_chunk=budget_extension_chunk)
        self.replay_budget = AdaptivePhaseBudget(
            soft_limit=self.maximum_replay_steps,
            hard_limit=self.maximum_hard_replay_steps,
            extension_chunk=budget_extension_chunk,
            require_target_advance=True)
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.joint_tolerance_rad = float(joint_tolerance_rad)
        self.default_minimum_replan_steps = max(0, int(minimum_replan_steps))
        self.default_minimum_replan_history_depth = max(
            0, int(minimum_replan_history_depth))
        self.default_minimum_replan_spatial_retreat_m = max(
            0.0, float(minimum_replan_spatial_retreat_m))
        self.controller = PoseRollbackController(
            maximum_translation_m=.003, maximum_rotation_rad=.04,
            translation_integral_gain=.12, translation_derivative_gain=.05,
            rotation_integral_gain=.05, rotation_derivative_gain=.03)
        self.reset()

    def reset(self):
        self.state = RollbackState.IDLE
        self.preparation = None
        self.reference_index = 0
        self.steps = 0
        self.join_steps = 0
        self.replay_steps = 0
        self.temporary_replay_escape_steps = 0
        self.join_budget.reset()
        self.replay_budget.reset()
        self.last_budget_decision = None
        self.controller.reset()
        self.minimum_replan_steps = self.default_minimum_replan_steps
        self.minimum_replan_history_depth = (
            self.default_minimum_replan_history_depth)
        self.minimum_replan_spatial_retreat_m = (
            self.default_minimum_replan_spatial_retreat_m)
        self.failure_eef_pos = None
        self.current_eef_pos = None
        self.deepest_reached_action_index = None

    def begin(self, preparation: RecoveryPreparationDecision, *,
              minimum_replan_steps: int | None = None,
              minimum_replan_history_depth: int | None = None,
              minimum_replan_spatial_retreat_m: float | None = None,
              failure_eef_pos: Sequence[float] | None = None) -> bool:
        self.reset()
        if (not preparation.ready or preparation.join_state is None
                or not preparation.join_state.eef_rotation
                or not preparation.replay_references):
            self.state = RollbackState.SAFE_STOP
            return False
        if any(not ref.eef_rotation_target for ref in preparation.replay_references):
            self.state = RollbackState.SAFE_STOP
            return False
        self.preparation = preparation
        if minimum_replan_steps is not None:
            self.minimum_replan_steps = max(0, int(minimum_replan_steps))
        if minimum_replan_history_depth is not None:
            self.minimum_replan_history_depth = max(
                0, int(minimum_replan_history_depth))
        if minimum_replan_spatial_retreat_m is not None:
            self.minimum_replan_spatial_retreat_m = max(
                0.0, float(minimum_replan_spatial_retreat_m))
        if failure_eef_pos is not None:
            self.failure_eef_pos = np.asarray(failure_eef_pos, dtype=float).copy()
        self.state = RollbackState.JOIN
        return True

    @property
    def reached_history_depth(self) -> int:
        if self.preparation is None or self.deepest_reached_action_index is None:
            return 0
        return max(0, int(self.preparation.join_state.action_index)
                   - int(self.deepest_reached_action_index))

    @property
    def spatial_retreat_m(self) -> float:
        if self.failure_eef_pos is None or self.current_eef_pos is None:
            return 0.0
        return float(np.linalg.norm(self.current_eef_pos - self.failure_eef_pos))

    def _minimum_replan_depth_reached(self) -> bool:
        return (
            self.steps >= self.minimum_replan_steps
            and self.reached_history_depth >= self.minimum_replan_history_depth
            and self.spatial_retreat_m >= self.minimum_replan_spatial_retreat_m
        )

    def _reached(self, eef_pos, eef_rotation, target_pos, target_rotation):
        return (
            np.linalg.norm(np.asarray(eef_pos) - np.asarray(target_pos))
            <= self.position_tolerance_m
            and np.linalg.norm(quaternion_error_axis_angle(
                eef_rotation, target_rotation)) <= self.orientation_tolerance_rad
        )

    @property
    def active_target_eef(self):
        if self.preparation is None:
            return None
        if self.state is RollbackState.JOIN:
            return tuple(self.preparation.join_state.eef_pos)
        if (self.state is RollbackState.REPLAY
                and self.reference_index < len(self.preparation.replay_references)):
            return tuple(self.preparation.replay_references[self.reference_index].eef_target)
        return None

    @property
    def active_target_joint(self):
        if self.preparation is None:
            return None
        if self.state is RollbackState.JOIN:
            return tuple(self.preparation.join_state.joint_pos)
        if (self.state is RollbackState.REPLAY
                and self.reference_index < len(self.preparation.replay_references)):
            return tuple(self.preparation.replay_references[
                self.reference_index].joint_target)
        return None

    def skip_blocked_replay_reference(self, *, eef_pos, eef_rotation,
                                      position_tolerance_m=.012,
                                      orientation_tolerance_rad=.12) -> bool:
        """Advance past a near replay waypoint only when its direct path is blocked."""
        if (self.state is not RollbackState.REPLAY
                or self.preparation is None
                or self.reference_index >= len(self.preparation.replay_references)):
            return False
        target = self.preparation.replay_references[self.reference_index]
        if (np.linalg.norm(np.asarray(eef_pos) - np.asarray(target.eef_target))
                > float(position_tolerance_m)
                or np.linalg.norm(quaternion_error_axis_angle(
                    eef_rotation, target.eef_rotation_target))
                > float(orientation_tolerance_rad)):
            return False
        self.deepest_reached_action_index = min(
            int(target.action_index), int(self.deepest_reached_action_index))
        self.reference_index += 1
        self.controller.reset()
        return True

    def reset_tracking_controller(self):
        """Drop PID memory across exceptional replay escape transitions."""
        self.controller.reset()

    def propose(self, *, eef_pos: Sequence[float],
                eef_rotation: Sequence[float], gripper_action: float,
                joint_pos: Sequence[float] | None = None,
                joint_position_mode: bool = False) -> RollbackProposal:
        self.current_eef_pos = np.asarray(eef_pos, dtype=float).copy()
        if self.state in (RollbackState.IDLE, RollbackState.SAFE_STOP,
                          RollbackState.REPLAN_PENDING):
            return RollbackProposal(self.state.value, None, None, "not_active")
        if self.state is RollbackState.JOIN:
            target = self.preparation.join_state
            target_pos, target_rot, target_index = (
                target.eef_pos, target.eef_rotation, target.action_index)
        else:
            if (self.replan_when_join_reached
                    and self._minimum_replan_depth_reached()):
                self.state = RollbackState.REPLAN_PENDING
                return RollbackProposal(
                    self.state.value, None,
                    int(self.deepest_reached_action_index),
                    "minimum_retreat_depth_reached")
            if self.reference_index >= len(self.preparation.replay_references):
                self.state = RollbackState.REPLAN_PENDING
                return RollbackProposal(self.state.value, None, None, "replay_complete")
            target = self.preparation.replay_references[self.reference_index]
            target_pos, target_rot, target_index = (
                target.eef_target, target.eef_rotation_target, target.action_index)

        target_joint = (
            target.joint_pos if self.state is RollbackState.JOIN
            else target.joint_target)
        if joint_position_mode:
            if joint_pos is None or not target_joint:
                self.state = RollbackState.SAFE_STOP
                return RollbackProposal(
                    self.state.value, None, None,
                    "joint_position_mode_missing_joint_state")
            reached = (
                np.max(np.abs(
                    np.asarray(joint_pos, dtype=float)
                    - np.asarray(target_joint, dtype=float)))
                <= self.joint_tolerance_rad)
        else:
            reached = self._reached(
                eef_pos, eef_rotation, target_pos, target_rot)
        if reached:
            self.controller.reset()
            if self.state is RollbackState.JOIN:
                self.deepest_reached_action_index = int(target_index)
                if (self.replan_when_join_reached
                        and self._minimum_replan_depth_reached()):
                    self.state = RollbackState.REPLAN_PENDING
                    reason = (
                        "join_trajectory_tube_reached"
                        if (self.minimum_replan_steps == 0
                            and self.minimum_replan_history_depth == 0)
                        else "minimum_retreat_depth_reached")
                    return RollbackProposal(
                        self.state.value, None, int(target_index),
                        reason)
                self.state = RollbackState.REPLAY
            else:
                self.deepest_reached_action_index = min(
                    int(target_index),
                    int(self.deepest_reached_action_index))
                self.reference_index += 1
            return self.propose(
                eef_pos=eef_pos, eef_rotation=eef_rotation,
                gripper_action=gripper_action, joint_pos=joint_pos,
                joint_position_mode=joint_position_mode)

        # Test the reached condition before enforcing the phase budget.  The
        # final allowed action may have landed inside tolerance; in that case
        # the state transition is already complete and must not be discarded.
        in_join = self.state is RollbackState.JOIN
        phase_steps = self.join_steps if in_join else self.replay_steps
        phase_budget = self.join_budget if in_join else self.replay_budget
        if self.adaptive_budget_enabled:
            budget_decision = phase_budget.decide(phase_steps)
        else:
            fixed_limit = self.maximum_steps if in_join else self.maximum_replay_steps
            budget_decision = BudgetDecision(
                phase_steps < fixed_limit, False, fixed_limit,
                "within_budget" if phase_steps < fixed_limit else "fixed_limit")
        self.last_budget_decision = budget_decision
        if not budget_decision.allow:
            self.state = RollbackState.SAFE_STOP
            if not self.adaptive_budget_enabled:
                reason = ("join_step_budget_exhausted" if in_join
                          else "replay_step_budget_exhausted")
            else:
                reason = (("join_" if in_join else "replay_")
                          + "step_budget_" + budget_decision.reason)
            return RollbackProposal(
                self.state.value, None, None, reason)

        reference_action = np.zeros(7, dtype=float)
        reference_action[6] = float(gripper_action)
        action = self.controller.next_action(
            eef_pos, target_pos, eef_rotation, target_rot, reference_action)
        return RollbackProposal(
            self.state.value, tuple(float(x) for x in action),
            int(target_index), "candidate_requires_safety_gate")

    def accept(self) -> None:
        if self.state not in (RollbackState.JOIN, RollbackState.REPLAY):
            raise RuntimeError("rollback action accepted outside active recovery")
        if self.state is RollbackState.JOIN:
            self.join_steps += 1
        else:
            self.replay_steps += 1
        self.steps += 1

    def accept_temporary_replay_escape(self) -> None:
        """Account an exceptional escape without consuming REPLAY's budget."""
        if self.state is not RollbackState.REPLAY:
            raise RuntimeError("temporary replay escape accepted outside replay")
        self.temporary_replay_escape_steps += 1
        # It remains real recovery motion and therefore counts toward minimum
        # physical retreat steps and the externally audited total.
        self.steps += 1

    def observe_progress(self, error_m: float | None, *, target_id=None,
                         phase: str | None = None) -> None:
        """Record post-action error for the active phase's next budget review."""
        active_phase = self.state.value if phase is None else str(phase)
        if active_phase == RollbackState.JOIN.value:
            self.join_budget.observe(error_m, target_id=target_id)
        elif active_phase == RollbackState.REPLAY.value:
            self.replay_budget.observe(error_m, target_id=target_id)

    def reject(self, reason: str = "safety_gate_rejected") -> RollbackProposal:
        self.state = RollbackState.SAFE_STOP
        return RollbackProposal(self.state.value, None, None, str(reason))
