"""Portable fault-state snapshots for causal rollback comparisons."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from vla_supervisor.checkpoint_recovery import HistoricalJoinState, ReverseReplayReference
from vla_supervisor.checkpoint_runtime import RecoveryPreparationDecision


SCHEMA_VERSION = 1


def _preparation_to_dict(preparation: RecoveryPreparationDecision) -> dict:
    return {
        "ready": bool(preparation.ready),
        "reason": str(preparation.reason),
        "join_state": (
            None if preparation.join_state is None
            else asdict(preparation.join_state)),
        "rollback_target_action_index": preparation.rollback_target_action_index,
        "replay_references": [asdict(item) for item in preparation.replay_references],
    }


def _preparation_from_dict(payload: dict) -> RecoveryPreparationDecision:
    join_payload = payload.get("join_state")
    join = None if join_payload is None else HistoricalJoinState(
        action_index=int(join_payload["action_index"]),
        eef_pos=tuple(float(x) for x in join_payload["eef_pos"]),
        joint_pos=tuple(float(x) for x in join_payload["joint_pos"]),
        eef_rotation=tuple(float(x) for x in join_payload.get("eef_rotation", ())),
        joint_velocity=tuple(float(x) for x in join_payload.get("joint_velocity", ())),
        gripper_state=float(join_payload.get("gripper_state", 0.0)),
    )
    references = tuple(
        ReverseReplayReference(
            action_index=int(item["action_index"]),
            joint_target=tuple(float(x) for x in item["joint_target"]),
            eef_target=tuple(float(x) for x in item["eef_target"]),
            eef_rotation_target=tuple(
                float(x) for x in item.get("eef_rotation_target", ())),
        )
        for item in payload.get("replay_references", ()))
    return RecoveryPreparationDecision(
        ready=bool(payload["ready"]),
        reason=str(payload["reason"]),
        join_state=join,
        rollback_target_action_index=payload.get("rollback_target_action_index"),
        replay_references=references,
    )


def write_fault_snapshot(path, *, sim, preparation: RecoveryPreparationDecision,
                         failure_eef_pos, gripper_action: float,
                         metadata: dict | None = None) -> Path:
    """Atomically write physics state plus the already-prepared recovery plan."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sim.forward()
    state = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
    data = getattr(sim, "data", None)
    mocap_pos = np.asarray(
        getattr(data, "mocap_pos", ()), dtype=np.float64).copy()
    mocap_quat = np.asarray(
        getattr(data, "mocap_quat", ()), dtype=np.float64).copy()
    actuator_ctrl = np.asarray(
        getattr(data, "ctrl", ()), dtype=np.float64).copy()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sim_state_size": int(state.size),
        "model_nq": int(sim.model.nq),
        "model_nv": int(sim.model.nv),
        "failure_eef_pos": [float(x) for x in failure_eef_pos],
        "gripper_action": float(gripper_action),
        "preparation": _preparation_to_dict(preparation),
        "metadata": dict(metadata or {}),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            sim_state=state,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            actuator_ctrl=actuator_ctrl,
            metadata_json=np.asarray(
                json.dumps(payload, ensure_ascii=False, sort_keys=True)))
    temporary.replace(destination)
    return destination


def load_fault_snapshot(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        state = np.asarray(archive["sim_state"], dtype=np.float64).copy()
        auxiliary = {
            "mocap_pos": np.asarray(archive["mocap_pos"], dtype=np.float64).copy(),
            "mocap_quat": np.asarray(archive["mocap_quat"], dtype=np.float64).copy(),
            "actuator_ctrl": np.asarray(
                archive["actuator_ctrl"], dtype=np.float64).copy(),
        }
        payload = json.loads(str(archive["metadata_json"].item()))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported fault snapshot schema")
    if state.size != int(payload["sim_state_size"]):
        raise ValueError("fault snapshot state length is inconsistent")
    return state, auxiliary, payload, _preparation_from_dict(payload["preparation"])


def restore_fault_snapshot(path, *, sim) -> tuple[dict, RecoveryPreparationDecision]:
    """Restore a compatible MuJoCo model exactly, then run forward kinematics."""
    state, auxiliary, payload, preparation = load_fault_snapshot(path)
    if int(sim.model.nq) != int(payload["model_nq"]):
        raise ValueError("fault snapshot qpos dimension does not match environment")
    if int(sim.model.nv) != int(payload["model_nv"]):
        raise ValueError("fault snapshot qvel dimension does not match environment")
    current_size = np.asarray(sim.get_state().flatten()).size
    if current_size != state.size:
        raise ValueError("fault snapshot flattened state does not match environment")
    sim.set_state_from_flattened(state)
    data = getattr(sim, "data", None)
    for saved_name, data_name in (
            ("mocap_pos", "mocap_pos"),
            ("mocap_quat", "mocap_quat"),
            ("actuator_ctrl", "ctrl")):
        saved = auxiliary[saved_name]
        current = np.asarray(getattr(data, data_name, ()), dtype=np.float64)
        if current.shape != saved.shape:
            raise ValueError(f"fault snapshot {saved_name} shape does not match environment")
        if saved.size:
            getattr(data, data_name)[:] = saved
    sim.forward()
    return payload, preparation
