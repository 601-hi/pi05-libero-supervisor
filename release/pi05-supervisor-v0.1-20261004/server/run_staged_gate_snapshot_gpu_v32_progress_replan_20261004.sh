#!/usr/bin/env bash
set -euo pipefail

TOOLS=/root/gpufree-data/supervisor-tools-v1
export MODE=load
export SNAPSHOT=${SNAPSHOT:-/root/gpufree-data/libero-traces/staged_gate_snapshot_v25d_capture_ep2/snapshots/fault_task00_ep002_a0081.npz}
export OUT=${OUT:-/root/gpufree-data/libero-traces/staged_gate_snapshot_v34_progress095_ep2}
export PROGRESS_ONLY_REPLAN=1
export PROGRESS_REPLAN_MINIMUM_SPATIAL_RETREAT_M=${PROGRESS_REPLAN_MINIMUM_SPATIAL_RETREAT_M:-0.095}

exec "$TOOLS/run_staged_gate_snapshot_gpu_v30_20261004.sh"
