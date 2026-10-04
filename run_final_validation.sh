#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/vla-workspace/openpi
export LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config
export PYTHONPATH=/root/gpufree-data/vla-workspace/openpi/third_party/libero
PY=examples/libero/.venv/bin/python
MAIN=examples/libero/monitoring_main.py
COMMON=(--args.task-suite-name libero_spatial --args.replan-steps 5 --args.seed 14 --args.sampling-noise-seed 20260904)
TRACE=/root/gpufree-data/libero-traces
VIDEO=/root/gpufree-data/libero-videos
LOG=/root/gpufree-data/experiment-logs
mkdir -p "$TRACE" "$VIDEO" "$LOG"

"$PY" "$MAIN" "${COMMON[@]}" --args.num-trials-per-task 5 \
 --args.trace-out-path "$TRACE/v2_final_normal_spatial_5ep_seed14_noise20260904.jsonl" \
 --args.video-out-path "$VIDEO/v2_final_normal_spatial_5ep_seed14_noise20260904" \
 > "$LOG/v2_final_normal_spatial_5ep_seed14_noise20260904.log" 2>&1

for scale in 0.25 0.50 0.75; do
 tag=${scale/./}
 "$PY" "$MAIN" "${COMMON[@]}" --args.num-trials-per-task 2 \
  --args.disturbance-start-step 40 --args.disturbance-num-steps 10 \
  --args.translation-action-scale "$scale" \
  --args.trace-out-path "$TRACE/v2_final_disturb_scale${tag}_start40_len10_spatial_2ep_seed14_noise20260904.jsonl" \
  --args.video-out-path "$VIDEO/v2_final_disturb_scale${tag}_start40_len10_spatial_2ep_seed14_noise20260904" \
  > "$LOG/v2_final_disturb_scale${tag}_start40_len10_spatial_2ep_seed14_noise20260904.log" 2>&1
done
date -Is > "$LOG/v2_final_validation_complete.txt"
