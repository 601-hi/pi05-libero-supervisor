#!/usr/bin/env bash
set -euo pipefail

repo=/root/gpufree-data/vla-workspace/openpi
normal_pid=95877
experiment_logs=/root/gpufree-data/experiment-logs
traces=/root/gpufree-data/libero-traces
videos=/root/gpufree-data/libero-videos

while kill -0 "$normal_pid" 2>/dev/null; do
  sleep 10
done

cd "$repo"
for scale_tag in 025 050 075; do
  case "$scale_tag" in
    025) scale=0.25 ;;
    050) scale=0.50 ;;
    075) scale=0.75 ;;
  esac
  tag="independent_disturb_scale${scale_tag}_start40_len10_spatial_3ep_seed11_noise20260902"
  trace="$traces/${tag}.jsonl"
  log="$experiment_logs/${tag}.log"
  if [[ -e "$trace" ]]; then
    echo "SKIP_EXISTING $trace"
    continue
  fi
  env \
    LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
    PYTHONPATH=/root/gpufree-data/vla-workspace/openpi/third_party/libero \
    MUJOCO_GL=egl \
    examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
      --args.task-suite-name libero_spatial \
      --args.num-trials-per-task 3 \
      --args.replan-steps 5 \
      --args.seed 11 \
      --args.sampling-noise-seed 20260902 \
      --args.disturbance-start-step 40 \
      --args.disturbance-num-steps 10 \
      --args.translation-action-scale "$scale" \
      --args.trace-out-path "$trace" \
      --args.video-out-path "$videos/$tag" \
    > "$log" 2>&1
done

date > "$experiment_logs/independent_anomalies_complete.txt"
