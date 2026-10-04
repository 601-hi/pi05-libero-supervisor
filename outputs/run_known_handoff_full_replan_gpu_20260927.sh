#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
TOOLS=/root/gpufree-data/supervisor-tools-v1
TRACE=/root/gpufree-data/libero-traces/spatial_alltasks_replan5_seed7_1ep.jsonl
OUT=/root/gpufree-data/libero-traces/known_handoff_full_replan_20260927
RESULT="$OUT/task0_prefix60_join20_target10.json"
mkdir -p "$OUT"

cleanup() {
  if [[ -f "$OUT/policy.pid" ]]; then
    kill "$(cat "$OUT/policy.pid")" 2>/dev/null || true
  fi
}
trap cleanup EXIT

test -s "$TRACE"

cd "$OPENPI"
nohup .venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /root/gpufree-data/openpi-data/openpi-assets/checkpoints/pi05_libero \
  >"$OUT/policy.log" 2>&1 &
echo $! >"$OUT/policy.pid"

for _ in $(seq 1 120); do
  if grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log" 2>/dev/null; then
    break
  fi
  sleep 2
done
grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log"

cd "$TOOLS"
LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
PYTHONPATH="$OPENPI/third_party/libero:$TOOLS" \
"$OPENPI/examples/libero/.venv/bin/python" \
  scripts/run_closed_loop_hybrid_recovery_smoke.py \
  "$TRACE" \
  --output "$RESULT" \
  --suite libero_spatial \
  --task-id 0 \
  --seed 7 \
  --prefix-actions 60 \
  --join-action 20 \
  --rollback-target-action 10 \
  --maximum-escape-steps 180 \
  --maximum-replay-steps 260 \
  --maximum-steps-per-reference 50 \
  --replan-actions 220 \
  --maximum-replan-inferences 50 \
  --sampling-noise-seed 2026092701 \
  --replan-safety-shadow-only \
  >"$OUT/runner.log" 2>&1

"$OPENPI/examples/libero/.venv/bin/python" - "$RESULT" <<'PY' | tee "$OUT/summary.txt"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
result = json.loads(path.read_text(encoding="utf-8"))
summary = result["replan_summary"]
print("RESULT:", path)
print("FINAL_STATE:", result["final_state"])
print("HANDOFF_STATE:", result["recovery_handoff_state"])
print("ESCAPE_STEPS:", result["escape_steps"])
print("REPLAN_INFERENCE_CALLED:", summary.get("inference_called"))
print("REPLAN_INFERENCE_CALLS:", summary.get("inference_calls", 0))
print("REPLAN_EXECUTED_ACTIONS:", summary.get("executed_actions", 0))
print("REPLAN_STOPPED_REASON:", summary.get("stopped_reason"))
print("TASK_SUCCESS:", summary.get("task_success"))
print("PHASE_COUNTS:", {
    phase: sum(row.get("phase") == phase for row in result["trace"])
    for phase in ("escape", "replay", "replan_inference", "replan_execute", "replan_precheck")
})
PY

echo "KNOWN_HANDOFF_FULL_REPLAN_DONE $(date --iso-8601=seconds)" | tee "$OUT/DONE"
