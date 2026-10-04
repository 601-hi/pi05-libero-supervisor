#!/usr/bin/env bash
set -euo pipefail
videos=/root/gpufree-data/vla-workspace/openpi/data/libero/videos
out=/root/gpufree-data/supervisor-results/visual_pilot_task1_ep1
mkdir -p "$out"
extract() {
  local input="$1" output="$2"
  ffmpeg -y -loglevel error -i "$videos/$input" \
    -vf "select='eq(n,25)+eq(n,29)+eq(n,32)+eq(n,35)+eq(n,38)+eq(n,42)',scale=256:256,tile=6x1" \
    -frames:v 1 "$out/$output"
}
extract rollout_seed21_noise2026090321_task01_episode001_random_start029_n000_scale1p000_success.mp4 normal_strip.png
extract rollout_seed21_noise2026090321_task01_episode001_random_start029_n010_scale0p250_success.mp4 scale025_strip.png
extract rollout_seed21_noise2026090321_task01_episode001_random_start029_n010_scale0p500_success.mp4 scale050_strip.png
extract rollout_seed21_noise2026090321_task01_episode001_random_start029_n010_scale0p750_success.mp4 scale075_strip.png
sha256sum "$out"/*.png
