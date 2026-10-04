#!/usr/bin/env bash
set -euo pipefail

source_path=/root/gpufree-data/droid-schema-sample/1.0.0/r2d2_faceblur-train.tfrecord-00022-of-00031.part
fixed_path=/root/gpufree-data/droid-schema-sample/1.0.0/r2d2_faceblur-train.tfrecord-00022-of-00031.fixed
final_path=/root/gpufree-data/droid-schema-sample/1.0.0/r2d2_faceblur-train.tfrecord-00022-of-00031
expected_source_bytes=109361871
expected_final_bytes=100973263

pkill -f '^python3 /root/gpufree-data/fetch_droid100_complete.py ' || true
test "$(readlink -f "$source_path")" = "$source_path"
test "$(stat -c %s "$source_path")" = "$expected_source_bytes"
dd if="$source_path" of="$fixed_path" bs=1048576 count=24 status=none
dd if="$source_path" of="$fixed_path" oflag=append conv=notrunc bs=1048576 skip=32 status=none
test "$(stat -c %s "$fixed_path")" = "$expected_final_bytes"
mv "$fixed_path" "$final_path"
echo "REPAIRED_BYTES=$(stat -c %s "$final_path")"
