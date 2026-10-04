# Causal identity v2 — no-GPU checkpoint (2026-09-15)

## What is established

- Wave 1 contains 20 frozen rigid-object episodes and complete SAM2 candidate tracks.
- Per-frame static-background homographies and candidate residual motion were extracted for all episodes.
- Conditioning motion on each episode's own gripper-close event improves the provisional development score from 10/20 to 13/20.
- A manually calibrated EEF-to-pixel quadratic projection has leave-one-episode-out median error 6.9 px and p90 error 20.6 px. It is usable only as a broad proximity gate with its uncertainty attached.
- Combining projected-gripper proximity, post-close motion, and background residual reaches a provisional 15/20 on Wave 1.

## Why these are not final accuracy numbers

The first human candidate labels were inferred partly from trajectories and are demonstrably wrong in several episodes. The runner saved boxes and centroids but not the initial masks, so overlapping candidates cannot be adjudicated reliably from crops alone. Wave 1 is therefore a consumed development/diagnostic set; its current numbers must not be presented as final evaluation.

## Negative findings

- Summed per-frame background residual alone is dominated by mask jitter and occlusion.
- Penalizing all pre-close motion does not improve beyond 13/20 because contact may begin before the command threshold and masks jitter while objects are static.
- Treating the most mobile pre-close SAM candidate as a gripper proxy falls to 9/20. Explicit gripper localization is required.
- A 400-frame failure contains 13 open-close cycles, and another late close occurs at frame 135/138. Retry state and insufficient-tail refusal are necessary.

## Implemented v2 components

- `vla_supervisor/controlled_object_selector_v2.py`: selective online state machine using close state, projected gripper proximity, static-background residual, robot-overlap rejection, margin, persistence, and unknown/refusal states.
- `vla_supervisor/gripper_projection.py`: runtime EEF-to-pixel projection carrying its p90 calibration uncertainty.
- `cross_suite_generalization/export_initial_candidate_masks_gpu.py`: GPU-only initial-mask reconstruction with a hard assertion that reconstructed candidate boxes exactly match frozen Wave 1 IDs.

## Exact next GPU action

Run only the initial-mask reconstruction for Wave 1. This requires 20 independent first-frame SAM2 calls and does not rerun video propagation. Review the resulting mask overlays to create corrected target-mask equivalence sets. Recompute Wave 1 diagnostics, freeze one v2 configuration, then evaluate it once on untouched Wave 2.
