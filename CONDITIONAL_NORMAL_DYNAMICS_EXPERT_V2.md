# Conditional Normal Dynamics Expert — Frozen v2

Status: frozen normal-only baseline, 2026-09-01.

## Model

- Five heteroscedastic MLP members, each trained on four of five episode folds.
- Training data: 20 normal episodes per LIBERO Spatial task from environment seeds 7, 11 and 13; 21,570 deployable step records total.
- Inputs: intended action, joint position/velocity, EEF pose, gripper state, previous action/response, action change, motion alignment, chunk phase and task one-hot. All inputs are observable on a physical robot before executing the monitored step.
- Outputs: conditional mean and diagonal aleatoric variance of the three-dimensional EEF translation. Ensemble disagreement contributes epistemic variance.
- Calibration: every normal episode is scored out-of-fold by a model that did not train on it. Per-task/per-axis variance scales and per-task episode thresholds are derived from these out-of-fold predictions only.
- Temporal rule: causal 2-of-3 persistence. Causal 3/5-step rolling alternatives were tested and rejected because they reduced weak-fault detection.

## Frozen operating profiles

- `q85` sensitive: accepts more false alarms.
- `q90` balanced and recommended default.
- `q95` conservative.

The checkpoint stores all three threshold sets. Profile selection is an operating-policy choice and does not change model weights.

## Independent seed17 results

Balanced `q90`, 20 episodes per condition:

| Condition | Episode detection | Active-step recall | Median delay |
|---|---:|---:|---:|
| Normal | 1/20 alarms | 0.0917% overall step alarm rate | — |
| scale 0.25 | 19/20 | 72.1% | 50 ms |
| scale 0.50 | 18/20 | 65.5% | 50 ms |
| scale 0.75 | 9/20 | 27.8% | 100 ms |

One episode exposed only four disturbed actions because the task completed; results retain it and separately record complete-exposure counts. Low-excitation episodes with mean active target norm below 0.015 m must be reported as low observability, not confidently normal.

The normal expert improves strong and medium random-onset faults over frozen old v2, but is not uniformly better for scale 0.75. This limitation motivates a later abnormal-pattern expert or principled score fusion; it must not be hidden by retuning this normal-only model on anomaly labels.

## Frozen artifacts

- Model SHA256: `2DCBF34388AB8B6BA8AF7C4FC3B5AAF64EAEB321DB998C04F3BFA99E3EE0D40A`
- Report SHA256: `41C98180DCC183EC925E440949206D3361E0686447E27EE93E379DD486824DDD`
- Training script SHA256: `0532B21636EFE779D3AC37DCD5FE57961C80713BBAD2B8FC1B8DE53F5B2C0F20`

Files:

- `outputs/conditional_normal_dynamics_expert_frozen_v2.pt`
- `outputs/conditional_normal_dynamics_expert_frozen_v2.json`
- `outputs/conditional_normal_dynamics_expert_seed17_validation.json`
- `outputs/conditional_normal_dynamics_expert_seed17_severity_sweep.json`
- `outputs/conditional_normal_dynamics_expert_temporal_seed17.json`
- `finalize_conditional_normal_dynamics_expert_v2.py`
- `evaluate_conditional_normal_expert_seed17.py`
- `evaluate_conditional_normal_expert_scales_seed17.py`
- `evaluate_normal_expert_temporal_scores.py`
