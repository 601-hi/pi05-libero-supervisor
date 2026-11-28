# Multi-fault paired evaluation of the π0.5 LIBERO supervisor

[Download the Chinese statistics workbook](intervention-statistics-zh.xlsx) · [Inspect the 80 paired outcomes](intervention_audit.csv) · [Machine-readable summary](summary.json) · [Frozen trial matrix](matrix.json)

Across 80 paired conditions, task completion increased from **66/80 (82.5%) to 68/80 (85.0%)**. The controller rescued two of the 14 baseline failures and preserved all 66 baseline successes.

## Protocol

The five LIBERO tasks were placing a wine bottle in a cabinet or on a rack, and placing alphabet soup, cream cheese, or salad dressing in a basket. Each task used initial-state indices 28 and 29, eight conditions, and two arms for 160 rollouts. The environment seed was 51 and the π0.5 sampling seed was 20261121. The policy normally consumed five actions per inference. After the first routed alert the arms had the same 240-step physical budget, including recovery actions.

The shadow arm logged the alert and continued the original plan. The intervention arm accepted at most one routed alert. Depending on the sensor evidence, it either refreshed the plan immediately or attempted a vision-guided return to a historical reference before requesting a new π0.5 plan. The primary endpoint was LIBERO's episode success flag. A replan counted only when a fresh inference and a subsequent policy action were both recorded after the alert.

| Condition | Intervention applied by the test harness | Shadow successes | Supervised successes |
|---|---|---:|---:|
| Normal | Nominal environment and commands | 10/10 | 10/10 |
| Transient slip | Zero target and finger contact-geometry friction parameters for 20 physical steps, then restore | 6/10 | 7/10 |
| Sustained sliding-friction reduction | Set target and finger sliding coefficients to 15% of nominal through the rest of the episode, including recovery | 10/10 | 10/10 |
| Vertical command stall | Suppress the vertical translation command | 7/10 | 8/10 |
| Direction error | Rotate the horizontal translation command by 75 degrees | 6/10 | 6/10 |
| Delay | Apply the action commanded five steps earlier | 9/10 | 9/10 |
| Reduced gain | Scale translation to 20% | 8/10 | 8/10 |
| Rotation stall | Suppress the rotation command | 10/10 | 10/10 |

The five command faults were scheduled at zero-based policy actions 45–84 inclusive. Friction injection began after two consecutive steps with a closing command and target height at least 3 cm above its initial value. Simulator state supplied the fault generator and offline exposure audit. The online recovery router used its sensor interface. All 70 fault-condition pairs experienced the scheduled intervention in both arms.

## What triggered recovery

Twelve cases triggered the router. Ten subsequently executed a new plan. Of those ten, two were rescues, four remained failures and four would also have succeeded without intervention. Two additional baseline failures hit the wrist-force limit during physical recovery before handoff. Six baseline failures received no routed alert.

| Module | Endpoint | Result |
|---|---|---:|
| Action supervision (TCN branch) | Baseline failures among first interventions attributed to this branch | 1/2 |
| Visual separation | Baseline failures among first interventions attributed directly to visual separation | 4/4 |
| Pad loss with a reliable visual reference | Additional baseline failures triggering this joint branch | 3/3 |
| Physical rollback | Historical reference reached and control handed back | 5/7 |
| Replanning | Final task successes after actual replanning | 6/10 |
| Replanning on baseline failures | Baseline failures rescued after actual replanning | 2/6 |

The rescues occurred on cabinet placement under vertical command stall and rack placement following transient slip, both on initial-state index 28. Both had exactly matched intended actions and post-action end-effector positions through the alert. Across all pairs, 79/80 passed this prefix check. The unmatched pair failed in both arms. These development results cover ten task-layout combinations with repeated conditions. The sustained friction reduction exposed the system to persistent low friction, but produced no baseline failure on these layouts.

The workbook groups statistics by condition, task, alert source and recovery action. Its final block defines each module rate and denominator. The second sheet lists all 80 pairs. In the CSV, `shadow` and `router` are binary task outcomes, and `post_infer` and `post_steps` count post-alert policy inference calls and actions in the intervention arm.
