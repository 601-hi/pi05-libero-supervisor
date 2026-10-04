# Seed21 冻结监督器误差诊断

本报告仅分析冻结测试结果，不使用 seed21 标签训练或调参。

## 公平基线

- 双专家：异常检出 50/57，正常误报 1/23。
- 正常专家 q90：异常检出 39/57，正常误报 0/23。

## 各强度的实际暴露与检出

- scale=0.25: 计划 20，实际暴露 19，零暴露 1；双专家 18，正常 q90 19。
- scale=0.5: 计划 20，实际暴露 19，零暴露 1；双专家 18，正常 q90 14。
- scale=0.75: 计划 20，实际暴露 19，零暴露 1；双专家 14，正常 q90 6。
- scale=1.0: 计划 20，实际暴露 0，零暴露 20；双专家 0，正常 q90 0。

## 双专家误报

```json
[
  {
    "episode_id": "source0:task4:episode1",
    "task_id": 4,
    "planned_scale": 1.0,
    "steps": 189,
    "active_steps": 0,
    "active_start_action": null,
    "mean_active_intended_target_norm_m": null,
    "mean_active_actual_translation_norm_m": null,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": false,
    "two_expert_any_alarm": true,
    "normal_q90_any_alarm": false,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": null,
    "active_state_counts": {},
    "active_ratio_z_median": null,
    "active_novelty_z_median": null,
    "active_normal_score_ratio_median": null
  }
]
```

## 双专家漏报

```json
[
  {
    "episode_id": "source1:task1:episode0",
    "task_id": 1,
    "planned_scale": 0.25,
    "steps": 112,
    "active_steps": 10,
    "active_start_action": 54,
    "mean_active_intended_target_norm_m": 0.009600926586426795,
    "mean_active_actual_translation_norm_m": 0.002037904604543361,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": true,
    "two_expert_any_alarm": false,
    "normal_q90_any_alarm": true,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": 9,
    "active_state_counts": {
      "ambiguous_overlap": 8,
      "normal": 1,
      "known_abnormal": 1
    },
    "active_ratio_z_median": 0.6524938260824595,
    "active_novelty_z_median": 38.09251942910907,
    "active_normal_score_ratio_median": 0.33233618604787546
  },
  {
    "episode_id": "source2:task1:episode0",
    "task_id": 1,
    "planned_scale": 0.5,
    "steps": 112,
    "active_steps": 10,
    "active_start_action": 54,
    "mean_active_intended_target_norm_m": 0.009795084258075803,
    "mean_active_actual_translation_norm_m": 0.0024022463159246985,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": false,
    "two_expert_any_alarm": false,
    "normal_q90_any_alarm": false,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": null,
    "active_state_counts": {
      "ambiguous_overlap": 9,
      "normal": 1
    },
    "active_ratio_z_median": 0.511736548333207,
    "active_novelty_z_median": 25.224603007900388,
    "active_normal_score_ratio_median": 0.21661717453651158
  },
  {
    "episode_id": "source3:task1:episode0",
    "task_id": 1,
    "planned_scale": 0.75,
    "steps": 110,
    "active_steps": 10,
    "active_start_action": 54,
    "mean_active_intended_target_norm_m": 0.009734092897269874,
    "mean_active_actual_translation_norm_m": 0.002777612366399819,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": false,
    "two_expert_any_alarm": false,
    "normal_q90_any_alarm": false,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": null,
    "active_state_counts": {
      "ambiguous_overlap": 9,
      "normal": 1
    },
    "active_ratio_z_median": 0.35106985697813875,
    "active_novelty_z_median": 13.150405237570624,
    "active_normal_score_ratio_median": 0.1235532032505384
  },
  {
    "episode_id": "source3:task1:episode1",
    "task_id": 1,
    "planned_scale": 0.75,
    "steps": 102,
    "active_steps": 10,
    "active_start_action": 29,
    "mean_active_intended_target_norm_m": 0.04728154949843884,
    "mean_active_actual_translation_norm_m": 0.008855614053436205,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": false,
    "two_expert_any_alarm": false,
    "normal_q90_any_alarm": false,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": null,
    "active_state_counts": {
      "ambiguous_overlap": 10
    },
    "active_ratio_z_median": 1.0433580881686568,
    "active_novelty_z_median": 107.72013559062603,
    "active_normal_score_ratio_median": 0.9203121499918381
  },
  {
    "episode_id": "source3:task2:episode0",
    "task_id": 2,
    "planned_scale": 0.75,
    "steps": 124,
    "active_steps": 10,
    "active_start_action": 53,
    "mean_active_intended_target_norm_m": 0.018431145790964366,
    "mean_active_actual_translation_norm_m": 0.0030079912143848165,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": false,
    "two_expert_any_alarm": false,
    "normal_q90_any_alarm": false,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": null,
    "active_state_counts": {
      "ambiguous_overlap": 10
    },
    "active_ratio_z_median": 1.7488931326505832,
    "active_novelty_z_median": 7.372034140994775,
    "active_normal_score_ratio_median": 0.07199313911025303
  },
  {
    "episode_id": "source3:task3:episode1",
    "task_id": 3,
    "planned_scale": 0.75,
    "steps": 92,
    "active_steps": 10,
    "active_start_action": 45,
    "mean_active_intended_target_norm_m": 0.014047227008268237,
    "mean_active_actual_translation_norm_m": 0.0029474513585073074,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": false,
    "two_expert_any_alarm": false,
    "normal_q90_any_alarm": false,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": null,
    "active_state_counts": {
      "ambiguous_overlap": 10
    },
    "active_ratio_z_median": 2.329834289024995,
    "active_novelty_z_median": 13.08864244823826,
    "active_normal_score_ratio_median": 0.09553080429098094
  },
  {
    "episode_id": "source3:task4:episode1",
    "task_id": 4,
    "planned_scale": 0.75,
    "steps": 145,
    "active_steps": 10,
    "active_start_action": 24,
    "mean_active_intended_target_norm_m": 0.028041224367916583,
    "mean_active_actual_translation_norm_m": 0.006152437242548478,
    "two_expert_detected_active": false,
    "normal_q90_detected_active": false,
    "two_expert_any_alarm": false,
    "normal_q90_any_alarm": false,
    "two_expert_delay_from_active": null,
    "normal_q90_delay_from_active": null,
    "active_state_counts": {
      "ambiguous_overlap": 7,
      "normal": 3
    },
    "active_ratio_z_median": 1.4220349330241049,
    "active_novelty_z_median": 14.800283482917678,
    "active_normal_score_ratio_median": 0.2094737634837319
  }
]
```
