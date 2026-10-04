# 自然失败采集阶段 A 结果（2026-09-11）

## 完整性

- 采集 10 个批次、120 个无注入 episode；
- 96 成功、24 自然失败；
- intended/executed action、关节/末端状态和结束记录完整；
- 外部相机与腕部相机均按步骤保存；
- 最终验收 `120/120`、错误 `0`、状态 `PASS`。

采集时发现 `monitoring_main.py` 的视频/sidecar 文件名未包含 suite。同 seed、task id、episode id、结果相同时，Spatial task 9 episode 0–4 被后运行的 LIBERO-90 task 9 覆盖。使用相同环境 seed 与固定扩散噪声复现 Spatial 5 条轨迹，596 个步骤的 intended action、executed action、EEF before/after、joint before/after 最大差异均为 0，结束结果及步数一致。恢复 sidecar 保存在隔离目录，验收器按动作索引选择唯一正确文件。未来采集必须把 suite 写入 artifact stem。

## 分任务结果

| suite/task | 任务角色 | 成功 | 失败 |
|---|---|---:|---:|
| Spatial 4 | 历史不稳定重点 | 15 | 0 |
| Spatial 0 | 稳定对照 | 10 | 0 |
| LIBERO-90 0 | 关闭上层抽屉 | 1 | 14 |
| Spatial 9 | 历史不稳定重点 | 14 | 1 |
| LIBERO-90 9 | 跨场景对照 | 5 | 0 |
| Spatial 3 | 历史不稳定重点 | 15 | 0 |
| Spatial 1 | 稳定对照 | 10 | 0 |
| LIBERO-90 29 | 黑碗放入上层抽屉 | 6 | 9 |
| Spatial 7 | 历史不稳定重点 | 15 | 0 |
| LIBERO-90 19 | 跨场景对照 | 5 | 0 |

## 初步机制结论

1. **LIBERO-90 task 0 是策略能力缺口。** 只有 1 条成功，无法可靠估计同任务正常分布。其失败包含长期停滞与长路径反复两种形态。执行响应可能完全正常，不能自动作为第二层动力学异常专家的正类。
2. **LIBERO-90 task 29 是同任务结果分叉集。** 6 成功、9 失败，背景和任务一致，适合研究物体是否被抓住/保持并最终进入抽屉。失败轨迹中夹爪命令反转可达 14–45 次，提示策略犹豫或反复抓取，但仍须结合视频确认。
3. **Spatial task 9 是视觉必要性案例。** 14 成功、1 失败；唯一失败的最大 jerk 和低进展连续段并不突出，仅靠运动统计可能漏检，适合检查掉落、放置关系或夹爪—物体时序。
4. **历史失败率不是固定属性。** Spatial 3/4/7 在 seed34 下均 15/15 成功，说明随机初始状态和扩散噪声会显著改变结果，不能围绕单次失败任务定制阈值。

## 质量候选

- 105 条 episode 拥有至少 5 条同任务成功参考；
- 其中自然失败 10 条，10 条全部进入质量复核池；
- 11 条成功轨迹进入低质量成功复核池；
- task 0 的 14 条失败因只有 1 条成功参考而明确标为不可条件化评估。

这里的 10/10 不是监督器检出率，而是包含最终步数、累计路径等事后指标的数据挖掘结果。下一步必须转为因果在线窗口，并在失败发生前计算首次可用预警时间。

## 下一步（无卡）

1. 逐帧比较 task 29 的成功/失败以及 Spatial task 9 的 14:1 配对；
2. 标注抓取成功、物体保持、掉落、目标关系、夹爪切换等事件；
3. 将失败拆分为策略无进展、抓取失败、掉落/错误放置、执行响应异常；
4. 在新标签下重算双动力学专家和 2-of-3/3-of-5 的首次预警；
5. 训练视觉结果专家时按 task/seed group 划分，禁止帧级随机切分；
6. 修复后续 artifact stem，使其包含 suite，防止跨套件同名覆盖。

机器可读结果位于：

- `outputs/remote_results/natural_failure_stage_a_seed34_validation_final.json`
- `outputs/remote_results/natural_failure_stage_a_seed34_outcome_audit.json`
- `outputs/remote_results/natural_failure_stage_a_seed34_quality_candidates.json`

## 冻结模型与因果信号诊断（2026-09-11 夜）

### 关系型双动力学专家

使用五个既有冻结专家零训练地评分 LIBERO-90 task 29 与 Spatial task 9。目标标签只用于最终报告，阈值仍来自独立源域正常校准集。所有时间规则（1-of-1、2-of-3、3-of-5、4-of-7、5-of-9、mean-3/5/10）对 10 条自然失败的检出均为 0。2-of-3 还误报 2/20 条成功轨迹，3-of-5 误报 1/20。

这并非完全没有排序信息：task 29 中，episode 内正似然比分数占比对失败的回顾性 AUC 为 0.889，最大分数 AUC 为 0.815；但源域 conformal 阈值远高于目标域分布。这同时暴露两点：自然任务失败多数不是执行响应失配，且源域到新场景存在明显分数尺度漂移。不能在这 15 条已打开目标轨迹上重新选阈值后再宣称泛化。

### 严格因果的策略停滞信号

把其余任务的 90 条成功轨迹作为独立校准集，在 task 29 上冻结测试：

| 在线信号 | 阈值（严格大于） | 成功误报 | 失败检出 |
|---|---:|---:|---:|
| 连续低进展步数 | 11 | 1/6 | 9/9 |
| 最近 40 步夹爪命令反转数 | 3 | 0/6 | 9/9 |
| 累积夹爪反转数 | 3 | 1/6 | 9/9 |
| 已执行步数 | 157 | 1/6 | 9/9 |

滚动夹爪反转的首次预警位于第 80–330 步，明显早于 400 步超时。它是目前可靠的“夹爪振荡/反复尝试”失败模式专家，但不是通用失败检测器：同一冻结方法在 LIBERO-90 task 0 只检出 3/14，在 Spatial task 9 的唯一失败上完全不触发。低进展信号在 task 0 检出 13/14，但把该任务唯一成功轨迹也误报；纯时长信号同样会误报慢速成功。因此后续融合必须保留故障类型，不能把所有信号压成一个无解释的异常概率。

### Spatial task 9 视觉边界案例

任务为“把木柜上的黑碗放到盘子上”。失败 episode 14 的机械臂稳定完成抓取、搬运、张开夹爪和抬升；末尾画面中碗位于盘边附近，却始终没有满足环境成功谓词。成功 episode 7 在碗进入目标区域且夹爪仍闭合时即结束。两者运动响应都合理，差别集中在对象与目标容器的精细空间/接触关系，因此应归入“视觉对象结果监督”，而不是动力学异常。

粗粒度全局光流无法可靠区分这一案例。下一视觉分支应显式估计目标碗、目标盘、夹爪三者的相对位置，以及放置后若干帧内的保持状态；不得直接读取仿真成功谓词作为部署输入。

### 采集命名缺陷修复

服务器当前 `monitoring_main.py` 的 artifact stem 已加入 `task_suite_name`。相同 seed、task id、episode id 的 Spatial task 9 与 LIBERO-90 task 9 现在分别以 `rollout_libero_spatial_...` 和 `rollout_libero_90_...` 开头，不再覆盖。安装前已保存 `monitoring_main.pre_suite_stem_20260911.py`，并在实际 LIBERO 导入环境下通过语法与两套件同编号不相等测试。旧数据文件名保持不变，由验收清单继续管理。

### 旧 v2 对照的证据限制

旧二分类 MLP 的精确权重未保存，只能由原训练代码重建。该重建在 Spatial task 9 上检出 1/1 失败、误报 3/14 成功，但重建阈值与历史冻结报告最大差 0.155，不能视为精确 checkpoint 回放，也不能据此断言旧 MLP 优于双专家。这个保存缺口应作为实验治理经验：今后必须同时保存模型权重、特征 schema、校准阈值、代码版本和数据清单。

新增机器可读结果：

- `outputs/remote_results/natural_failure_stage_a_seed34_relational_scores.npz`
- `outputs/remote_results/natural_failure_stage_a_seed34_temporal_rules.json`
- `outputs/remote_results/natural_failure_stage_a_seed34_task29_causal_stall.json`
- `outputs/remote_results/natural_failure_stage_a_seed34_task0_causal_stall.json`
- `outputs/remote_results/natural_failure_stage_a_seed34_spatial9_causal_stall.json`
- `outputs/remote_results/natural_failure_stage_a_seed34_spatial9_legacy_v2.json`
