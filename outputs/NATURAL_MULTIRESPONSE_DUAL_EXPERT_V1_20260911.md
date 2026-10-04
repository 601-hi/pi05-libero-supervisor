# 自然多响应双专家 v1（2026-09-11）

## 目标与标签语义

本阶段落实“使用新自然数据训练双专家”，但严格限定模型语义：它是弱 episode 标签监督的**任务失败风险双专家**，不是电机、碰撞或硬件故障专家。正常专家只使用成功 episode，异常专家只使用自然失败 episode；人工 scale 不再作为异常真值。

条件输入包含 7 维 π0.5 动作、7 关节位置、7 关节速度、相对初始末端位置、末端四元数、夹爪位置和上一时刻完整响应。预测目标为：

\[
y_t=[\Delta q_t(7),\Delta\dot q_t(7),\Delta x_t(3),\Delta\theta_t(3),\Delta g_t(2)]
\]

总计 22 维，已覆盖全部关节、末端平移/旋转和夹爪，而不再局限于末端平移。

两个 MLP 分别输出 22 维条件高斯均值与对数方差，通过正常/异常条件似然和双向显式 margin 联合训练：正常样本要求正常专家似然至少高出异常专家，异常样本反之。每条 episode 最多均匀抽取 96 步，避免跑满 400 步的失败因步骤更多而支配训练。

## 数据

- 自然 episode：120；
- 成功：96；
- 失败：24；
- 最终训练行：正常 8,911，异常 2,304；
- 失败仅来自三个任务族：LIBERO-90 task 0（14）、task 29（9）、Spatial task 9（1）。

这意味着异常分布的机制覆盖仍然狭窄，并且 Spatial task 9 只有一个正例。

## 防泄漏评估

采用整失败任务族留出：被测 suite/task 的全部成功和失败 episode 都不参加该折训练。模型不输入 task id，阈值只由训练任务中的成功 episode 确定。

| 留出任务 | 成功/失败 | mean AUC | q90 AUC | max AUC | 正分数占比 AUC | q90阈值误报/检出 |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-90 task 0 | 1/14 | 1.000 | 0.286 | 0.429 | 0.714 | 100% / 92.9% |
| LIBERO-90 task 29 | 6/9 | 0.630 | 0.981 | 0.722 | 0.657 | 100% / 100% |
| Spatial task 9 | 14/1 | 0.857 | 0.714 | 0.786 | 1.000 | 100% / 100% |

## 客观结论

1. **训练已完成，且模型确实覆盖全关节响应。** 最终 checkpoint 已保存并可由 CPU 加载。
2. **部分任务内排序有信号。** 特别是 task 29 的 q90 分数能较好排列成功和失败。
3. **绝对分数不可跨任务直接共用。** 三个留出任务的成功误报率均为 100%，表明状态/动作/响应尺度的域漂移大于现有阈值容忍度。当前模型不能部署。
4. **AUC 不等于可用检测率。** AUC 只要求同一留出任务内失败排在成功之前；若该任务的全部分数相对源域整体上移，固定阈值仍会把成功全部报警。
5. **episode 失败是弱标签。** 失败轨迹早期通常仍是正常响应，把整条轨迹都交给异常专家会污染异常分布；对象放置边界失败也未必包含动力学异常。

## 下一步

下一版不应继续盲目加 epoch，而应先完成两件事：

1. 将失败 episode 标为具体事件区间，只把夹爪振荡、持续停滞、抓取丢失、放置失败等事件附近窗口交给对应异常专家；
2. 把专家相对分数与任务/机构条件下的在线正常校准分离，使用少量无标签正常启动数据或 conformal 条件校准解决跨任务尺度漂移。

模型及结果：

- `outputs/remote_results/natural_multiresponse_dual_expert_v1/natural_multiresponse_dual_expert.pt`
- `outputs/remote_results/natural_multiresponse_dual_expert_v1/natural_multiresponse_dual_expert_report.json`
- `outputs/remote_results/natural_multiresponse_dual_expert_v1/natural_multiresponse_dual_expert_history.json`
- `outputs/remote_results/natural_multiresponse_dual_expert_v1/train.stdout`
