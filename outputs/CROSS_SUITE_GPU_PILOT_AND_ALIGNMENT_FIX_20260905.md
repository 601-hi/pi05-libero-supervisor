# 跨 suite GPU pilot 与视觉时间对齐修正（2026-09-05）

## 本阶段完成内容

未触碰预留的 `libero_10`，并用固定环境 seed=30、固定 π0.5 扩散噪声 seed=2026090530 完成跨 suite 能力闸门：`libero_object` task 0/2/4 与 `libero_goal` task 0/2/4 均成功；`libero_90` task 0/9/19/29 为2/4成功。后续泛化评价必须区分策略本身不会任务与监督器漏检。

object task0、goal task0、libero_90 task9的scale=0.5配对扰动均完成且最终任务成功，说明检测标签是人工注入执行故障，不等同于任务失败。object/goal在扰动前动作0–69的 intended/executed action、EEF before/after逐元素一致；动作70的intended action与扰动前状态一致，executed action和执行后EEF首次分离。

## 关键时间对齐错误

代码审计确认图像在 `env.step(executed_action)` 之前保存，所以正确关系是

\[
a_t \longrightarrow I_t^{pre}\rightarrow I_{t+1}^{pre}.
\]

旧导出器却把 `I_{t-1}->I_t` 归给 `a_t`。此前全图低方向响应AUC=0.734、distal ROI AUC=0.549和“约一帧物理延迟”均降级为错位探索结果，不能作为正式证据。

保持非视觉字段逐元素不变后，重算训练集8649行/80 episode与校准集4341行/40 episode；每条轨迹仅最后一个动作因无后继图像而无效。

## 正确对齐后的结果

正常视觉专家只用正常数据，输入去除task one-hot的动作/本体条件，预测4×4 patch水平/垂直光流，不使用颜色、灰度背景或任务编号。

| seed20校准分数 | AUC |
|---|---:|
| 高预测残差 | 0.597 |
| 低定向响应 | **0.785** |
| 低余弦一致性 | 0.619 |

低定向响应在旧双专家 `ambiguous_overlap` 内AUC=0.670，与动力学似然比相关系数仅0.261，说明它包含互补证据。但“只在模糊区介入 + 3点中至少2点持续”的级联在正常episode误报预算0/1/2下均未新增episode检出：仍为26条，正常误报0。

4条漏检全部为scale=0.75，每条8–9/10个active step已处于模糊区，但全图视觉并未产生低响应证据。因此瓶颈不是门控位置，而是全局光流混合背景、机械臂和物体运动，对轻衰减缺少稳定局部可观测性。

## 决策与工程状态

1. 不通过放宽阈值美化结果；下一版采用腕部相机、多尺度时序、夹爪—物体相对运动/物体跟随。
2. AUC=0.785及级联结果仅是seed20 calibration diagnostics，需用新seed独立测试。
3. `libero_10`继续封存，直到视觉结构、融合规则和阈值冻结。
4. π0.5服务已关闭；端口8000和GPU compute process为空。远端新构件文件名含 `aligned`，防止与错位版本混用。
5. 脚本通过本地与远端Python语法检查，JSON/Markdown统一使用UTF-8。
