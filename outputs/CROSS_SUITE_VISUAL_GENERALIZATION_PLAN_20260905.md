# 跨 LIBERO 任务泛化与动作条件视觉监测计划（2026-09-05）

## 本阶段回答的问题

1. 动力学监督器离开 `libero_spatial` 后，能否在未见 suite 中识别相同的执行异常？
2. 当两个动力学专家给出 `ambiguous_overlap` 时，动作条件视觉变化能否增加有效证据而不学习任务背景捷径？

## 必须避免的混淆

当前动力学条件向量末尾包含 10 维 `task_onehot`。直接在其他 suite 复用会把新 suite 的 task 0 错当成 spatial task 0；`libero_90` 的 task 10–89 则无法编码。因此跨 suite 报告必须并列比较：

- 原 spatial 冻结专家：只作为“原系统直接迁移”的基线，并明确 task-ID 混淆；
- task-agnostic 专家：删除 task one-hot，仅用指令、关节/末端状态和历史响应，在 spatial 上重新训练后零样本迁移；
- 后续 universal-context 专家：加入 suite 身份或语言嵌入，但语言/任务上下文只进入条件动力学模型，不允许直接作为异常标签捷径。

## 数据切分

- development：`libero_object`、`libero_goal`、分层抽样 `libero_90`，seed 30；用于能力闸门和工程调试。
- calibration：只含 `libero_object` 与 `libero_goal`，seed 31；用于阈值/持久性规则选择。
- held-out-suite test：整个 `libero_10` 在冻结前不读，seed 32；用于真正的跨 suite 测试。
- `libero_90` 第一阶段仅作覆盖性探针，通过后再决定是否扩展到完整 90 任务。

`libero_10` 未出现在 development/calibration 中，因此可以作为整套 held-out 测试。若以后需要提前 smoke，必须把它标成独立的 engineering-smoke，且其结果不得再作为严格冻结测试指标。

## 指标分层

- 策略能力：任务成功率。它回答 π0.5 会不会做任务，不等于监督器正确率。
- 暴露率：`disturbed_steps > 0` 的 episode 比例。未触发扰动的提前结束 episode 按物理正常处理。
- episode detection：扰动有效期间是否触发；分 scale 报告。
- active-step recall：真正扰动步的召回率。
- inactive-step alarm rate：未扰动步报警率。
- normal-episode false alarm：正常 episode 是否出现任意报警。
- coverage/reject：四状态中 `ambiguous_overlap` 和 `unknown_abnormal` 比例，避免只报准确率掩盖拒识。

## 视觉模块 v3：动作条件视觉残差

不再直接训练“这张图像是否异常”的分类器。定义视觉编码为 `z_k`，上一动作块和本体条件为 `c_{k-1}`，只用正常训练轨迹学习：

\[
\widehat{\Delta z}_k=g_\theta(z_{k-1},c_{k-1}),\qquad
\Delta z_k=z_k-z_{k-1}.
\]

视觉异常证据来自预测变化与真实变化的差，而不是图像里是哪一种厨房或物体：

\[
r_k=\Delta z_k-\widehat{\Delta z}_k.
\]

主相机和腕部相机按空间 patch 分别计算残差，并使用 top-k patch 聚合，减少大量静态背景像素稀释局部物体/机械臂异常的风险。候选证据包括：归一化残差、预测/实际变化余弦、变化幅值比。阈值仅由正常 calibration 的 episode/block maximum 标定；视觉只在动力学 `ambiguous_overlap` 中验证，不覆盖明确的正常或异常判定。

## GPU 运行闸门

先对每个新 suite 跑少量正常 episode。若策略成功率太低，则该 suite 只用于“监督器在策略分布外的拒识”分析，不用于宣称执行故障检出泛化。通过后才运行 scale 0.25/0.50/0.75。冻结前不得打开 held-out-suite 的逐步结果。

## 当前状态

实验矩阵、命令生成器和静态泄漏审计器已在本地准备；远端 30227 当前拒绝连接，因此尚未启动采集。下一步是修正矩阵中的 held-out suite 重叠、完成 task-agnostic 特征分支及视觉残差训练/评分原型，然后在有卡实例恢复时执行小规模能力闸门。
