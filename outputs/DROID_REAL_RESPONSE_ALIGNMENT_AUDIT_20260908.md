# DROID 真实机器人指令—响应对齐审计（2026-09-08）

## 目的与边界

本阶段不使用 LIBERO 测评数据训练，也不启动 π0.5。目标是检验 DROID 是否能作为正常动力学专家的外部真实机器人训练源。审计仅使用命名动作字段与机器人本体状态；未查看或导出图像，未使用语言、奖励或成功标签。

预先固定分片索引 6、9、19、29，以覆盖数据集早期、中段和后段，同时控制下载规模。四个官方分片共 56,791,212 bytes，含 6 个 episode、860 步；URL、字节数与 SHA-256 记录于 `outputs/remote_results/droid_schema_pilot/preregistered_shards_manifest.json`。

## 为什么不能直接令动作 t 对应状态 t→t+1

DROID 论文给出约 15 Hz 采集频率，但公开 RLDS 1.0.0 没有逐步实测时间戳和控制器延迟。真实链路包含通信、命令排队、控制器滤波、伺服和机械响应。对速度命令 `u_t`，本阶段比较：

`u_t` 与 `(x_{t+l+1} - x_{t+l}) / dt`，其中正 lag `l` 表示测得响应晚于指令。

按 episode 单独差分，禁止跨 episode 边界拼接；旋转欧拉角先沿时间解缠，避免 ±π 跳变制造假速度。

## 结果

汇总最佳响应 lag：

| 模态 | lag 0 相关系数 | 最佳 lag | 最佳相关系数 | 最佳 lag 的 episode 分布 |
|---|---:|---:|---:|---|
| EEF 平移 | 0.781 | +2 | 0.830 | 5 条为 +2，1 条落在搜索上界 +5 |
| EEF 旋转 | 0.779 | +1 | 0.824 | 3 条 +1，2 条 +2，1 条 +5 |
| 关节 | 0.868 | +1 | 0.909 | 2 条 +1，3 条 +2，1 条 +5 |

其中第 4 条、50 步 episode 并非指令缺乏激励：三个命令 RMS 与其他轨迹同量级；但实际响应 RMS 约低一个数量级，三种模态在全部候选 lag 上的相关性都接近零。它可能来自停机/同步问题，也可能与未公开的 episode 原始 action space 有关。现有证据不能区分，因而不能把它删除或强行归入某个固定延迟。

## 对监督器设计的影响

1. **否定零延迟理想化。** 正常专家不能把 `u_t→Δx_t` 视作唯一正确映射，否则真实机器的正常控制延迟会被当成异常。
2. **暂不写死 +2。** 六条轨迹只足以揭示问题，不足以证明所有设备和 episode 都共享同一时滞。
3. **模型应接收动作历史。** 更合理的是预测 `p(Δx_t | x_t, u_t, u_{t-1}, …, controller/domain)`，或对经过正常数据校准的 lag 做边缘化；固定 lag 仅作为消融。
4. **加入可辨识性/质量门。** 当指令有激励但全部模态响应几乎静止、且控制模式未知时，样本不能静默进入“正常”训练集，应进入拒识/人工审计状态。
5. **不同物理通道可有不同 lag。** 平移、旋转与关节的最优 lag 不完全相同，说明不能用一个全局标量响应模型覆盖全部通道。

## 代码与验证

- `audit_droid_episode_values.py`：单 episode 数值与目标跟踪审计。
- `audit_droid_multi_episode.py`：多分片、多 episode、分模态 lag 审计。
- `droid_rlds_adapter.py`：新增显式 `response_lag_steps` 与 `timing_alignment_verified`；默认不猜测、不允许训练。
- `common_dynamics_transition.py`：允许历史指令对应稍后的因果响应，但仍拒绝晚于响应的未来指令。
- 公共契约与 adapters 共 18 项测试通过。

## 当前决策

DROID 仍是有价值的真实机器人外部证据源，但暂不允许直接训练正常专家。下一步需要扩大不依赖任务标签的时序审计，并找到原始 action-space/采集同步信息，或把未知控制模式作为显式域变量与拒识条件。完整原始数值报告保存在 `outputs/remote_results/droid_schema_pilot/droid_multi_episode_alignment_v2.json`。

## 后续实现：因果延迟似然银行

本阶段随后实现了透明的延迟响应基线。对当前响应 `r_t`，部署端只允许历史命令参与：

`p(r_t | history) = sum_l w_l N(r_t; alpha_l u_{t-l}, sigma_l^2 I)`，其中 `l >= 0`。

诊断仍可计算负 lag，用于发现记录反向错位；但拟合函数会显式拒绝负 lag，防止未来命令泄漏。对三个模态，质量门均接受5/6个episode并拒绝同一条低响应、低相关轨迹。基于这5条轨迹的权重峰值分别为：平移+2帧、旋转+1帧、关节+1帧；相邻lag保留非零权重，以表达控制延迟的不确定性和遥操作命令自相关。

该组件的定位是时序对齐与可解释基线，不是最终正常专家。最终模型仍应条件化于状态、构型、速度历史和可观测运行阶段。新增合成延迟恢复、匹配/异常似然、质量拒识和非因果lag拒绝测试后，测试总数为22项，全部通过。完整结果为 `outputs/remote_results/droid_schema_pilot/droid_multi_episode_lagbank_v2.json`。

## 官方策略动作语义与绝对目标复核

DROID 官方策略学习代码的 `droid_dataset_transform` 使用 `action_dict.cartesian_position` 构造动作，而不是直接使用 `cartesian_velocity`；下游 `robomimic_transform` 又取 `actions[1:]`。这是一种模仿学习标签构造，不能直接当作机器人动力学的命令—响应时间定义。

因此 v3 审计另外比较绝对 EEF/关节/夹爪目标与未来测量状态。五条可辨识轨迹的 EEF 和关节目标通常在约4至5帧后误差最小；此前的问题轨迹却在lag 0误差最小，且后续不收敛。换成绝对位置目标并没有恢复该轨迹的执行跟踪关系，质量拒识结论保持不变。

这一区分必须长期保留：策略监督中的 `observation_t -> demonstration action label` 对齐，不等于系统辨识中的 `issued command -> measured plant response` 对齐。v3完整报告为 `outputs/remote_results/droid_schema_pilot/droid_multi_episode_lagbank_v3.json`。
