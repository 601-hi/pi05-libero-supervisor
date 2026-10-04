# 零样本数据治理与外部数据审计（2026-09-08）

## 结论

不能用最终测评域的数据训练模型、选择特征、归一化或确定阈值，再把所得结果称为广泛泛化。本项目从本阶段起将结果严格分为域内开发、正常数据适配、外部数据训练后的零样本评测三条轨道。

现有 `libero_spatial`、已查看的 `libero_object` 与 `libero_goal` 数据全部降级为开发或适配数据。`libero_90` 已进入旧开发规划，不作为主要零样本测试域。2026-09-08连接远端后，只按文件名、大小和修改时间检查了`/root/gpufree-data/libero-traces`，没有打开任何JSONL内容；`*libero_10*`匹配文件数为0。因此`libero_10`已从候选状态升级为正式封存测试套件。

## 外部数据不是越大越好

动力学专家所需的不是普通的 `(image, action)` 模仿学习样本，而是可建立因果对应的一步转移：

\[
(u_t, x_t, t_u, t_x) \longrightarrow (x_{t+1}, t_{x+1}).
\]

其中必须从一手资料确认动作究竟是绝对目标、增量还是速度，坐标系、单位、控制周期和夹爪语义也必须明确。RLDS 或 HDF5 只统一存储格式，不会自动统一物理语义。

## 候选优先级

### 1. DROID：首选真实机器人 schema pilot

DROID 使用统一的 Franka Panda 平台，覆盖大量真实场景。官方数据说明列出了 commanded Cartesian/joint position/velocity、gripper command，以及实际 Cartesian/joint/gripper state；机器人代码还暴露测得电机扭矩、控制器延迟和命令成功状态。它最接近本项目所需的“指令—实际响应”监督。

但第一步只能抽取极小 schema 样本，验证每个时间步究竟使用哪个控制动作、动作与反馈的时间关系。公开的 policy-learning transform 会把动作转换成绝对表示，不能把转换后的训练动作误当作原始控制器命令。

资料：<https://droid-dataset.github.io/>；<https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py>。

### 2. ManiSkill：不同物理引擎的仿真开发域

ManiSkill demonstrations 保存动作、每回合 control mode 与长度为 `T+1` 的环境状态，并能重放生成视觉观测。它适合检查公共 adapter 能否跨控制模式和跨 PhysX/MuJoCo 工作。

它不能证明真实世界泛化；使用 `--use-env-states` 强制状态重放时，也不能把得到的状态变化称为控制动作自然产生的动力学响应。

资料：<https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/datasets/demos.md>。

### 3. RH20T：力/力矩与接触安全辅助来源

RH20T 明确提供关节角、关节扭矩、TCP、夹爪信息、六维力/力矩、多视角 RGBD 及时间对齐 API，适合第二层中的电流/扭矩和接触安全分支。但当前一手说明只清楚确认夹爪 command，尚未确认机械臂控制命令是否完整公开，因此暂不进入双动力学专家训练。

资料：<https://rh20t.github.io/>；<https://github.com/rh20t/rh20t_api>。

### 4. Open X-Embodiment：逐数据集筛选目录

Open X-Embodiment 包含不同机器人和动作空间；官方明确动作维度可能表示 absolute、delta 或 velocity。它适合作为候选目录，不适合作为统一七维动作数据直接混训。每个子数据集必须单独通过 schema、单位与时间对齐审核。

资料：<https://github.com/google-deepmind/open_x_embodiment>。

### 5. BridgeData V2：暂缓

当前定位到的一手材料不足以严格确认所需 command/state 语义。它可以适合策略或视觉表示学习，但在完成原始字段审核前不得用于指令—响应动力学专家。

### 6. robomimic/robosuite：管线 sanity check

robomimic 格式具有 actions、obs、next_obs 和模拟器 state，适合验证 adapter。但它与 LIBERO 共享 robosuite/MuJoCo 家族，不能单独支撑广泛跨域泛化结论。

资料：<https://robomimic.github.io/docs/datasets/overview.html>。

## 已冻结的公共表示原则

公共 schema 保留 SI 单位原始量，并另外产生无量纲特征。每条样本必须具有数据集、机器人和控制器身份，但 dataset ID 不得作为检测器输入。动作向量维数或数值范围不能被用来猜测动作语义。

视觉若使用，必须匹配动作前图像与因果上最早的动作后图像。reward、success、模拟器 contact、物体真值位姿及 `oracle_only_*` 字段只能用于离线评价，不能进入部署特征。

## 下一步与停止条件

1. 在具备足够CPU和内存的无卡环境，通过 DROID 官方单episode或最小shard验证真实数值、单位和相邻步对齐；不下载完整数据。
2. 对真实样本输出缺失字段、数值范围、相邻响应相关性与对齐质量报告。
3. 获取一个ManiSkill极小轨迹测试第二个adapter，禁止使用强制env-state replay冒充自然动力学。
4. 在训练数据、adapter、模型和阈值冻结前，不运行或读取`libero_10`。

当前无需 GPU，也不应为了提高 LIBERO 分数继续查看封存测试域。

## DROID adapter 的当前状态

已实现依赖无关的 `droid_rlds_adapter.py`，按 `observation[t] + action_dict[t] -> observation[t+1]` 形成一步因果转移，最后一个没有后继状态的 step 必须丢弃。动作暂定选择 `cartesian_velocity` 六维加绝对 `gripper_position`，因为 DROID 采集代码的默认 action space 是 Cartesian velocity，并由 `create_action_dict` 同时生成其他等价表示。

但该 adapter 当前**不允许训练**。RLDS 公布 schema 没有逐步时间戳，只能暂按论文说明的 15 Hz 产生 `fixed_cadence_assumption`；同时 RLDS 字段没有逐样本附带原始 action-space、坐标系和单位声明。官方代码足以支持候选解释，却不足以证明每个 release 中每条数据都满足解释。因此 adapter 的默认 `command_semantics_verified=False`、`units_verified=False`，直接运行会被公共校验器拒绝。只有获得官方小样本并记录 release/version 证据后才能显式放行。

公共契约与 DROID adapter 共 11 项单元测试通过，包括：相邻 step 对齐、末步丢弃、缺字段拒绝、错误频率拒绝、未验证语义拒绝、未来时间戳拒绝和 oracle 泄漏拒绝。

随后直接从 DROID 官方 GCS bucket 匿名读取了两个小型元数据对象，而未下载 TFRecord：`dataset_info.json` 为760字节，`features.json` 为18665字节，并按官方对象 MD5 校验。元数据确认样本集版本1.0.0、100个episode以及上述 action/observation 维度，同时确认 measured joint velocity、motor torque和逐步timestamp不在RLDS observation中。自动网络审计脚本 `audit_droid_official_metadata.py` 会在元数据身份或字段结构变化时失败，并明确输出“仅通过metadata审计仍不允许训练”。

## 第二个公共 adapter：ManiSkill

已实现不依赖 ManiSkill 安装的解码后轨迹 adapter。它要求动作数组为 `[T,A]`、每个测量状态为 `[T+1,...]`，并要求调用者显式提供 `control_mode`、控制频率和机器人型号。当前只接受已登记的 `pd_ee_delta_pose`、`pd_joint_delta_pos`、`pd_joint_pos`，未知控制模式直接拒绝。

ManiSkill 的 object pose 等特权状态即使存在，也不能以部署特征名通过校验。该限制不禁止用 object pose 做离线标签，而是禁止 adapter 把它装进监督器输入。公共契约、DROID adapter和ManiSkill adapter共16项测试通过。这证明数据接口不依赖LIBERO/MuJoCo单一格式，但尚未证明模型跨域性能。

当前无卡服务器只有0.5核CPU、2 GB内存，且两个项目虚拟环境均未安装TensorFlow/TFDS；数据盘剩余约23 GB。为避免安装时OOM或下载无用的2 GB样本，本次停止在metadata和adapter阶段。该限制属于资源条件，不是DROID schema失败。
