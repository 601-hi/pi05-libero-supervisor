# 故障快照捕获与重放 CPU 节点（2026-10-03）

## 动机

v24f/g/h 虽使用同一环境种子与 π0.5 扩散噪声，但从动作0起已经分叉。固定扩散噪声只保证相同观测下的条件采样一致，不能冻结环境图像和接触动力学。继续从 episode 起点复测会把上游轨迹差异错误归因给回退器。

## 快照边界

快照建立在监督器已经确认异常、成功构建 `RecoveryPreparationDecision`，但尚未执行第一个恢复动作的时刻。保存：

- `MjSimState.flatten()`完整主状态；
- mocap位置与四元数；
- actuator `ctrl`；
- 故障末端位置和夹爪命令；
- JOIN历史状态、倒序REPLAY关节/末端参考及目标索引；
- task、episode、action、诊断和回退层级元数据。

文件使用压缩NPZ数组和UTF-8 JSON，不使用pickle。加载时校验schema、`nq`、`nv`、flattened state长度及辅助数组形状，避免跨机器人模型误用。

## 在线接入

- `--args.supervisor-fault-snapshot-path`：第一次成功启动恢复时捕获，不被后续更深层恢复覆盖；模板支持task、episode和action编号。
- `--args.supervisor-fault-snapshot-load-path`：构建同一任务环境后恢复故障状态，跳过初始稳定等待与故障前策略前缀，恢复原动作编号和夹爪状态，直接把第三层恢复置为ACTIVE。
- 加载后仍运行真实JOINT_POSITION控制、MuJoCo动力学和分阶段碰撞门；快照只固定实验起点，不直接写入未来状态。

## 验证

- 远端相关回归：`151 passed`。
- 真实LIBERO/MuJoCo smoke：先运行轨迹，保存快照；再执行三个不同动作产生扰动；最后恢复。
- flattened state、qpos、qvel逐位一致；末端MuJoCo site最大误差`0.0 m`；恢复计划严格相等。
- 保存哈希与扰动哈希不同，恢复哈希与保存哈希完全相同：`7ec793...e4f20`。
- 中文UTF-8元数据往返正常。

证据：`outputs/fault_snapshot_cpu_20261003/fault_snapshot_smoke.json`与同目录NPZ快照。

下一GPU实验分两步：先运行一次完整系统捕获真实失败快照；随后所有碰撞门/回退器版本从同一NPZ直接启动。这样才能严格评价分阶段碰撞门是否让v24f类状态跨过10 cm/18层并进入重规划。
