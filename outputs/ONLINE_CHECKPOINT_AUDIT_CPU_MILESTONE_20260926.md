# 在线安全检查点审计 CPU 里程碑（2026-09-26）

## 问题

复杂回退在 smoke 中使用了已知成功轨迹，并显式把历史状态标为 `recovery_safe=True`。正式在线循环不能这样做：动作没有报警不代表无接触，也不代表相机可观察、阶段可靠或机械臂远离奇异位形。若直接把所有正常历史状态作为回退点，会把后验成功信息和仿真便利条件带入控制器。

## Fail-closed 检查点契约

新增 `OnlineCheckpointRecorder`。每个候选历史点必须同时具备：

- 有限、维度正确的末端位置与四元数；
- 关节位置、速度、关节限位余量与Jacobian最小奇异值；
- 明确阶段及阶段置信度；
- 双相机传感器健康度；
- 所有在线监督事件明确正常所形成的响应可靠性；
- 来自独立传感器或显式 evaluation oracle 的无接触证据。

任一项未知都会写入拒绝原因，但 `recovery_safe=False`。尤其禁止用“没有报警”替代无接触证据。证据还保存来源字段，便于 LIBERO 的 MuJoCo oracle 与未来真机的力/力矩、电流、触觉或占用模块替换。

相机健康度只检查成像是否有限、非严重饱和并具有基本纹理，不声称目标物体可见。阶段跟踪器只在第一次闭合夹爪以前输出通用 `approach`；一旦闭合便永久弃权，不猜测抓取、运输或放置阶段。因此现阶段只建立保守的接触前回退点。

## 在线恢复准备器

新增 `OnlineRecoveryPreparer`，它不生成或执行动作，只回答当前异常是否具备合法恢复计划：

1. 至少需要两个独立准入的异常前检查点；
2. 最近的准入点作为6D汇合目标；
3. 按诊断类别、当前阶段和回退层级选择更早目标；
4. 只使用准入历史构造倒序参考；
5. 缺少更早目标、姿态或至少两个参考时明确 `not_ready`。

正式 `monitoring_main.py` 已接入只读审计：每步写入检查点准入结果；异常时写入准备状态、汇合动作编号、目标动作编号和参考数。默认不开 MuJoCo oracle，所以不会产生虚假的可执行恢复计划。显式 `--args.supervisor-checkpoint-oracle-enabled` 只用于 LIBERO evaluation，并在来源中标记 `mujoco_geometry_oracle_evaluation_only`。

## 验证

新增11项检查点/准备器测试，覆盖未知接触、缺失位姿、间隔门、在线字段别名、夹爪闭合后阶段弃权、相机健康度、全事件正常一致性、检查点不足与合法倒序计划。LIBERO Python 3.8语法和CLI参数检查通过，完整项目回归 **407项全部通过**。

## 下一次 GPU shadow smoke

下一次只验证数据契约，不开放恢复控制：task0、固定seed与固定π0.5采样噪声，开启 supervisor 与 checkpoint oracle，在action40注入确定性执行不匹配，但保持 `supervisor_intervention_enabled=False`。应检查：

- 策略轨迹不因监督器发生变化；
- action40前准入检查点数量和索引；
- 告警时 `recovery_preparation_ready` 是否为true；
- join/target索引严格早于故障；
- references全部来自 `recovery_safe=True`；
- oracle-only来源在日志中可见。

通过后才把 `HybridRecoveryExecutorAdapter` 接到在线物理执行分支，并进行 treatment 单臂 smoke；再通过后运行冻结的12对跨suite pilot。
