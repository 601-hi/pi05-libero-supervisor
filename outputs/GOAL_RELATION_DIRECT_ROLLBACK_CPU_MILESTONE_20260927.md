# 目标关系异常直连安全回退：CPU节点（2026-09-27）

## 目的

纠正上一轮“视觉告警后原地保持并附加文字提示重规划”的临时验证路线。正式主线改为：

`goal_no_progress → 停止旧动作块 → 最近可靠历史点回退 → 正常原任务重新观察与规划`。

文字提示不再是恢复前提；GPU验证时将保持 `supervisor_recovery_prompt_enabled=false`。

## 修复的跨层契约

视觉机构监督输出的诊断是 `goal_relation_progress_stalled`。旧检查点选择器只允许该诊断回到 `transport/place/goal_relation` 阶段，但保守历史记录器在首次夹爪闭合后停止授信，实际保存的主要是 `observe/approach/pregrasp`。这会造成“存在安全点却无法生成回退计划”。

现在目标关系停滞或未满足可以降级选择任一通过安全准入的历史阶段。它只扩大“可作为导航参考的历史点”集合，不绕过回退动作的指令门、逐候选扫掠碰撞门、关节/奇异性门和执行后响应监测。

## 执行预算修复

旧循环补偿回退动作占用的步数，但回退完成后的π0.5仍继承失败前剩余的任务步数。后期发生故障时，可能刚完成回退就撞上原400步上限。

新增 `RecoveryAwarePolicyBudget`，把预算分为：

1. 当前策略任务窗口；
2. 独立有界的join/replay回退预算；
3. 回退完成后的新策略任务窗口。

只有回退状态机确认 `REPLAN_PENDING`、即安全回退确实完成并准备交还π0.5时，策略计数才清零并获得一个完整新窗口。回退尚在执行时即使旧策略窗口已耗尽也允许完成；回退被拒绝、未完成、普通重规划或文字提示均不会刷新预算。最大恢复次数和回退步数上限仍保留，因此不会无限续跑。

日志新增：

- `policy_budget_reset`事件；
- `new_policy_budget_steps`；
- `policy_budget_reset_count`；
- episode_end中的`policy_actions_since_budget_reset`。

## 冻结真实轨迹验证

使用此前同服务严格配对的自然失败控制轨迹，只替换诊断类型为 `goal_relation_progress_stalled`，不重新采样数据。回退准备结果：

- failure action：54；
- 历史证据行：55；
- ready：true；
- 汇合点：action 21；
- 回退终点：action 15；
- 倒序参考点：4。

机器可读结果：`outputs/goal_relation_direct_rollback_cpu_audit_20260927.json`。

## 验证

- 相关单元/集成测试：113项通过；
- 服务器全量回归：480项通过；
- 监控程序语法与CLI参数检查通过。

## 下一步 GPU 门

同一π0.5服务生命周期内运行冻结的五回合配对：baseline保持不变，control仅允许视觉关系源控制，开启碰撞oracle与复杂回退，关闭恢复文字提示。必须审计首次视觉介入前严格一致，并报告 `rescued/harmed/preserved/unresolved`。回退完成后检查预算重置事件及完整新策略窗口；任何原成功回合受损立即停止。
