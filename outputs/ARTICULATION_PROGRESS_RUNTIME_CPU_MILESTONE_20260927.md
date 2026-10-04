# 机构关系进展运行时 CPU 集成节点（2026-09-27）

## 已完成链路

GPU阶段冻结的抽屉进展逻辑已经从一次性分析脚本重构为 `vla_supervisor/articulation_progress.py` 中的通用组件：

```text
perception provider
  -> ArticulationMeasurement
  -> ArticulationProgressWatchdog
  -> ArticulationProgressMonitor
  -> MonitorEvent
  -> TemporalFusion
  -> InterventionPlanner
  -> restore_goal_relation + hold_then_replan
```

组件不读取 LIBERO task ID、reward、success predicate 或仿真对象关节。感知提供者必须显式提交：面积代理量、观测置信度、身份是否可靠、是否被冻结标定覆盖、关系 ID、谓词和证据来源。

## 权限边界

- `unknown/observing/progress` 只记录，不改变动作队列；
- `goal_no_progress` 映射为 `OBJECT_FAILURE + goal_relation_progress_stalled`；
- `goal_regression` 映射为 `OBJECT_FAILURE + goal_relation_not_satisfied`；
- shadow 模式把潜在 `OBJECT_FAILURE` 降为 `OBJECT_AMBIGUOUS`，不申请重规划；
- control 模式仍须经过现有 `TemporalFusion`、`RecoveryController` 和重规划预算；
- 默认恢复是保持后重规划，不因视觉进展停滞直接执行盲目回退；
- 该组件没有 `goal_satisfied` 输出，因此不能宣布任务完成。

## 真实轨迹回放

新运行时组件重新回放 GPU 阶段的同一组冻结 SAM2 面积序列，结果与原分析脚本逐项一致：

- 开发成功 seed37 ep3：`continue`；
- seed35 四条留出失败：均为 `goal_no_progress`，触发于动作 80、80、80、97；
- 组件不会把成功轨迹的末端状态声明为完成。

机器可读结果：`outputs/task0_close_relation_20260927/runtime_component_replay.json`。

## 测试

新增测试覆盖：

- 先改善后平台化；
- 改善后退化；
- 不可观测、低置信和标定不匹配时弃权；
- actionable 事件锁存；
- shadow/control 权限隔离；
- 在真实 `SupervisorRuntime` 中清空动作块、保持、申请重规划并路由到 `restore_goal_relation`；
- shadow 运行时保留原策略队列。

服务器完整测试结果：`465 passed in 8.51s`。

## 下一次 GPU 冻结实验

下一次不再调整 task 0 阈值，而应执行严格配对在线试验：

1. 同一服务、同一环境 seed、同一 π0.5 sampling noise；
2. baseline 禁用关系进展控制，control 启用；
3. 两组都记录固定/腕部图像、关系测量、事件、恢复目标、重规划提示和动作；
4. 第一目标是确认自然失败能在约 80–97 步清空错误动作块并请求“恢复关闭关系”；
5. 统计 rescued、harmed、preserved success、unresolved failure；
6. 若重规划仍重复原行为，问题归入上层指令纠偏，不继续修改视觉阈值；
7. task 0 闭环仅证明纵向链路，随后必须转向未见 `Open/Close` 场景验证关系族迁移。
