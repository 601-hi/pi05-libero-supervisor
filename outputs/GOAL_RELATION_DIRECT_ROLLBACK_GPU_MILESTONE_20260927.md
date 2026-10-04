# 目标关系异常直连安全回退：GPU闭环节点（2026-09-27）

## 冻结实验

- suite/task：`libero_90/task0 Close(top_drawer)`；
- 5个固定初始状态；
- seed 7，显式采样噪声2026092700；
- baseline与control共享同一π0.5服务生命周期；
- control只允许`articulation_relation_progress`进入控制融合；
- 恢复文字提示关闭；
- 碰撞oracle、复杂回退和回退后策略预算刷新开启。

视觉sidecar启动时因默认缓存目录错误尝试联网；改为显式`HF_HOME=/root/gpufree-data/hf-cache`及离线模式后，加载的是原有冻结权重，没有下载或替换模型。

## 因果门

5/5回合在首次介入前严格一致。四条失败都在action82开始第一个复杂回退动作；唯一成功回合没有介入，baseline/control均成功且均为94步。

## 端到端结果

- rescued：0；
- harmed：0；
- preserved：1；
- unresolved：4。

这不是“回退后重规划失败”。四条失败中没有一条完成回退并把控制权交还π0.5，因此没有发生任何`policy_budget_reset`；预算逻辑正确地没有误刷新。

## 四条失败的回退终点

1. episode 0：无需join，直接进入replay；replay 160步耗尽，`replay_step_budget_exhausted`。
2. episode 1：执行480个回退动作后仍处于join；其中join接受184步，另有候选拒绝/策略重新接管交错，全局回退预算耗尽。
3. episode 2：join 320步耗尽，`join_step_budget_exhausted`。
4. episode 4：join 320步耗尽，`join_step_budget_exhausted`。

唯一成功episode 3：无回退、无预算刷新、成功保留。

机器可读审计：`outputs/goal_relation_direct_rollback_gpu_20260927/direct_rollback_causal_audit.json`。

## 当前结论

视觉检测→控制隔离→直接调用复杂回退已经真实贯通，而且没有伤害冻结成功回合；但现有回退状态机在这四个视觉触发状态上无法在有界预算内收敛。因此下一问题位于回退器内部，不能继续调整视觉阈值，也不能评价回退后的π0.5重规划能力。

下一CPU阶段必须分别审计：

- join目标与当前位置的距离是否单调下降；
- 为什么episode 0的160步replay没有跨过4个冻结参考点；
- 候选被碰撞门拒绝后，external recovery与策略队列为何出现交错；
- 总回退预算是否错误地跨多次恢复累计；
- 到达阈值是否与OSC单步误差下限相容。

在这些问题解释清楚前，不增加预算来掩盖不收敛。
