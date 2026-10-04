# Recovery v1 正式在线接线（CPU，2026-09-26）

目标被主动收缩为“证明回退器可用且带来净收益”，不在第一版追求全部异常、任务和机器人上的最优恢复。

本阶段增加环境无关的 `OnlineRollbackCursor`，把已经准备好的 6D join 与安全历史倒序参考转换为有界 OSC_POSE 小步。游标本身不声称动作安全：正式 LIBERO 循环对每个候选先做 evaluation-only MuJoCo 关节扫掠，再送入原有 instruction guard；任何无效值、非正碰撞间隙、缺失姿态、步数耗尽或安全门拒绝都会进入 safe-stop。回放完成后控制端口转入 replan-pending，之后才允许 π0.5 重新观察并生成动作。

新增参数 `--args.supervisor-rollback-enabled`，默认关闭；它必须同时满足 intervention 与 checkpoint oracle 显式开启，避免误把仿真特权几何当成部署传感器。完整回归为 `411 passed`。

下一步只需 GPU 最小验证：固定 task0、seed 和扩散噪声，运行 control、shadow、treatment 三臂。首先确认 treatment 实际经历 complex_rollback 并到达重新规划；随后只看相对 control 是否挽救失败且没有新增危险。得到至少一个严格配对的有效恢复案例后冻结 recovery v1，转入视觉结果监督和最终 GitHub 集成。

