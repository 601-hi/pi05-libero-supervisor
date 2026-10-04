# VLA 监督主线框架 v0.1（2026-09-13）

## 本阶段目标

先闭合“规划—执行—监测—中断—重规划”软件主链路，不再等待所有专家达到最终精度后才搭系统。检测、时间融合和恢复处置被设计为独立模块，后续替换模型不会改变主控制流程。

## 已实现组件

| 组件 | 状态 | 职责 |
|---|---|---|
| `MonitorEvent` / `SupervisorDecision` | 完成 | 统一故障、置信度、证据和处置 schema |
| `InstructionGuard` | 完成 | 动作数值、突变、关节/速度/工作空间确定性约束 |
| `CallbackExecutionMonitor` | 接口完成 | 接入冻结双动力学专家的统一适配器 |
| `PolicyStallMonitor` | 原型完成 | 因果低进展与滚动夹爪反转 |
| `CallbackObjectResultMonitor` | 接口完成 | 接入对象结果视觉评分器 |
| `TemporalFusion` | 完成 | 按故障类型分别执行 k-of-m，不混合推进时间 |
| `RecoveryController` | 完成 | 重规划预算、冷却期、高优先级故障穿透、安全停止 |
| `SupervisorRuntime` | 完成 | 持有动作块、执行前安全门、执行后监测、清空剩余动作 |
| `Utf8JsonlLogger` | 完成 | 每次事件、证据、决策、动作块余量和恢复状态实时落盘 |
| 执行双专家模型 | 占位 | 接口存在，但最终模型必须由外部数据训练后接入 |
| 对象结果模型 | 占位 | 明确返回未实现、置信度 0，不伪造视觉证据 |

## 关键控制时序

```text
π0.5 返回 action chunk
          │
          ▼
SupervisorRuntime.install_chunk
          │
          ▼
InstructionGuard（执行前，即时）
          │ PASS
          ▼
env.step(action)
          │
          ▼
执行一致性 / 停滞 / 对象结果监测（执行后）
          │
          ▼
TemporalFusion（每个环境步只推进一次）
          │
          ├─ CONTINUE
          └─ STOP_AND_REPLAN
                 │
                 ├─ 清空剩余 action chunk
                 ├─ 使用当前观测重新请求 π0.5
                 └─ RecoveryController 冷却/预算/安全停止
```

指令安全门与执行后监测不共享时间计数。测试曾发现执行前与执行后各推进一次会导致 2-of-3 提前触发，现已修复：安全门正常通过不推进融合窗口，违规则立即处置。

## CPU 端到端验收

模拟场景：首个动作块连续无末端进展；`low_progress_run_threshold=2`，融合规则为 2-of-3。第三个执行步结束后：

- `POLICY_STALL` 成立；
- 剩余动作块被清空；
- `request_replan=true`；
- 重规划计数增加为 1；

## 真实 π0.5/LIBERO 回路验收

同日完成了固定环境 seed 与固定 π0.5 采样噪声的配对实验。显式 `test_only` 执行失配在 actions 10、11 提供两票，action 11 报警并清空当前 5-step chunk 的剩余 3 个动作；下一次推理由基线的 action 15 提前到 action 12。报警前动作与 EEF 状态最大差异均为 0，action 12 新动作与基线最大差异为 0.05295。基线和监督组均成功。

机器验收结果见 `outputs/SUPERVISOR_REAL_LOOP_VALIDATION_20260913.json`。本验收只证明在线控制链有效，不代表占位的执行与视觉模型已经完成。
- 新动作块安装后进入冷却；
- 三个产生正常进展的新步骤全部继续执行。

结果：`PASS`。模块测试 6/6 通过，另验证了立即拒绝危险指令、超过恢复预算安全停止、冷却抑制轻微重复报警、高优先级执行失配穿透冷却，以及执行/视觉回调共享同一融合接口。

## 数据与评测边界

- 框架测试使用合成状态，不读取 LIBERO 测试标签；
- LIBERO 已打开数据只能用于接口调试和机制发现；
- 最终执行专家由外部机器人数据训练；
- `NullObjectResultMonitor` 不把缺失视觉模型伪装成正常高置信判断；
- 未来每个模型必须附带权重、schema、阈值、数据清单和代码版本。

## 下一次 GPU 开机的唯一任务

把 `SupervisorRuntime` 接入服务器当前 `monitoring_main.py`：

1. π0.5 返回动作块后调用 `install_chunk`；
2. 每次 `env.step` 前调用 `next_action`；
3. `env.step` 后构造统一状态并调用 `observe_step`；
4. `request_replan` 时丢弃旧动作块，用当前观测立即重新推理；
5. 先使用可控的模拟事件分别验证 `INSTRUCTION_UNSAFE`、`EXECUTION_MISMATCH`、`POLICY_STALL`；
6. 保存基线与监督版本的配对 JSONL 和视频。

GPU 验收完成前不训练新专家、不打开新的最终测试任务。
