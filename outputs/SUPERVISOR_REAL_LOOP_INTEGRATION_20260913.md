# 监督器真实 π0.5/LIBERO 回路验收（2026-09-13）

## 结论

监督框架已接入真实 π0.5 WebSocket 推理与 LIBERO 环境循环。一次显式标记为 `test_only` 的执行失配注入使 `2-of-3` 规则在 action 11 报警，运行时清除了当前 action chunk 中尚未执行的 3 个动作，并在 action 12 使用最新观测重新请求 π0.5。配对基线与监督实验均成功完成任务。

这次实验验证的是控制链和因果时序，不是检测模型精度。执行双专家和对象结果视觉专家仍是占位接口，不能把本结果解释成最终监督器已经具有良好异常识别率。

## 配对实验控制

- 任务：`libero_spatial` task 0，单 episode；
- 环境 seed：34；
- π0.5 sampling-noise seed：2026091134；
- 每次通常执行 5 个 chunk 动作；
- 基线关闭监督器；
- 监督组只在 action 10–12 注入 `execution_mismatch` 测试事件；
- 注入只影响监督事件，不修改动作、观测或环境动力学。

因此 action 0–11 的 intended action、executed action、EEF 执行前位置和执行后位置逐项最大差异均为 0。这排除了“报警前两次运行本来就随机分叉”的解释。

## 关键时序

| action | 监测事件 | 决策 | 动作块结果 |
|---:|---|---|---|
| 9 | normal | continue | 正常执行 |
| 10 | execution mismatch（第 1 票） | continue | chunk 2 还剩 4 个动作 |
| 11 | execution mismatch（第 2 票） | stop and replan | 清空 chunk 2 剩余 3 个动作 |
| 12 | execution mismatch（测试区末步） | cooldown 中 continue | 新 chunk 3 的第 1 个动作 |
| 13 | normal | continue | 继续新 chunk |

基线 chunk 2 对应 actions `[10, 11, 12, 13, 14]`；监督组只执行 `[10, 11]`。基线前五次推理发生于 `[0, 5, 10, 15, 20]`，监督组变为 `[0, 5, 10, 12, 17]`。action 12 两组 intended action 的最大差异为 `0.05295043535602084`，证明 action 12 不是继续取旧队列，而是新推理的结果。

## 结果

- 自动验收：`PASS`；
- 报警原因：`execution_mismatch`；
- 报警数：1；重规划数：1；
- 基线：成功，108 actions，22 inference calls；
- 监督组：成功，74 actions，16 inference calls；
- 两组报警前状态与动作完全一致；
- EGL 析构警告发生在 episode 已成功并完整写盘之后，不影响本次结论。

机器可读证据见 `outputs/SUPERVISOR_REAL_LOOP_VALIDATION_20260913.json`，复验脚本见 `cross_suite_generalization/validate_supervisor_real_loop.py`。

## 阶段边界与下一步

框架主链已闭合：规划、动作块所有权、执行、监测、时间融合、中断、重规划和 UTF-8 日志均能在真实循环中工作。下一阶段应停止继续用人工事件证明管线，转而把冻结的真实执行监测器接入 `CallbackExecutionMonitor`，并以外部训练数据、独立正常校准数据和未参与训练的测试数据评价检出、误报和恢复收益。
