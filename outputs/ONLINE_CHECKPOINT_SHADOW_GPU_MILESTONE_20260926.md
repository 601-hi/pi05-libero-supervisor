# 在线安全检查点影子验证（GPU 里程碑，2026-09-26）

## 目标

在不执行任何恢复动作的条件下，验证正式 LIBERO 在线脚本能否完成：正常执行证据审计、历史安全检查点保存、异常持续判决，以及只由已审计历史状态组成的恢复计划准备。实验使用固定环境种子与固定 π0.5 扩散噪声，确保基线与异常注入组可逐步配对。

## 暴露并修复的接口问题

1. 几何 oracle 在 `env.reset()` 之前绑定 `env.sim`。LIBERO reset 会重建仿真对象，旧对象的数据随后被释放，首步出现 `MjSim has no attribute data`。现改为每个 episode reset 后重新绑定当前 simulator。
2. plumbing-only 确定性测试监视器在异常窗口外输出 `NORMAL`，却报告零置信度，使 fail-closed 检查点门永久拒绝正常历史。现令其明确正常窗口使用配置置信度；该监视器仍标记 `test_only`，禁止用于正式评测。
3. 未实现的物体结果占位器输出 `NORMAL/confidence=0/status=not_implemented`，污染有效执行证据的最低置信度。现先剔除明确未实现的占位事件；若剔除后无有效证据则仍返回未知，因此没有放宽 fail-closed 原则。
4. 将笼统的 `checkpoint_buffer_gate_rejected` 拆成观测、响应、关节裕度、奇异性与间隔等具体原因，并在 step trace 中保存完整检查点证据，便于审计而非盲调阈值。

## 配对实验

- task：`libero_spatial` task 0
- 环境 seed：7
- π0.5 sampling-noise seed：20260926
- replan steps：5
- 基线：无测试事件
- 影子异常组：action 40–42 注入 `execution_mismatch`
- 两组均启用检查点审计与 evaluation-only MuJoCo 几何 oracle
- 两组均关闭 supervisor intervention

## 结果

- 两组均成功完成任务，均为 108 steps、22 inference calls。
- intended action、executed action、EEF position before/after 的逐步最大差值全部为 `0.0`；此前比较的 action chunks 亦完全一致。
- 异常组接纳 11 个检查点：`[0, 3, 6, 11, 14, 17, 20, 23, 26, 29, 32]`。
- 其余主要拒绝原因：22 次时间间隔门、74 次几何接触不清、70 次首次闭爪后的保守阶段弃权、3 次异常窗口内响应可靠性未知。
- 2-of-3 持续规则使 request-replan 在 action 41、42、43 出现。
- 三次都成功准备恢复计划：6D join action 32、rollback target action 29、2 条倒序回放参考。
- 干预关闭，因此上述计划只被记录，没有改变环境动作或任务结果。
- 完整单元/回归测试：`408 passed`。

## 解释与限制

本里程碑证明的是正式在线链路的因果隔离和数据契约正确，不是异常检测准确率，也不是恢复成功率。异常事件是显式的测试注入，接触清除来自仿真几何 oracle；二者均不能作为真实部署输入。下一阶段应打开恢复执行，但仍采用同起点配对实验，比较“无介入、检测但不介入、实际恢复介入”三组，验证安全退出、6D 汇合与倒序回放的物理效果。

