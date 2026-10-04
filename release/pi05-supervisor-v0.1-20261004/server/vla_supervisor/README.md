# VLA Supervisor 主线框架

该包把监测、时间融合与恢复处置明确分离：

1. `InstructionGuard`：动作和机器人状态的确定性安全门；
2. `CallbackExecutionMonitor`：现有或未来双动力学专家适配器；
3. `PolicyStallMonitor`：因果低进展与夹爪振荡；
4. `NullObjectResultMonitor` / `CallbackObjectResultMonitor`：对象结果接口；
5. `TemporalFusion`：按故障类型独立执行 k-of-m；
6. `RecoveryController`：重规划预算、冷却和安全停止；
7. `SupervisorRuntime`：管理动作块，中断剩余动作并输出 UTF-8 JSONL 决策记录。

视觉执行结果支线新增：

8. `background_motion`：无深度条件下的背景光流、RANSAC单应与可信度；
9. `causal_object_identity`：多候选的因果运动身份锁定；
10. `grasp_control_state`：闭爪、邻近、独立运动和末端同步的选择性抓持状态机；
11. `grasp_safety_fusion`：将局部视觉状态与独立动力学证据区分为漏抓、推动/碰撞、掉落和可能受阻。

选择性多模态主链路新增：

12. `FourStateExecutionMonitor`：保留双专家的`known_normal / known_abnormal / ambiguous_overlap / unknown`四态输出，不把模糊区硬压成二分类；
13. `AmbiguityGatedObjectMonitor`：仅当执行一致性为模糊或低支持时调用视觉，正常步骤不支付视觉计算成本，明确执行异常也不允许被视觉否决；
14. `create_selective_multimodal`：组装“指令安全门→执行四态判断→模糊区视觉诊断→k-of-m时序融合→停止与重规划”的主链路。

视觉返回的是结构化后果证据（可见性、夹爪—物体附着、物体是否进入目标、释放与掉落），而不是重复判断末端位移。默认时序策略仍为2-of-3；视觉触发后短暂保留两步跟踪窗口，避免只观察单帧。所有视觉调用与跳过原因都会进入UTF-8 JSONL事件证据。

当前对象结果模块默认使用 `NullObjectResultMonitor`，它明确返回“未实现”且置信度为零，绝不伪造视觉判断。下一 GPU 阶段把 `SupervisorRuntime` 接到 `monitoring_main.py` 的动作队列和策略客户端即可。

上述视觉状态机的接口和行为测试已经完成，但RGB到候选/夹爪掩码、候选—末端同步量仍需要冻结留出测试。未完成感知接线前，默认对象结果模块继续保持`NullObjectResultMonitor`，不得把结构代码误称为已部署视觉专家。
