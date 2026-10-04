# 端到端监督闭环CPU准备阶段（2026-09-18）

## 阶段目标

停止继续局部优化视觉候选，优先完成“动作指令检查—执行四状态监测—选择性视觉—干预编排—动作块中断—重规划”的最小可用闭环。该阶段不运行π0.5、LIBERO或SAM2，只完成CPU部署接口、真实模型加载和因果时序测试。

## 已存在且复核通过的主链

`SupervisorRuntime`已经真实拥有动作块并在报警时清空剩余动作；既有配对实验曾验证action 11报警后旧chunk剩余3步被删除，action 12从最新观测重新调用π0.5。指令安全门、k-of-m时间融合、重规划预算、冷却期、UTF-8日志和视觉模糊路由均已存在。本阶段没有重复重写这些部件。

## 新增恢复干预编排

新增`vla_supervisor/intervention.py`，输出可审计而不直接执行机器人的恢复指令：

- `nominal`：策略块保持5步；
- `cautious_continue`：模糊证据不直接报警，但把剩余动作块截短到1步；
- `replan_short`：未执行的危险指令直接丢弃并短视野重规划；
- `hold_then_replan`：平移和旋转清零、保持夹爪后重规划；
- `retreat_then_replan`：仅在独立诊断为固定障碍或卡住时，沿最近平移指令反向执行最多2步、幅值0.15、旋转清零并保持夹爪；
- `safe_stop`：恢复预算耗尽后停止。

无法估计最近接近方向时，撤退自动降级为保持。桥接动作通过`stage_intervention_bridge()`进入同一个受InstructionGuard约束的动作队列；耗尽后`needs_policy_replan`变为真，新动作块安装后退出待重规划并进入冷却期。

## 真实双专家四状态部署

新增`CalibratedFourStateConditionalScorer`，实际加载5个冻结条件双专家和source-only可靠性校准。输出严格保留`known_normal / known_abnormal / ambiguous_overlap / unknown`，不压成二分类。传感器质量检查有限数、末端单步跳变和关节速度；命令历史不足或10步序列窗口未填满时强制unknown。

部署中发现并修复两项真实兼容问题：

1. 训练脚本使用裸模块导入，作为在线包加载时报`ModuleNotFoundError`。改为优先包相对导入，直接执行脚本时保留兼容回退。
2. LIBERO Python 3.8中的旧PyTorch不支持`torch.load(..., weights_only=False)`。对可信本地checkpoint加入旧版本回退。修复后5个模型已在LIBERO自己的虚拟环境中成功加载。

真实CPU冒烟中，前5步为命令历史预热，之后继续积累10步序列证据；窗口未满期间不会报警。模型不需要GPU运行。

## LIBERO候选在线脚本

从服务器生产版派生`outputs/monitoring_main_intervention_candidate.py`，未覆盖生产文件。新增：

- `supervisor_execution_mode=placeholder|frozen_four_state`；
- 5模型与校准文件严格存在性检查；
- `supervisor_intervention_enabled`默认关闭；
- 动态选择1步或5步策略块；
- 恢复桥接动作优先执行；
- `policy_chunk/recovery_bridge`动作来源日志；
- episode级干预计数。

默认参数保持历史行为不变。真实视觉在线适配尚未接入，工厂状态明确标为`explicit_null_until_online_visual_adapter`，不得声称视觉专家已在线部署。

## 验证

- 恢复干预定向测试：7/7；
- 四状态与真实工厂测试合计：12/12；
- 全量服务器回归：184/184；
- 候选LIBERO脚本Python编译和CLI参数检查通过；
- 5模型在LIBERO Python 3.8环境实际加载通过。

## 下一次GPU实验顺序

1. 备份生产`monitoring_main.py`并安装候选版；
2. 启动π0.5服务；
3. 固定环境seed和sampling-noise seed跑原始基线；
4. 跑`frozen_four_state`但关闭恢复干预，测纯监测副作用；
5. 开启恢复干预，验证模糊缩短动作块、报警桥接动作和重规划时序；
6. 先做1任务1条冒烟，再扩展多任务；
7. 在线视觉仍为null，本轮不能评价多模态收益。

评价同时报告成功率、误干预、重规划次数、动作数、推理次数、unknown/ambiguous比例和恢复后成功率，不能只报告LIBERO成功率。
