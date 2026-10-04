# 在线任务后果监督链路里程碑（2026-09-25）

## 本阶段目标

将已经完成的任务进展/语义后果判断从独立模块接入在线监督运行时，形成可审计的链路：

`校准证据 -> 时间确认 -> 诊断状态 -> 恢复目标 -> 有界桥接动作 -> π0.5 重规划提示`

本阶段只完成接口闭环和安全语义，不宣称当前已有可部署的通用视觉证据模型。

## 审计发现

1. `SemanticConsequenceMonitor` 已能输出 `no_progress`、`wrong_object`、`object_lost`，并映射到通用诊断状态，但 `SupervisorRuntime` 从未调用它。
2. 在线 JSONL 已能记录全部事件、融合决策、恢复目标和干预指令，因此日志字段不是瓶颈。
3. 当前生产视觉模式使用 `BackgroundConsequenceEvidenceScorer(candidate_provider=None)`，只能检查背景健康度，不能提供目标物体或任务进展证据。
4. `monitoring_main.py` 已调用 `supervisor.build_policy_prompt(...)`，但运行时此前没有该方法；真正开启恢复提示会在第一次重规划查询时报错。

## 实现

### 1. 条件监测器接口

`SupervisorRuntime` 新增可选 `conditioned_monitors`。调用顺序为：

1. 执行一致性/停滞监测器；
2. 原有接触/物体监测器；
3. 任务后果监测器读取成对图像及上游事件；
4. 所有事件共同进入时间融合与恢复规划。

旧构造函数保持兼容，重置 episode 和完成重规划时也会重置新增监测器。

### 2. 校准语义后果适配器

新增 `CalibratedSemanticConsequenceAdapter`。证据源必须返回结构化的
`SemanticConsequenceEvidence`，不能直接发机器人指令。适配器执行：

- 信息防火墙：历史中仅保留动作序号和指令，不向视觉证据源泄露实测状态、奖励、成功标签或扰动标签；
- 可观测性与校准门控；
- 最高假设分数和假设间隔门控；
- `no_progress` 的连续两次确认；
- 默认 shadow 模式；
- 未校准/不可见的 `unknown` 仅写日志并输出 `NORMAL`，不能缩短动作块；
- 已校准但互相冲突的 `ambiguous` 才能请求更多证据；
- 只有显式启用控制权后，确认的失败才可进入融合器。

### 3. 恢复目标

可靠阶段元数据将持续无进展映射为：

- approach -> `reacquire_intended_target`；
- grasp -> `establish_stable_control`；
- place/manipulate/goal_relation -> `restore_goal_relation`。

进展停滞只产生 `HOLD_THEN_REPLAN`，不会凭弱视觉证据自动释放或后退。释放、后退仍要求独立且高置信的具体机制诊断。

### 4. 一次性恢复提示

补全 `SupervisorRuntime.build_policy_prompt`：

- 初始和正常推理不修改任务提示；
- 仅当恢复状态为 `REPLAN_PENDING` 且桥接动作已结束时追加恢复里程碑；
- `mark_replan_completed` 安装新动作块后，后续推理恢复普通任务提示；
- 禁用恢复提示时保持原提示逐字不变。

## 测试覆盖

- 信息防火墙移除实测状态与 reward；
- 默认 shadow 模式不能把视觉失败提升为控制事件；
- 未校准的高分错误证据不能清空或截短策略队列；
- 连续两次、校准且阶段可靠的 place 无进展会生成
  `restore_goal_relation + HOLD_THEN_REPLAN`；
- UTF-8 在线日志包含语义事件、恢复目标及干预指令；
- 恢复提示仅在待重规划查询期间有效。
- LIBERO inference 轨迹显式记录 `recovery_prompt_target`，避免只看到
  `unknown` 机制而无法还原实际发送给策略的恢复里程碑。

最终完整 CPU 回归：`316 passed`。在线 `monitoring_main.py` 通过其 Python 3.8
环境的 `py_compile`；覆盖前保留了
`monitoring_main.pre_task_consequence_runtime_20260925.py`。

## 诚实边界与下一步

现在完成的是监督—诊断—恢复接口闭环，不是视觉识别率突破。真实 GPU 闭环测试前仍缺少经过冻结验证的在线任务后果证据源。下一阶段应先做最小因果闭环验证：使用冻结证据源，在配对相同 seed 的 LIBERO 轨迹上比较监督器关闭、shadow、控制开启三组；重点报告原本成功被破坏数、原本失败被挽救数、误干预数和恢复后成功率，而不是只报异常识别率。

## 配对评估器修正

真实 LIBERO trace 的 `action_index` 从 1 开始，旧评估器却把首次干预的
动作编号直接作为 Python 切片长度，因此会把首个已经改变的动作错误纳入
“干预前一致性”前缀。现改为按 step 行的零基位置切片，动作编号只用于报告，
并新增 1 起始轨迹反例测试。

修正后重放现有完整配对集：

- `closed_loop_v1`：20 对，20 个 preserved success，因果前缀错误 0；
- `closed_loop_v1_libero90`：10 对，2 个 preserved success、8 个 unresolved failure，因果前缀错误 0；
- 旧 recovery-prompt 开发 manifest 引用了已移动的轨迹文件，不能由当前 manifest 直接重放，未伪造补齐。

这些结果说明旧执行层在已成功任务上尚未观察到破坏，但对 LIBERO-90 的 8 个
基线失败也没有救回；这正是下一阶段必须加入“恢复哪个任务里程碑”而不能只
重复停滞重询的原因。
