# 同候选语义对齐与任务进程恢复桥（2026-09-25）

## 冻结语义复核

GPU只运行了预注册的task58 step195和task85 step326，没有增加样本或打开密封测试集。首次运行发现通用语言适配器不支持“pick up X and put it in Y”和方向关系句式，导致整句被错误地用作物体查询。该批结果被判无效，没有进入融合。修复并新增回归测试后：

- `pick up the ketchup and put it in the tray`被解析为`place_in(ketchup, tray)`；
- `pick up the white mug and place it to the right of the caddy`被解析为`place_right_of(white mug, caddy)`。

## 为什么两个possible事件最终都弃权

task58 step195中，固定视角候选3与DINO的ketchup框IoU约0.915，但它的闭合变点为-0.615；候选8的变点为+0.467，却与DINO和Florence的ketchup框均不重合。运动与语义证据落在不同候选上。

task85 step326中，唯一持续运动候选6的变点仅约+0.023，与DINO和Florence的white mug框均无重合；图像复核也显示它是机械臂结构。

因此最终18事件状态为：1个`preexisting_or_robot_motion`、10个`fixed_motion_only`、4个`ambiguous`、2个`evidence_split_across_candidates`、1个`insufficient_window`，0个`control_established`。该结果禁止把候选A的运动、候选B的语义和候选C的腕部稳定拼成一个抓取结论。

## 接通任务进程到恢复里程碑

代码审计发现语义后果模块虽能输出`no_progress`、`wrong_object`和`object_lost`，却没有写恢复路由器要求的统一`diagnostic_state`，因此检测与恢复此前并未真正连通。

现在只有在语义后果已经校准、连续确认且任务阶段可靠时，持续无进展才映射为：

- approach → `approach_progress_stalled` → 重新获取目标；
- grasp → `grasp_progress_stalled` → 重新建立稳定控制；
- place/manipulate/goal_relation → `goal_relation_progress_stalled` → 恢复目标关系。

阶段不可靠时不生成里程碑诊断。上述进度诊断只改变下一次π0.5重规划的局部目标，执行器仍采用保持动作；只有独立的高置信度抓空、抓错、丢失或卡住证据才可能解锁释放或退避原语。

## 工程状态

- 修复两类通用英文任务关系解析，不使用LIBERO task ID。
- 新增同候选语义—物理对齐审计。
- 正式状态机新增`evidence_split_across_candidates`。
- 语义后果与恢复里程碑路由完成数据契约连接。
- 完整服务器回归310/310通过。

下一步是审计生产采集器是否逐步写入语义状态、可靠阶段、诊断状态、恢复目标和实际使用的恢复提示；随后做最小闭环烟测，先验证因果分叉和无危险桥接动作，再运行成对任务效果测试。
