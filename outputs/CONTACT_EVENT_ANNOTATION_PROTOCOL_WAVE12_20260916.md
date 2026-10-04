# Wave1/2 接触与受控对象开发标注协议

## 目的与数据角色

该标注只用于拟合/校准接触、独立运动、附着、受阻和语义接口。Wave1和Wave2均已参与算法开发，因此不能产生新的测试成绩；Wave3也已打开，只能诊断。最终结论必须来自规则冻结后的Wave4。

## 两阶段隔离

### Pass A：物理事件盲标

标注者只看匿名双视角视频、帧号和夹爪开合提示，不看：

- 任务语言；
- episode成功/失败、reward；
- 动作专家分数；
- 算法选择的候选编号；
- 语义模型结果。

每次闭爪事件记录：

1. `contact_onset_frame`：首次可见接触；不可判断填`null`并标`uncertain`。
2. `contact_entity_type`：`movable_object`、`fixed_scene`、`robot_self`或`unknown`。
3. `independent_motion_interval`：物体首次明显脱离世界静止背景的区间。
4. `control_established_interval`：固定视角出现独立运动且腕部视角与夹爪稳定附着的共同区间。
5. `physical_mechanism`：`successful_pickup`、`empty_grasp`、`push_without_grasp`、`fixed_obstacle_or_jam`、`object_slip_or_loss`、`planned_release`或`unknown`。
6. 遮挡、分割不可见和证据冲突必须单列，不得强行补标签。

### Pass B：语义核验

只有Pass A的受控对象和时间区间冻结后，第二位标注者才查看任务语言与冻结的受控对象裁剪，回答：

- 受控对象是否符合任务名词；
- 空间关系是否符合限定语，例如“抽屉中”“炉灶上”“盘子与ramekin之间”；
- 置信度和歧义原因。

Pass B不得修改Pass A的物理受控对象选择。这样可以测量“抓到了什么”和“抓到的是不是目标”两个不同误差源。

## 标注一致性

- 至少20%样本由两位标注者独立复标；
- 离散机制报告Cohen's kappa和混淆矩阵；
- 起止帧报告绝对误差中位数、p90以及容差±3帧的一致率；
- 冲突样本由第三人仲裁，但保留原始两份标签；
- 不得因算法预测错误而回看并修改真值定义。

## 概率模型的开发原则

- 几何残差必须在Wave1/2标注上校准，不能手工缩放成概率；
- `blocked_motion_probability`可以联合末端低响应与“没有独立运动物体”用于故障类型诊断，但不能循环地充当接触的唯一证据；
- 语义探针训练只能使用Wave1/2或额外独立数据；
- 所有阈值、窗口、模型哈希和缺失值策略在Wave4生成前冻结。
