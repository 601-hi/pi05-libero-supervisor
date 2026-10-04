# 已知平移衰减异常动力学专家：实现与验证协议

## 研究边界

第一版异常专家只学习当前有可控数据支持的故障族：执行平移相对于策略意图发生持续衰减。它不能被称为通用碰撞、滑移或触壁专家；这些故障需要单独的数据和视觉/力觉证据。

异常专家建模：

\[
p_A(\Delta x_t\mid s_t,u_t,h_t)
\]

正常专家建模：

\[
p_N(\Delta x_t\mid s_t,u_t,h_t)
\]

融合的基础证据为似然比：

\[
\Lambda_t=\log p_A(\Delta x_t\mid s_t,u_t,h_t)-\log p_N(\Delta x_t\mid s_t,u_t,h_t)
\]

若两个专家似然都低，输出应为 `unknown_abnormal`，而不是强行归入已知衰减。

## 为什么异常专家采用混合密度

scale 0.25、0.50、0.75对应不同响应增益；接触和控制阶段还会进一步分裂分布。单高斯会把多个异常模式平均成一个并不存在的响应。第一版使用3分量对角高斯混合网络，scale只作为训练标签分析分层性能，绝不作为部署输入。

## 禁止进入模型的字段

- `executed_action`
- `disturbance_active`、`translation_action_scale`
- `disturbance_start_mode`、`scheduled_disturbance_start`
- 绝对 `action_index`（只允许chunk phase和因果历史）
- reward、done、success
- 仿真接触真值、物体真值位姿
- 未来观测或完整episode统计

允许输入仅为现实机器人部署时可获得的执行前信号：策略意图动作、关节位置/速度、EEF姿态、夹爪状态、上一步动作/响应和chunk phase。当前步实际位移是专家要解释的观测，不是执行前条件。

## 数据协议

- 起点必须随机或由可部署动作事件触发，禁止只使用固定action 40。
- split单位为完整episode，且env seed不得跨train/calibration/test重复。
- 训练建议：至少3个新env seeds，scale 0.25/0.50/0.75，每task每scale每seed至少2 episodes。
- calibration与test各保留至少1个完全未见env seed。
- 正常配对轨迹必须使用相同初态和固定π0.5采样噪声；扰动前动作与EEF最大差异应为0。
- 报告完整10步暴露与提前成功导致的短暴露；mean target norm低于0.015m标为低可观测。

## 评价顺序

1. 单独评价正常专家 \(p_N\)。
2. 单独评价异常专家 \(p_A\) 对不同scale的负对数似然和模式覆盖。
3. 冻结两专家后，只在calibration集拟合似然温度和先验。
4. 在未见seed测试正常、三种scale和未知扰动。
5. 同时报告正常episode误报、active recall、episode检测率、延迟、低可观测率和unknown率。

## 无卡阶段产物

- 正常专家CPU运行时与状态机。
- 异常数据审计器和泄漏黑名单。
- 异常混合密度专家训练骨架。
- GPU阶段轨迹采集矩阵与验收规则。
