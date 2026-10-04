# 视觉响应与 task-agnostic 双专家阶段报告（2026-09-05）

## 阶段目标

在不启动 π0.5、不使用 GPU 的条件下，推进两条路线：动作条件视觉异常证据；跨 LIBERO suite 的任务无关动力学基线。所有文本和 JSON 均以 UTF-8 保存。

## 视觉实验

### 语义潜变量残差 v3

仅用 399 个正常 chunk 拟合 patch-preserving 岭回归，输入为上一视觉状态摘要与去 task-ID 的机器人条件，预测主/腕相机的视觉 latent delta；异常标签未参与拟合。

seed20 calibration 结果：全体 active-vs-normal AUC=0.433，双专家模糊区 AUC=0.488。异常残差没有升高，原因可能是执行衰减使视觉 latent 变化更小、更接近静止均值。结论：该版本无有效异常可观测性，不进入阈值冻结。

### 定向光流响应 v3

从已有第三人称视频中只使用 4×4 patch 的水平/垂直 Farneback 光流（32维），丢弃灰度、颜色、场景外观和 task-ID。正常模型学习：

\[
\hat f_t=h(c_t),
\]

其中 `c_t` 是 58 维动作/本体条件。seed20 calibration：

- 高残差 AUC=0.605；
- 低方向响应 AUC=0.734；
- 低余弦 AUC=0.632；
- 在旧双专家 `ambiguous_overlap` 内，低方向响应 AUC=0.601；
- 简单阈值 + 2-of-3 在不超过既有正常误报预算时救回 0 个 episode。

剩余动力学漏检均为 scale=0.75。全图粗光流对 25% 衰减的信噪比不足；下一版应局部跟踪夹爪/末端与被操作物体，并比较两者相对运动，而不是继续调全局光流阈值。

## task-agnostic 双专家

从条件向量删除最后 10 维 task one-hot，保留 58 维指令、关节状态、末端状态、历史响应和 chunk phase。正常专家使用 200 个正常 episode、21,570 step、5 折 episode 隔离训练；异常专家使用 60 个异常 episode、600 个 active step。

seed20 只用于全局阈值校准；seed21 只执行冻结评估。与旧 task-specific 基线（50/57、正常 FP 1/23、active recall 0.711）相比：

| calibration 正常 FP 预算 | seed21 检出 | seed21 正常 FP | active-step recall |
|---:|---:|---:|---:|
| 0 | 52/57 | 3/23 | 0.730 |
| 1 | 54/57 | 5/23 | 0.749 |
| 2 | 54/57 | 7/23 | 0.767 |

budget1 分 scale 检出为 0.25:19/20、0.50:19/20、0.75:16/20。任务无关化提高了召回和轻异常敏感性，但全局正常尾部分布从 seed20 到 seed21 漂移，误报明显增加；不能宣称整体胜过旧模型。下一步需引入不依赖任务身份的运动状态/可观测性条件标准化。

## 跨 suite 预注册

已生成 100 条命令、预计 120 episode 的 staged matrix，并通过静态审计。development/calibration 使用 `libero_object`、`libero_goal` 和分层抽样 `libero_90`；整个 `libero_10` 在冻结前封存。每个新 suite 先做策略成功率能力闸门，再做扰动检出；策略失败与执行故障检出分别报告。

## 本阶段发现并修复的问题

1. 原命令生成器假设不存在的 `--args.task-indices`，已改为每个 `--args.task-id` 一条命令。
2. 初版矩阵让 `libero_10` 同时出现在 development 与 held-out，审计发现后已移除泄漏。
3. PyTorch 2.6 默认 `weights_only=True` 无法加载自建含 NumPy 数组的可信 checkpoint，已显式使用 `weights_only=False`。
4. 一次将 refit 分数与非 refit fusion config 错配，导致误报预算异常；对齐诊断后改回配套构件。后续需把 checkpoint/score/config 哈希写入统一 manifest。
5. PowerShell 提前展开远端 `$b`，导致 shell 循环参数为空；改为完全显式命令。

## 下一阶段

无卡可继续：实现夹爪/机械臂局部 ROI 与物体相对运动特征；增加构件哈希配对审计；为 task-agnostic 专家设计运动状态条件标准化。需要 GPU 时：先执行各 suite 的正常能力闸门，再采集 development/calibration；封存 `libero_10` 直到模型和阈值冻结。

## 2026-09-05 时间对齐更正

后续代码审计确认图像在执行当前动作之前保存，因此动作 `a_t` 应对应 `frame[t] -> frame[t+1]`。本报告早先的全图低方向响应AUC=0.734使用了错位的`frame[t-1] -> frame[t]`，现予以撤回，不应在论文、简历或项目介绍中引用。

保持所有非视觉字段不变并正确重算后，seed20 calibration的全图低定向响应AUC为 **0.785**，模糊区AUC为 **0.670**；但在既定正常误报预算及2-of-3持续规则下仍未新增episode检出。详细审计见 `outputs/CROSS_SUITE_GPU_PILOT_AND_ALIGNMENT_FIX_20260905.md`。
