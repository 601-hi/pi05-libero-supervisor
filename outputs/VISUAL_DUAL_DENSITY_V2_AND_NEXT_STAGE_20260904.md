# 视觉双密度专家 v2 与下一阶段预注册

日期：2026-09-04

## 本阶段结论

视觉双密度专家 v2 被否决，不并入冻结动力学监督器。该模型分别拟合正常和受扰视觉潜变量的条件密度，并使用

\[
S_V=\log p_A(z\mid task)-\log p_N(z\mid task)
\]

作为相对证据。它解决了“单一分类概率无法解释”的形式问题，但没有解决跨环境 seed 的表征漂移和类别重叠。

## 实现

- 输入只用因果对齐的 π0.5/PaliGemma 视觉潜变量：agent/wrist × current/delta 的固定投影，不使用实际扰动比例或未来信息。
- 在训练集上标准化后做 PCA，候选维数 8/16/32。
- 每个 task 分别拟合正常/异常对角高斯；样本不足时向全局类别统计收缩，候选强度 5/10/20。
- 原始对数似然比使用各 task 的训练正常轨迹做稳健中心/尺度标准化。
- 只允许视觉在冻结双动力学专家的 `ambiguous_overlap` 区域触发。
- 模型与阈值只由 seed18/19 训练及 seed20 校准决定；seed21 已经在 v1 中打开，所以只作诊断，不作为新的独立测试证据。

## 结果

seed20 的基础监督器已经占用 1 个正常 episode 误报预算。九组候选在该预算内均无法新增异常检出，最优合法阈值为正无穷，即关闭视觉门。

标准化视觉似然比的条件 AUC：

| split | AUC |
|---|---:|
| train seed18/19 | 0.8795 |
| calibration seed20 | 0.5726 |
| opened diagnostic seed21 | 0.5407 |

seed20 每个有异常样本的 task 中，异常视觉分数超过同 task 正常最大值的比例均为 0。故失败不是单一全局阈值导致，而是当前密度假设和视觉摘要在新 seed 上没有稳定的异常方向。

## 与显式间隔实验的共同解释

此前异常动力学专家加入显式 margin 后，训练集正常/异常似然比确实被拉开，但 seed20 检出由原始 25/30 降至 21–24/30。视觉双密度也在训练集取得高 AUC，却在校准集接近随机。

因此不能把“训练分数间隔”本身当成研究目标。安全监督器需要的是在未见环境条件下仍稳定的物理或语义不变量。

## 现有信息边界

当前 68 维动力学条件包含目标平移、旋转/夹爪动作、关节位置和速度、末端位姿、夹爪位置和速度、上一动作/实际平移、动作转向、速度对齐、上一末端速度、task one-hot。它不含物体位姿、目标位姿、接触力、关节力矩或是否抓持成功。

`chunk_phase_onehot` 只表示 5 步 action chunk 内的位置，不表示接近、抓取、搬运、放置等操作阶段。因此不能再把它解释成语义阶段。

## 下一阶段预注册

1. seed21 仅作为 hard-negative 开发集，用来设计特征，不再报告为独立泛化结果。
2. 新增结构化操作状态：夹爪闭合趋势、末端速度/转向、视觉物体—夹爪相对关系、物体是否随夹爪运动，以及视觉不可观测标记。
3. 动力学双专家继续输出物理响应似然；视觉分支输出任务进展/物体关系证据，不再直接重复拟合末端运动异常。
4. 融合器保留 `normal / known abnormal / unknown abnormal / ambiguous`，视觉不可见时不得否决强动力学异常。
5. seed22 仅用于校准阈值与每 task 正常误报预算；冻结全部参数和清单后，seed23 才作为新的最终测试。
6. 接受标准同时报告 episode 检出、正常 episode 误报、active-step recall、inactive-step alarm rate、按 scale/task 分层结果；不能只用 AUC。

## 无卡阶段的操作状态代理诊断

使用现有68维条件比较seed21中的291个正常模糊块、13个视觉正常误触发块、22个动力学漏检active块和62个已检出active块：

- 目标平移均值中位数：正常39.9 mm、视觉误触发17.9 mm、漏检19.9 mm、已检出40.2 mm；
- 目标平移最后一步中位数：视觉误触发仅7.9 mm；
- 上一末端速度中位数：漏检0.1068 m/s、已检出0.1047 m/s，速度本身无法分开；
- 漏检active块主要集中task1（12/22），视觉误触发主要集中task0/task4（各5/13）。

解释：大动作自由空间衰减更容易形成明显动力学证据；小动作、转向、抓取/放置附近，正常控制误差与弱扰动更重叠。下一版应将机器人状态用于条件化视觉解释，而不是把小动作直接判为正常或异常。关键新增量仍是“物体是否随夹爪运动”和“视觉是否可观测”。

## 可复现文件

- `visual_latent_v2/visual_dual_density.py`
- `visual_latent_v2/fit_visual_dual_density.py`
- `visual_latent_v2/evaluate_visual_dual_density.py`
- `visual_latent_v2/diagnose_visual_dual_density.py`
- `visual_latent_v2/analyze_operation_state_proxies.py`
- `outputs/visual_dual_density_v2_exploratory.json`
- `outputs/visual_dual_density_v2_seed21_diagnostic.json`
- `outputs/visual_dual_density_v2_overlap.json`
- `outputs/operation_state_proxy_diagnostic.json`
