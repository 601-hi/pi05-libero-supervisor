# 双动力学专家VLA监督器：相关工作、相似度与创新边界

日期：2026-09-05

## 结论先行

截至本次检索，没有发现与本项目**完整结构完全相同**的公开工作：在action-chunked VLA执行中，同时训练正常与故障条件响应密度模型，以相对似然和两个绝对似然把在线状态显式分为`normal / known abnormal / unknown abnormal / ambiguous overlap`，并只在证据重叠区调用视觉动力学复核。

但各个组成思想大多有明确先例：

- 正常/故障多个模型竞争与似然比：传统model-based FDI已有长期历史；
- 只学习正常动作条件动力学：VLA-Corrector、PATCH、FoMo-FD等已有；
- 使用失败数据训练故障识别器：监督式failure detection和VLA-in-the-Loop已有；
- 拒识、conformal阈值与时间窗口：FAIL-Detect、FIPER等已有；
- 动作、本体响应、视觉多模态监测：已有多种机器人故障检测工作。

所以可维护的创新主张应是**面向VLA执行层的开放集双动力学证据分解与选择性视觉复核**，而不是“两专家”“似然比”“动力学异常检测”本身。

## 1. 本项目方法的可检索定义

条件为可部署的当前信息

\[
c_t=(u_t,q_t,\dot q_t,x_t,\text{gripper},\text{history},\text{task}),
\]

响应为执行后的末端变化

\[
y_t=\Delta x_t.
\]

正常专家与故障专家分别估计

\[
p_N(y_t\mid c_t),\qquad p_A(y_t\mid c_t).
\]

相对证据为

\[
\Lambda_t=\log p_A(y_t\mid c_t)-\log p_N(y_t\mid c_t),
\]

但决策不只使用一个二分类边界，还保留两个绝对支持度：

| 正常支持 | 异常支持 | 状态 |
|---|---|---|
| 高 | 低 | normal |
| 低 | 高 | known abnormal |
| 低 | 低 | unknown abnormal |
| 高 | 高 | ambiguous overlap |

视觉模块只在`ambiguous overlap`中检查动作条件视觉变化，而不能否决强本体异常。

## 2. 相似度分级方法

以下百分比是研究设计相似度审计，不是论文给出的指标。评分维度：

- 监测目标与action-chunk/VLA背景：15%；
- 动作条件动力学建模：20%；
- 正常/异常双模型竞争：20%；
- 似然或概率证据：15%；
- known/unknown/overlap拒识：15%；
- 多模态选择性融合：10%；
- 在线事件与恢复：5%。

分数只能帮助定位近邻，不能作为法律意义上的新颖性检索结论。

## 3. 最近邻矩阵

| 工作 | 估计相似度 | 与我们重合 | 关键差异 |
|---|---:|---|---|
| 传统hierarchical/interacting multiple-model FDI | 55–65% | 正常/故障模型、模型似然、后验模式概率、故障隔离 | 使用解析运动学/动力学和预定义执行器故障；不是VLA、学习策略或接触任务开放集监测 |
| Mendoza等context-dependent model inaccuracy | 45–55% | 上下文条件残差、名义/替代模型似然比、置信检测 | 目标是机器人模型不准确区域与模型修正；没有VLA双学习专家和四态拒识 |
| VLA-Corrector | 当前35–45%；计划融合后约55% | action-chunk、冻结VLA特征、动作条件latent动力学、在线持续检测 | 只有成功示范的正常corrector；无显式故障专家、unknown/overlap；有截断和OGG |
| PATCH | 当前30–40%；计划视觉融合后约55–65% | action-chunk条件视觉动力学、概率patch创新、持续证据、在线路由 | 强调空间执行走廊、self-motion过滤和外部场景扰动；无本体正常/故障双专家四态 |
| FoMo-FD | 当前35–45%；计划融合后约50–60% | 动作条件窗口动力学、概率非一致度、任务阈值、正常误报控制 | 只学正常视觉世界模型；无故障专家和证据重叠；面向手术机器人 |
| FAIL-Detect | 30–40% | 运行时失败检测、成功数据建模、flow密度、conformal时间阈值 | 主要是正常/OOD标量信号；没有物理响应双专家和已知/未知故障分解 |
| FIPER | 20–30% | 观察OOD+动作chunk熵双信号、conformal、窗口聚合 | 监测策略不确定性而非执行后的命令—物理响应 |
| Model-Based Runtime Monitoring/Sirius | 35–45% | 动作条件latent dynamics、失败分类、在线交互改进 | dynamics rollout再接failure classifier；没有双条件响应密度与四态证据 |
| VLA-in-the-Loop | 25–35% | 高价值失败数据、关键时刻风险判别、在线纠正 | 专注闭爪关键帧，VQA/视频生成/逆动力学；不是连续物理响应监测 |
| Runtime surgical errors with dual Siamese networks | 30–40% | 正常/错误轨迹对比、上下文、显式错误模式 | Siamese判别而非两个生成式响应密度；特定手术动作，不是VLA开放集 |
| Distance-aware temporal failure detection | 35–45% | 机器人状态/物体距离、时序残差、rollout校准带 | 主要为正常时序模型；使用物体相对距离，不含异常专家竞争 |
| RoboFailRing/AHA | 20–30% | 已知故障库、未知/OOD失败评估、语义解释 | 视觉语言检索/推理，不建模命令—物理响应 |

## 4. 特别需要吸收的相关工作

### PATCH

PATCH是对“视觉背景会误导监测器”处理得最直接的近邻：

1. 动作块投影为未来执行走廊，只关注可能受机器人影响的patch；
2. 预测每个latent patch的条件高斯均值和方差；
3. 以patch负对数似然作为innovation；
4. 过滤可由机器人self-motion解释的变化；
5. 累积持续、局部、动作相关的外部创新；
6. 报告未见背景和障碍实验。

值得复现：空间局部化、self-motion过滤、证据记忆。其固定相机/标定执行走廊也是限制；本项目可以研究不依赖精确图像空间标定的本体—视觉证据融合。

### FoMo-FD

FoMo-FD表明直接预测误差不一定是最好的似然分数。它用动作条件flow-matching世界模型对观察到的endpoint latent做inverse-transport nonconformity，并用成功rollout conformal校准。在四类手术任务和20种失败中，腕部视角达到96.6% FDR、1.3% FAR。其消融还显示动作条件将腕部FDR从87.2%提高到96.6%，窗口K从1增至4将FDR从42.2%提高到96.6%。

值得复现：窗口endpoint、腕部优先、非一致度而非单点预测误差、成功rollout校准。

### FAIL-Detect/FIPER

它们提示正常阈值不能只取一个训练分位数，应使用部署成功rollout的conformal校准，并针对任务阶段或时间构造变化阈值。FIPER还说明把“观察OOD”和“策略动作不确定性”作AND组合可以排除无害OOD。

值得复现：有限样本误报控制、独立信号组合和检测时间评价。

### 传统multiple-model FDI

传统方法已经明确使用名义/故障模型的似然比和后验模式概率。因此我们的论文必须引用这条历史，并说明区别不是数学公式，而是：模型从VLA闭环轨迹学习、条件包含策略指令与接触相关本体状态、异常专家来自可控执行扰动、并保留未被任一模型解释的unknown状态。

## 5. 哪些创新表述不能使用

不能写：

- “首次使用双模型检测机器人异常”；
- “首次使用正常/异常似然比”；
- “首次利用动力学残差监测VLA”；
- “首次在VLA中融合视觉和本体状态”；
- “首次区分已知和未知异常”（开放集识别本身已有大量先例）。

这些表述容易被传统FDI、开放集识别和近期VLA监测工作直接推翻。

## 6. 目前较可辩护的创新主张

在完成统一基准和新异常实验后，可以争取如下表述：

> We introduce an open-set dual-dynamics execution monitor for action-chunked VLA policies. Unlike nominal-only world-model monitors or closed-set failure classifiers, it contrasts learned nominal and fault-conditioned physical response likelihoods while retaining absolute support from both experts. This decomposes execution into nominal, known-fault, unknown-fault, and overlap states, and invokes action-conditioned visual monitoring only for unresolved overlap.

中文：

> 提出一种面向动作分块VLA的开放集双动力学执行监督器。区别于仅建模正常执行的世界模型监测器或闭集故障分类器，该方法比较学习得到的正常与故障条件物理响应似然，同时保留两个专家的绝对支持度，将执行证据分解为正常、已知故障、未知故障和证据重叠，并仅对未决重叠状态调用动作条件视觉复核。

这项主张的潜在新意由四个环节共同构成：

1. **故障生成模型而非故障判别器**：异常专家建模`故障条件下会产生什么物理响应`；
2. **相对证据与绝对支持并存**：避免似然比在两个模型都很差时仍被迫二选一；
3. **known/unknown/overlap的操作性分解**：直接决定停止、继续、复核或采集新故障数据；
4. **选择性跨模态复核**：视觉不是全程OR门，而只解决本体专家重叠，且不能推翻强物理异常。

## 7. 要让创新主张站稳，还缺哪些实验

1. 与nominal-only正常专家、单一二分类MLP、传统似然比、VLA-Corrector式LVM、PATCH式局部创新和conformal基线统一比较。
2. 未知异常必须真正未进入异常专家训练：如完全卡滞、反向偏置、单轴故障、延迟、动作丢包、滑移/目标移动。
3. 做四态消融：只用似然比、加入绝对门、加入unknown、加入overlap视觉复核，逐项证明价值。
4. 报告unknown异常召回、known异常识别、overlap覆盖率、选择性风险—覆盖率曲线，而不只报总体检测率。
5. 对不同task、seed、异常严重度、VLA骨干和至少一种真实机器人验证。
6. 视觉shortcut测试：背景替换、无关运动、相机轻扰、遮挡、物体纹理替换；同时测PATCH式action corridor能否降低误报。
7. 报警后的截断/普通重规划/引导重规划分别测试，证明监督信号能转化为任务收益。

## 8. 当前科研定位

目前最合理的定位是：**具有潜在结构创新、已有有希望先导结果，但尚未完成创新性验证的研究原型**。不能宣称SOTA或“主流从未用过”，但可以明确指出：近期主流集中于nominal-only视觉世界模型、策略不确定性或闭集失败分类；本项目探索的是故障条件物理生成模型与正常模型竞争后的开放集证据结构。

最终论文价值不应依赖一个漂亮名称，而应由统一基准证明：在相同误报预算下，双专家四态结构是否比nominal-only和binary classifier更好地同时识别已知与未知执行故障。

## 主要检索来源

- VLA-Corrector, arXiv:2607.01804
- PATCH, arXiv:2606.16690
- FoMo-FD, arXiv:2607.27511
- FAIL-Detect, arXiv:2503.08558
- FIPER, arXiv:2510.09459
- Model-Based Runtime Monitoring with Interactive Imitation Learning, arXiv:2310.17552
- Runtime Detection of Executional Errors in Robot-Assisted Surgery, arXiv:2203.00737
- Hsiao & Weng, A hierarchical multiple-model approach for detection and isolation of robotic actuator faults, Robotics and Autonomous Systems, 2012
- Mendoza et al., Detection and correction of subtle context-dependent robot model inaccuracies, IJRR, 2019
- RoboFailRing, ACL 2026

