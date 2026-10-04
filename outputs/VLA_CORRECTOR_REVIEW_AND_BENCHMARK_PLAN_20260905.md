# VLA-Corrector方法审查与本项目统一比较计划

日期：2026-09-05

## 1. 它如何降低视觉背景捷径

VLA-Corrector没有直接训练`图像潜变量 -> 正常/异常`分类器。冻结VLA视觉编码器后，它在成功示范转移上学习

\[
\Delta\hat Z_{t+k}=M_\phi(Z_t,a_t),\qquad
\Delta Z^*_{t+k}=Z_{t+k}-Z_t.
\]

在线分数为

\[
E_t=1-\cos(\Delta\hat Z_{t+k},\Delta Z^{real}_{t+k}).
\]

其抗捷径机制包括：

1. **预测残差而非绝对潜变量**：时间不变的桌面、柜子和纹理在`Z_{t+k}-Z_t`中被一阶抵消。
2. **动作条件化**：只有与当前动作一致的视觉变化才算正常，不能仅靠任务背景给出标签。
3. **冻结VLA表征、外置corrector**：避免异常辅助目标改写策略内部表征；论文的耦合检测头明显弱于外置LVM。
4. **方向误差而非原始概率**：使用预测/真实潜变量变化的余弦不一致度，降低潜变量整体尺度与相机亮度变化的影响。
5. **在线稳健门**：15步窗口的median/MAD、on/off滞回、连续5步确认和10步冷却，抑制单帧尖峰与局部尺度漂移。
6. **episode级划分及跨域试验**：论文报告了固定episode split与LIBERO→MetaWorld迁移，用于检查一定程度的记忆化。

但它**没有彻底证明不存在背景捷径**。当前状态`Z_t`仍作为corrector输入；若动作阶段与背景/姿态高度相关，模型仍可能利用相关性。论文和公开README未报告背景替换、相机扰动、目标纹理交换、因果遮挡或shortcut probe，也未公开训练数据与corrector checkpoint。因此更准确的说法是“结构上显著抑制静态背景”，不是“已经证明消除背景作弊”。

## 2. 为什么我们此前视觉v1/v2失败

我们的v1直接把current/delta潜变量及动力学摘要送入异常分类头；v2则在固定随机投影后拟合正常/异常视觉密度。虽然也含delta，但没有学习`当前视觉+动作 -> 应有视觉变化`这一条件动力学关系，绝对current特征和task相关外观仍可成为捷径。结果是train AUC 0.8795，而seed20/seed21降至0.5726/0.5407。

正确修正不是继续调分类阈值，而是实现条件视觉残差预测，并把检测与干预分离。

## 3. 与本项目及其他路线的可比定位

| 方法 | 核心监督信号 | 在线输出/作用 | 公开证据 | 与本项目关系 |
|---|---|---|---|---|
| 本项目双动力学专家 | 指令、本体状态、实际末端响应；正常/异常条件密度 | known/unknown/ambiguous状态，尚未闭环纠正 | scale扰动57条检出50，正常23条误报1，active-step recall约71% | 故障检测统计细、传感器现实可得；异常种类、任务范围和样本量有限 |
| VLA-Corrector | 成功示范上的动作条件视觉残差预测 | 动态MAD检测、截断、OGG重规划 | MetaWorld、LIBERO、三种VLA、真实PiPER；端到端成功率提升 | 系统更完整；论文未给与我们同定义的异常precision/recall，不能直接数字排名 |
| VLA-in-the-Loop | 成功+高价值失败数据，抓取关键帧判别与生成成功视频 | 只在闭爪关键点介入，逆动力学解码纠正动作 | 仿真与真实抓取；ICLR 2026在审版本 | 语义纠错强但限定抓取关键点、计算重，非连续通用安全监测 |
| PhysReflect-VLA | 物理可行性、转移一致性、LLM反思 | 检查候选动作并生成纠正指导 | 接触丰富真实任务平均增益5.4% | 与我们的物理一致性思想接近，但增加语言反思和端到端纠正 |
| SafeContract/黑盒动作监测 | 动作反转、jerk、速度等，conformal校准 | 训练自由报警 | 450 episodes；不同策略架构的AUROC差异 | 低成本强基线；不观察实际物理响应，也不执行恢复 |
| PiL-World | 成功与失败轨迹训练的多视角chunk世界模型 | 离线policy-in-the-loop评估 | 真实VLA成功率估计误差63.2%降至12.0% | 主要是评估模拟器，不是轻量在线安全门 |

## 4. 当前能否宣称领先

不能。原因：

- 我们报告的是注入执行衰减的检测率/误报率，VLA-Corrector等主要报告干预后的任务成功率；指标不同。
- 我们尚未测试截断能否提高成功率、策略调用开销、恢复率和检测延迟。
- 当前主要覆盖LIBERO-spatial、π0.5和一种translation scale异常；缺少碰撞、卡滞、滑移、目标移动、传感异常及真实机器人。
- 57条受扰、23条物理正常的冻结测试规模远小于论文的多任务/多骨干评估。

可以合理声称的阶段优势是：配对固定噪声因果实验、明确区分物理正常与真正受扰episode、报告step/episode级误报和检出、保留unknown/ambiguous拒识状态；这些比只给任务成功率更有诊断价值。但这属于**评估严谨性和可解释性优势**，不是总体性能领先。

## 5. 统一基准的预注册方案

### Baseline A：当前冻结双动力学专家

保持模型、阈值和状态机不变。

### Baseline B：VLA-Corrector风格LVM

第一阶段使用现有chunk级数据：

```text
for each clean transition (Z_k, action_chunk_k, Z_{k+1}):
    target = Z_{k+1} - Z_k
    predicted = residual_predictor(Z_k, action_chunk_k)
    loss = cosine_loss(predicted, target) + optional small L2 loss

online:
    expected = residual_predictor(Z_k, executed_action_chunk)
    actual = Z_{k+1} - Z_k
    score = 1 - cosine(expected, actual)
    alarm = robust_state_machine(score)
```

不得输入扰动scale、reward、success、未来标签或仿真物体真值。训练只用seed18/19 clean转移；seed21可用于hard-negative结构开发但不选最终阈值；seed22校准；冻结后seed23测试。

第二阶段采集每控制步视觉latent，复现论文的step级window=15、patience=5、on/off MAD门，并与chunk级延迟比较。

### Baseline C：融合

仅在动力学`ambiguous_overlap`时调用LVM，同时测试OR、AND和拒识融合；全部使用同一正常误报预算。

### 闭环干预

先只测试“报警即丢弃剩余动作并普通重规划”，把检测收益与OGG收益分开。若截断有效，再实现OGG或其他纠正策略。

### 统一指标

- 正常episode误报率；
- 受扰episode检出率；
- active-step recall、inactive-step alarm rate；
- 检测延迟与扰动结束前检出率；
- 任务成功率及相对无监测基线的提升；
- policy calls/episode、success-per-call与墙钟开销；
- 按task、scale、异常类型分层和95%置信区间。

只有在同一轨迹、同一随机种子、同一扰动和同一误报预算下比较，才能判断双专家、LVM或融合方案谁更强。

## 6. 直接影响下一步工程的结论

下一视觉模块优先复现VLA-Corrector的**动作条件残差预测器**，而不是立即做物体检测大模型。结构化物体跟随仍作为后续互补模态，因为LVM残差本身未显式保证物体语义正确。当前无卡阶段可以完成数据接口、训练脚本和评估状态机；训练残差网络和逐步latent采集需要GPU。

