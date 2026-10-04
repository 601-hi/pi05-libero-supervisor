# 语义视觉潜变量级联 v1：冻结测试报告

日期：2026-09-04

## 结论

v1 不通过接受门，不能并入当前冻结监督器。它在双专家模糊区具有一定相对排序信息，但跨seed绝对分数失准：冻结seed21上没有增加异常episode检出，却把正常误报从1/23提高到10/23。

当前生产候选仍保持动力学双专家，不启用视觉v1。

## 不干预验证

固定task0、环境seed7、diffusion noise seed 20260904，视觉旁路关闭/开启各运行1条：

- 17次推理、82步；
- action chunk最大绝对差0；
- intended action最大绝对差0；
- EEF position最大绝对差0米；
- 两相机潜变量均为`[17,4,4,2048]`、float16、全部有限；
- JSONL/NPZ的17个chunk shape、索引和SHA256全部通过。

因此视觉旁路本身没有改变策略轨迹。

## 数据与因果对齐

- train：seed18/19，80 episode，8495 step，1649 chunk transition；
- calibration：seed20，40 episode，4316 step，839 chunk transition；
- frozen test：seed21，80 episode，8828 step，1715 chunk transition；
- 第k次重规划视觉状态用于复核已执行的第k−1动作块；首块排除；视觉证据不复制到5个step伪造时间多数票。

跨日期旧trace与新trace虽然固定噪声哈希、初始EEF相同，但首个action chunk已有约0.001–0.002的差，闭环后终止步数不同。推断原因是不同GPU/驱动/JAX bfloat16内核的数值差异。故没有把新视觉强行拼接旧score，而是使用原封不动的动力学checkpoint对新trace重评分。

## 模型冻结过程

输入394维：四组64维固定随机投影（agent/wrist × current/delta）、两个68维条件摘要、两个专家chunk似然均值。训练仅选双专家上一chunk含`ambiguous_overlap`的clean正常或active异常样本。

候选：logistic margin 0/0.5/1；两层MLP margin 0.5/1。seed20选择`mlp_m1`：

- 模糊区条件AUC 0.7031；
- 双专家基线26/30、正常FP 0/10；
- 视觉级联27/30、正常FP 1/10；
- 冻结阈值0.998891055583954；
- checkpoint SHA256 `795cedfd6237afe1898d1ab9532d1b1e381c1dc5e0db9b5adef6beb37fb8d9b3`。

## seed21冻结结果

今天同轨迹双专家基线：

- 23条物理正常、57条实际受扰；
- 异常检出49/57；
- 正常误报1/23；
- active step recall 69.30%；
- scale 0.25/0.50/0.75分别18/19、18/19、13/19。

加入冻结视觉v1：

- 异常检出仍为49/57，visual rescue 0；
- 正常误报升为10/23；
- 模糊区条件AUC 0.7190；
- active视觉触发6个chunk，但均来自基线已检出的episode；
- inactive视觉触发47个chunk。

9个视觉新增正常误报中task0占5个、task3占2个、task1和task4各1个。多数episode只有一个极高分chunk，说明排序信息尚可，但全局绝对阈值对任务与seed的条件偏移非常敏感。

## 研究解释

视觉语义token不是无效：AUC从低级视觉的0.451提高到约0.719，表明它确实包含自由空间、任务阶段或物体关系信息。但当前小头把“训练seed上的异常相似度”变成了过度自信概率，任务背景与阶段偏移会制造极端误报。高AUC并不保证在低误报安全系统中存在可用阈值。

下一版应优先研究：

1. task/阶段条件化或分层校准，而非全局阈值；
2. 正常视觉专家与异常视觉专家的相对证据及拒识区，而非单一过度自信分类头；
3. episode/group-aware训练与hard-negative正常chunk；
4. conformal或分位数门控制每task正常误报；
5. 显式物体跟随/接触区域表示，而非仅扁平随机投影。

这些方向必须使用新的校准seed，不能在已打开的seed21上重新调参后声称独立提升。
