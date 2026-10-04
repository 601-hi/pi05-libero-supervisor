# 双专家可信度与序列监督器 v1

## 阶段状态

本阶段完成了无卡可实现的代码、测试和预注册，但尚未在真实 DROID 校准数据上生成五成员集成产物。原因是无卡重启后服务器清空了 SSH 公钥，本地副本不包含 DROID 训练/校准 NPZ。禁止使用已打开的 LIBERO-10 拟合可信度或序列参数。

## 可信度不是异常概率

双专家原始证据为

\[
\Lambda_t=\log p_A(\Delta x_t\mid c_t)-\log p_N(\Delta x_t\mid c_t).
\]

可信度衡量“当前证据是否处在专家有数据支持、多个模型意见稳定、传感器有效的区域”，而不是再次预测异常类别。实现采用

\[
r_t=\min(r_{support,t},r_{stability,t},r_{sensor,t}).
\]

采用最小值而非乘积，是为了让任一关键失效模式明确成为瓶颈，并避免三个中等分数相乘后无意义地趋近零。

### 数据支持度

令

\[
B_t=\max(\log p_N,\log p_A).
\]

用 DROID 可靠性拟合子集上正常与合成异常样本的 \(B\) 经验分布计算当前百分位。若两个专家都给出极低绝对似然，支持度趋近零。这样可避免 \(\log p_N=-100,\log p_A=-95\) 虽似然比为5却被误称为“已知异常”。

### 集成稳定性

训练五个不同随机初始化的双专家，计算

\[
\sigma_{ens,t}=\operatorname{Std}_m(\Lambda_t^{(m)}).
\]

其在源域校准分布中的百分位越高，稳定性越低。当前版本是深度集成；未来若实现 episode bootstrap，应仍保持 episode 整体抽样，不能逐帧打散。

### 传感器质量

当前确定性检查包括：有限数值、时间间隔偏差、异常末端位置跳变和异常关节速度。真机阶段再加入时间戳、丢包、电流/力矩传感器饱和与视觉质量。

## 四状态决策

\[
\begin{cases}
known\ normal,&r_t\ge r_{min},\ \Lambda_t\le\tau_N\\
known\ abnormal,&r_t\ge r_{min},\ \Lambda_t\ge\tau_A\\
ambiguous\ overlap,&r_t\ge r_{min},\ \tau_N<\Lambda_t<\tau_A\\
unknown,&r_t<r_{min}
\end{cases}
\]

正常门来自正常样本似然比的高分位，异常门来自异常样本似然比的低分位。两者不能简单都叫“q90”；如果 \(\tau_N\ge\tau_A\)，代码把校准标为重叠且拒绝加载。

## 序列风险

\[
R_t=\max\{0,\gamma R_{t-1}+r_t\operatorname{clip}(\Lambda_t,-a,a)-\kappa\}.
\]

风险超过 \(h_A\) 报警，降到更低的 \(h_R\) 才复位，构成迟滞。`unknown`不会作为负证据清除风险；连续未知达到三步时显式请求保护性观测，供上层减速、补充观测或重规划。

## 防止校准泄漏

DROID 校准 episode 按完整 episode 确定性划分：偶数编号仅拟合支持度、集成分歧和双门；奇数编号仅在80个预定候选中选择序列参数。LIBERO-10和未来目标留出集禁止参与。

主指标必须同时报告：正常 episode 误报、异常 episode 检出、延迟、未知率和非未知覆盖率。不能通过大量拒判使表面准确率变好。

## 已完成文件

- `reliability_sequence_supervisor.py`：可信度、传感器检查、四状态和序列累积器。
- `fit_reliability_calibration.py`：源域集成校准和严格 episode 分割。
- `fit_source_sequence_parameters.py`：源域序列参数选择。
- `score_relational_ensemble.py`：冻结集成的目标轨迹评分器。
- `evaluate_reliability_sequence.py`：冻结目标评估器。
- `run_source_reliability_stage_v1.sh`：服务器无卡运行入口。
- `RELIABILITY_SEQUENCE_PREREGISTRATION_V1.json`：冻结前预注册。
- `test_reliability_sequence_supervisor.py`：9项行为测试，全部通过。

## 服务器恢复后的唯一下一步

先同步上述文件，再执行 `bash cross_suite_generalization/run_source_reliability_stage_v1.sh`。只有源域门不重叠、正常 episode 约束满足且全部测试通过，才允许冻结哈希并选择新的未查看目标场景。LIBERO-10最多做诊断性回放，不再产生主结果。

