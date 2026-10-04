# 语义视觉潜变量监督器 v2：预注册设计

日期：2026-09-04

## 1. 研究问题

动力学双专家在 seed21 上优于旧 v2，但仍有 6 个受扰 episode 未检出，并产生大量
`ambiguous_overlap` 步。低层光流/灰度/颜色特征的条件 AUC 只有 0.451，不能解释“机械臂是否仍在
朝正确物体、正确接触阶段运动”。本阶段只在双专家判为模糊时，引入 π0.5 自身的冻结视觉表征。

视觉模块不替代动力学模块。它回答的是：在动力学证据无法分开时，当前视觉状态及其变化是否更像
正常演化还是受扰后的偏离。

## 2. 提取层与机器人学含义

LIBERO 的第三人称和腕部图像均为 224×224。π0.5 的 SigLIP So400m/14 将每路图像切为
16×16 个 patch，经 27 层视觉 Transformer 后，再投影到 PaliGemma 的 2048 维前缀空间。

首版提取 `PaliGemma.img` 的输出，而不是动作专家的隐藏状态：

- 它已包含高层视觉语义，优于原始光流；
- 它仍保持 16×16 空间对应，可区分末端、目标物、容器和障碍物所在区域；
- 它位于语言融合之前，避免修改或复制 KV-cache/action denoising 主路径；
- 两路相机分开保留，以区分全局任务阶段与近距离接触状态。

每 4×4 patch 做一次均值池化，得到每相机 `4×4×2048` 的 float16 数组。右腕图像是 padding，
依据 image mask 丢弃。每次重规划（默认每 5 个动作）记录一次。

## 3. 不干预动作的工程约束

`Pi0.sample_actions` 保持逐行不变。新增 `encode_visual_latents(observation)` 旁路，并在动作采样完成后
独立调用。它不读取或推进策略 RNG，也不写 KV cache，输出不反馈给策略。

正式采集前必须做 A/B 配对：相同仿真 seed、相同固定 diffusion noise、无扰动；A 关闭潜变量，B 开启。
以下量必须逐元素严格等于 0：

1. action chunk 最大绝对差；
2. 每步 intended action 最大绝对差；
3. EEF position 最大绝对差。

任一失败即停止，不得采集训练数据。这是为了排除监视器本身改变策略时延/RNG/轨迹的混杂。

## 4. 对齐与存储

JSONL 的 inference 行仅保存：`chunk_id`、潜变量形状、两相机拼接字节的 SHA256。每个 episode 单独保存
压缩 NPZ：

- `task_id`, `episode_idx`；
- `chunk_ids`；
- `action_start_indices`（该视觉观测将生成的第一个动作索引）；
- `base_0_rgb`: `[num_chunks,4,4,2048]`, float16；
- `left_wrist_0_rgb`: 同上。

校验器要求 episode、chunk、shape、hash 全部吻合。按约 22 次推理/episode 估算，原始未压缩潜变量约
5.5 MiB/episode；80 episode 约 440 MiB，规模可控。

## 5. 数据切分（结果揭晓前固定）

- 训练：seed18、seed19；
- 标定：seed20，只用于 PCA/标准化/阈值；
- 冻结测试：seed21，不用于模型选择；
- 所有划分按 episode，而不是按 step，防止相邻帧泄漏；
- 类别定义沿用受扰 active interval，但训练视觉模型只取“双专家模糊区”中的样本；
- 正常类来自 clean normal 的模糊区；异常类来自 exposed rollout 的 active 模糊区；
- 同一 `(task, episode)` 的所有 chunk 必须在同一 split。

## 6. 视觉验证器候选（最小复杂度优先）

不直接训练大视觉网络。冻结 π0.5 后只训练小头：

1. 用固定种子的高斯随机投影将每个32768维相机张量压到64维；它不看标签、无需在高维上求协方差，且可由seed精确重建。PCA仅作为训练split内可选消融，不作为默认路径；
2. 构造当前潜变量、与上一 chunk 差分、第三人称/腕部差异，共 256 维左右；
3. 拼接双专家的chunk似然摘要、动作阶段特征；
4. 比较 logistic regression、带 class weight 的 2 层 MLP；
5. 只允许在 `ambiguous_overlap` 状态触发，连续规则仍使用因果 2-of-3。

因果对齐按chunk转移完成：第k次重规划的视觉状态用于复核刚执行完的第k-1块。首块没有历史，必须丢弃；检测延迟上界为一个replan chunk。不得把同一chunk视觉分数复制成5个“独立”step再套2-of-3，视觉时间滤波只按独立重规划观测计数。

显式间隔作为小头的监督目标，而不是再次修改已失败的异常 GMM：正常样本令 logit ≤ -m，异常样本令
logit ≥ +m，损失为交叉熵加 hinge margin。m 只在 seed20 上选。

## 7. 主要评价指标与接受条件

首先报告 episode-level detection、正常 episode false alarm、active-step recall、inactive-step alarm rate；
再报告双专家模糊子集的 AUROC/AUPRC 与按 scale 分层结果。

视觉级联只有同时满足以下条件才接受：

- seed21 正常 FP 不高于双专家基线的 1/23；
- seed21 检出数严格高于 50/57；
- scale=0.75 的 14/19 有提升；
- 改善不能只来自一个 task；
- 固定噪声动作一致性测试严格通过。

若不满足，则将“π0.5 冻结视觉 token 在现有数据量下无增益”记录为负结果，不追逐测试集阈值。
