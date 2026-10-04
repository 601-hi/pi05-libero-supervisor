# 显式间隔与视觉模糊区复核先导实验

日期：2026-09-04

## 1. 旧v2独占检出样本

样本：`source3:task1:episode1`，scale0.75，active actions 29–38。

- 旧v2从action30起持续报警；双专家10步全部为`ambiguous_overlap`。
- 每步目标平移约41.8–50.2 mm；异常实际位移约8.5–10.0 mm，配对正常约11.1–12.0 mm。
- 异常/正常实际位移比在稳定段约0.74，与75%执行缩放一致；action38后EEF相对配对正常累计偏离27.9 mm。
- 双专家标准化似然比约0.81–1.51，低于冻结known-abnormal门2.953；旧v2的actual/residual/progress/response与历史窗口组合则给出高风险。
- 第三人称画面显示该阶段夹爪仍在桌面上方自由空间接近目标，没有明显合法接触约束。因此该样本首先暴露训练分离不足，而不是必须依赖视觉才能判断。

配对帧条带保存在`outputs/visual_pilot_task1_ep1/`，每条依次为actions 25、29、32、35、38、42。

## 2. 显式间隔异常专家

正常专家冻结。异常专家保留条件混合密度NLL，并加入：

\[
\mathcal L=-\log p_A(\Delta x_A|c_A)
+\lambda_A\operatorname{softplus}(m_A-z_\Lambda^A)
+\lambda_N\operatorname{softplus}(m_N+z_\Lambda^N).
\]

`zΛ`使用train正常轨迹在原异常专家下的按task稳健中心/尺度固定标准化。seed21不参与训练或选择。

训练层面，强候选使active ratio-z中位数约5.63→6.72，clean normal约0→−1.93，异常标准化NLL约−0.87→−4.48，说明间隔确实被拉开且密度拟合未崩坏。

但在seed20、正常episode误报预算0下：

| 候选 | 检出/30 | 0.25 | 0.50 | 0.75 | 正常误报 |
|---|---:|---:|---:|---:|---:|
| 原双专家 | 25 | 10 | 10 | 5 | 0 |
| 强margin m2, λA=.25, λN=.10 | 23 | 10 | 8 | 5 | 0 |
| 温和margin m1, λA=.10, λN=.025 | 24 | 10 | 8 | 6 | 0 |
| 只推动异常 m2, λA=.10, λN=0 | 21 | 9 | 8 | 4 | 0 |

结论：训练分数分离不等于校准后检测更好。融合门会吸收整体分数平移，且margin会改变局部密度形状。三个候选均被否决，不替换冻结v1，也不在已打开的seed21上继续调参。

## 3. 第三人称视觉模糊区先导

现有视频资产：seed18/19各40条、seed20 40条、seed21 80条第三人称MP4；视频帧数与step数逐一一致。此前未设置`visual_out_path`，因此没有腕部相机数组。

视觉特征严格因果：当前/前一帧的4×4分块Farneback光流x/y/幅值与帧差、8×8灰度上下文、4×4颜色上下文，再加入意图动作字段与task one-hot。不输入EEF实际位移、执行scale、起点、reward、success或未来帧。

训练只选择冻结双专家的`ambiguous_overlap`：97个active异常重叠step、1172个clean正常重叠step。训练集分离很强，但seed20上：

- active异常重叠66 steps，clean正常重叠594 steps；
- 条件AUC仅0.451；
- active分数中位数2.38e−11，clean中位数7.01e−10；
- 正常0误报预算下最优视觉门为∞，即不调用视觉；级联保持原25/30；
- 即使允许1条正常误报，也没有新增episode检出。

结论：低级第三人称运动特征跨seed过拟合，而且与本体运动信息冗余。它不能承担语义上的“自由空间/正常接触/物体跟随”判断，视觉先导被否决。

## 4. 下一版视觉设计

1. 继续使用`ambiguous_overlap`作为按需调用门，不全程运行视觉模型。
2. 使用预训练语义视觉表征，优先评估π0.5/PaliGemma视觉潜变量或冻结的通用视觉编码器，而不是从少量episode学习低级光流。
3. 新GPU采集必须设置`visual_out_path`，同步保存agent view与wrist view原始数组，并保留action索引。
4. 数据必须专门覆盖困难正常接触、自由空间弱衰减、物体跟随/未跟随、遮挡和视觉不可判定窗口。
5. 输出至少为`normal-contact / abnormal / visually-unobservable`三态；视觉不可见不能否决强动力学异常。
6. 训练/校准使用新split；冻结后使用全新seed测试。seed21只保留为诊断集。

## 5. 可复现文件

- `finetune_abnormal_expert_margin.py`
- `run_margin_candidates_cpu.sh`
- `export_visual_motion_features.py`
- `train_visual_ambiguity_verifier.py`
- `evaluate_visual_ambiguity_cascade.py`
- `extract_task1_visual_pilot.sh`

