# DROID-100 外部动力学预训练准备完成（2026-09-08）

## 阶段目标

把外部数据工作推进到“训练命令已经可运行，但尚未更新任何模型权重”的停止点。该阶段不使用 `libero_10`，不启动 π0.5，不使用 GPU，也不使用 DROID 的图像、任务语言、奖励或成功标签作为动力学特征。

## 完整数据与可追溯性

下载官方 DROID-100 `r2d2_faceblur/1.0.0` 的全部31个TFRecord分片，对应100个episode。每个分片记录官方URL、字节数与SHA-256；完整清单为 `outputs/remote_results/droid100_training_ready/droid100_complete_manifest.json`。

下载期间最后一个分片出现工程故障：中断本地SSH未终止远端串行下载器，随后并行下载器与其同时写入同一`.part`，导致长度虽正确但官方区间抽样字节不一致。系统没有按长度误判成功；损坏文件被改名保留，随后从官方对象以四个互不重叠区间完整重下，并在文件开头、拼接边界两侧、中部和尾部做逐字节在线比对，五处全部一致，才进入最终清单。

## 100条真实轨迹的时序结论

关闭TFDS自动缓存与并行交错，并对三路图像使用`SkipDecoding`后，100个episode、32,212步全部流式审计成功。

| 模态 | 通过单模态可辨识门 | 主导最佳lag |
|---|---:|---|
| EEF平移 | 98/100 | 89条为+2帧 |
| EEF旋转 | 97/100 | 76条为+1帧、21条为+2帧 |
| 关节 | 98/100 | 55条为+1帧、43条为+2帧 |

所有三种模态同时通过的episode为97条。该结果把小样本中的1--2帧时延提升为DROID-100层面的稳定证据；仍不能把它解释为跨机器人固定常数。

## 低维因果训练包

只有三模态质量门全部通过的97条episode进入导出。按`SHA256(file_path) mod 5`在episode层固定80/20近似划分：78条train、19条calibration，3条拒识轨迹仍完整记录在manifest中。

- train：25,509个转移；SHA-256 `414192f48c0d9c16fb371bedc5f29e58b817550f0db5dc43f875d5c36d053182`。
- calibration：5,547个转移；SHA-256 `4c5fd8afc7586b4028658bf4836b82f4f420348bb27681d2028d68edc2301650`。
- 输入：lag 0--5的6维Cartesian command历史（6×6），当前EEF位姿6维、关节位置7维、夹爪位置1维，共50维。
- 目标：下一步EEF平移/包角旋转增量6维、关节增量7维、夹爪增量1维，共14维。
- episode路径只用于稳定划分并以哈希记录，不作为模型特征。

训练前授权器只在train split拟合median/IQR；检查完整100条、通过率不低于预注册75%、数组形状/有限性、episode身份唯一、禁用字段和文件哈希。结果为97%通过、零违规、`authorized: true`。授权只适用于“使用DROID域内归一化的条件响应表征预训练”，不允许直接设LIBERO阈值或混合两域原始动作单位。

## 已冻结训练入口，但尚未训练

配置：`cross_suite_generalization/DROID_RESPONSE_PRETRAIN_V1.json`。模型为两层128宽SiLU MLP，输出14维响应均值和14维对数方差，以对角异方差Gaussian NLL训练。训练入口默认只验证；必须显式提供`--train`才执行优化。

真实数据`validate_only`结果：train 25,509、calibration 5,547、输入50、目标14、探针输出8×28且全部有限。没有创建模型checkpoint，没有反向传播。

下一次真正训练命令为：

```bash
cd /root/gpufree-data
CUDA_VISIBLE_DEVICES='' /root/gpufree-data/vla-workspace/openpi/.venv/bin/python \
  train_external_response_encoder.py \
  --config /root/gpufree-data/DROID_RESPONSE_PRETRAIN_V1.json \
  --readiness /root/gpufree-data/droid-schema-sample/training_bundle_v1/TRAINING_READINESS.json \
  --data-directory /root/gpufree-data/droid-schema-sample/training_bundle_v1 \
  --output /root/gpufree-data/droid-schema-sample/training_bundle_v1/droid_response_encoder_v1.pt \
  --train
```

当前按用户要求停在命令执行之前。

## 2026-09-09训练与独立校准补记

在无GPU、0.5核CPU实例上，batch 256首轮超过160秒且未生成checkpoint，安全停止后仅将batch改为2048并固定Torch单线程；数据、模型宽度、学习率及episode级划分不变。完整40轮约11秒。

首版把14维响应全部作为连续高斯，产生第40轮校准NLL 75139。逐维审计确认不是训练成功：夹爪增量82.2%为零，IQR=0被替为1e-6，活动样本被放大至约7万至12.8万标准化单位，单维均方5.89e8并支配总损失。该checkpoint已改名保留为`invalid_gripper_gaussian`，禁止使用。

V1随即改为结构化响应模型：13维EEF/关节连续高斯头、夹爪活动事件头，以及仅对活动步计算的夹爪幅值高斯头。新模型在第19轮达到最低校准总损失0.973，第27轮早停并恢复第19轮。

封存校准split的独立对照结果：

- 连续响应NLL 0.291，对常数基线1.190，改善75.5%。
- 连续响应归一化RMSE 2.035，对常数基线2.681，改善24.1%。
- 连续响应绝对标准化残差覆盖：1σ 79.4%、2σ 95.9%、3σ 98.8%，略保守但无方差塌缩。
- 夹爪活动事件ROC-AUC 0.818，0.5门平衡准确率0.756。
- 夹爪事件Brier 0.179，差于常数先验0.151；因此输出目前是排序分数，不是校准概率。
- 活动夹爪幅值归一化RMSE 0.442，对常数基线0.490，仅改善约9.8%。

可复核产物位于`outputs/remote_results/droid100_response_encoder_v1/`，包括checkpoint、逐轮训练史和完整evaluation JSON。该结果只证明DROID域内未见episode上的条件响应学习有效，尚未授权直接设置LIBERO阈值，也尚未证明跨机器人零样本泛化。
