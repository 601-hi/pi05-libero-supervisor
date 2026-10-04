# 腕部与物体相对运动视觉实验预注册（2026-09-05）

## 研究问题

在双动力学专家输出 `ambiguous_overlap` 时，腕部视觉的动作条件局部运动能否识别全图光流漏掉的25%轻执行衰减（scale=0.75），且不提高正常episode误报？

## 数据边界

- development：`libero_spatial` seed18/19及已采跨suite pilot；
- calibration：全新seed22；
- frozen test：全新seed23；
- `libero_10`全suite继续封存；
- seed20仅用于复核旧阶段，seed21已降级为hard-negative development，不得再充当最终测试。

## 输入与禁止信息

允许输入：当前执行前腕部图像与下一执行前腕部图像、intended action、夹爪开合指令、EEF线速度/角速度、chunk phase、多尺度历史。

禁止输入：task-ID one-hot、任务成功标签、扰动scale、disturbance active标志、episode未来帧、仿真内部物体真值位姿、碰撞真值。检测器必须可以迁移到真实机器人传感器。

## 候选证据（冻结顺序）

1. 腕部全图4×4定向光流响应，作为最小基线；
2. 腕部中心/周边分区的径向与切向光流，用于区分相机自身运动和近场物体相对运动；
3. 1/3/5步累计光流与动作积分的方向一致性，避免单帧噪声；
4. 夹爪开合条件分层：approach/open、closing/contact-transition、transport/closed、release；该阶段由可观测夹爪指令和状态生成，不用任务文本；
5. 若有可靠目标追踪，再加入目标相对夹爪的位移；不得用光流强度本身选择ROI后又用同一强度判异常。

## 模型与融合

- 正常视觉专家只用正常轨迹拟合动作条件响应；异常标签不参与其参数学习。
- 先报告视觉单独的step AUC、分scale AUC、按夹爪阶段AUC和跨suite分层结果。
- 视觉只复核动力学 `ambiguous_overlap`，不能否决 `known_abnormal`。
- 点证据经预先固定的2-of-3或累计序贯门转为报警；两者只能在seed22选择其一。
- 阈值按正常episode误报预算0/1/2分别冻结，不只汇报最优预算。

## 采集矩阵

每个开发suite先运行正常能力闸门，只在π0.5具备基本成功能力的task上做配对扰动。每个入选task至少采正常、scale0.75和scale0.5；正常/扰动使用相同环境seed与固定扩散噪声。保存逐控制步agentview与wrist RGB，且日志记录图像是在动作前还是动作后采集。

## 必须通过的审计门

1. 无扰动重复运行的noise hash、action chunk、逐步动作和EEF轨迹完全一致；
2. 配对轨迹在扰动开始前完全一致，开始步intended action相同、executed action首次分离；
3. 动作`t`只使用`frame[t] -> frame[t+1]`，每个episode最后动作标无效；
4. 数据、模型、阈值、融合config和代码SHA-256写入同一manifest；
5. seed23打开前冻结特征、模型、阈值和评价脚本。

## 成功标准

主要标准：在相同正常episode误报数下，融合后异常episode检出严格高于冻结双动力学专家，并至少救回一条原先漏检的scale0.75轨迹。次要标准：active-step recall提高、inactive alarm不显著恶化、跨suite方向一致。若只提高AUC但不增加低误报episode检出，则判为“可观测但尚不可部署”，不宣称监督器改进。

## 无卡实现与单对pilot sanity check

已实现 `visual_latent_v2/export_wrist_multiscale_motion.py`。每个1/3/5步窗口输出4×4水平/垂直/幅值光流，以及腕部中心与环带的幅值、径向和切向统计，共72维；动作`t`严格对应执行前腕图`t`到`t+1`。合成水平/垂直平移测试在远端OpenCV环境通过，实际轨迹的有效行数也精确符合因果边界。

固定噪声 `libero_object task0` 正常/scale0.5配对pilot中，扰动前定向光流比值为1.000。扰动同一chunk与重规划后的disturbed/normal中位数：1步约0.600/0.580，3步约0.657/0.604，5步约0.972/0.866。5步窗口在扰动起始处混入正常历史，造成明显稀释和延迟。因此正式候选以1步和3步为主，5步仅作为预注册消融。
