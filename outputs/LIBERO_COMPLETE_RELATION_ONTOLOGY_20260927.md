# LIBERO 全关系视觉监督本体与覆盖矩阵（2026-09-27）

## 1. 审计范围与边界

本次直接解析 LIBERO 安装目录中的全部 130 个 BDDL 文件，而不是按照少数测试视频人工枚举规则：

- `libero_90`：90 个任务；
- `libero_10`、`libero_goal`、`libero_object`、`libero_spatial`：各 10 个任务；
- 共 151 个最终目标原子；
- 在线监督器不会读取 task ID、仿真对象位姿或环境成功判据；这些信息只能用于离线真值和评估。

审计发现最终目标只使用六种原始谓词：`In` 63 次、`On` 61 次、`Close` 11 次、`Turnon` 8 次、`Open` 7 次、`Turnoff` 1 次。所有谓词可归并为四个可复用关系族，机器可读映射见 `libero_visual_relation_coverage_matrix_20260927.json`。

这里的“全覆盖”严格表示：每个 BDDL 目标都能被解析，并有明确的监督接口、状态空间和弃权语义；它不表示视觉估计已经对每个目标达到可靠精度。

## 2. 两级问题分解

不能把任务语言直接交给一个“成功/失败图像分类器”。监督应拆成两个独立问题：

1. **实体落地（grounding）**：语言里的主体和目标到底对应画面中的哪个实体。例如“左边的碗”决定应跟踪哪个碗。
2. **关系验证（verification）**：已确认的主体和目标是否满足 `In/On/Open/...`。例如已锁定碗和盘子后，再判断碗是否稳定在盘子上。

形式化地，任务目标是谓词图：

\[
G=\bigwedge_k R_k(e_{k,1},e_{k,2}),
\]

视觉侧首先维护实体信念 \(q(e\mid I_{0:t},L)\)，再由关系适配器输出：

\[
(\hat s_{k,t},u_{k,t})=h_{R_k}(I_{0:t},\hat e_{k,1},\hat e_{k,2}),
\]

其中 \(\hat s\) 是关系状态，\(u\) 是不确定度。证据不足时必须输出 `uncertain/unobservable`，不能把猜测当作成功或异常。

## 3. 四类关系适配器

### 3.1 包含关系 `In`

覆盖抽屉内部、微波炉腔体、篮筐/托盘/收纳盒等容器区域。状态为：

`outside → crossing_boundary → inside → uncertain`

主要证据是主体掩膜相对容器开口/内部区域的边界穿越、遮挡顺序、释放后持续性，以及固定相机和腕部相机的一致性。单帧二维包围框重叠不能独立证明三维“在里面”。

### 3.2 支撑关系 `On`

覆盖桌面区域、炉灶表面、物体顶部以及物体堆叠。状态为：

`not_supported → approaching → supported → released_and_stable → uncertain`

主要证据是主体—支撑体相对几何、接触高度一致性、相对运动锁定、夹爪释放后的持续稳定。对“左/右/前/后区域”，空间词首先用于实体/区域落地，随后仍由同一个支撑适配器验证。

### 3.3 机构状态 `Open/Close`

覆盖柜子抽屉和微波炉门。状态为：

`open / closed / intermediate / moving / uncertain`

测量对象不是“机械臂有没有动”，而是把手或面板相对固定机构的位移/角度，并用该机构的开、闭端点做归一化。端点标定属于少量机器人/场景适配参数，不应写成 task ID 特例。

### 3.4 设备状态 `Turnon/Turnoff`

当前 BDDL 中对应炉灶开关。状态为：

`on / off / uncertain / unobservable`

若图像中有可靠指示灯、旋钮状态或其他可观测通道，使用时序确认；若当前相机根本不可观测，就必须弃权，并由低频语义复核或其他传感器补充，不能由末端运动间接猜测。

## 4. 目标区域与实体选择

151 个目标进一步映射为 10 个适配器子类型：

| 子类型 | 数量 |
|---|---:|
| containment/container_interior | 46 |
| containment/drawer_interior | 16 |
| containment/microwave_cavity | 1 |
| support/object_or_stack | 31 |
| support/workspace_region | 11 |
| support/object_top_surface | 10 |
| support/appliance_surface | 9 |
| articulation/drawer | 15 |
| articulation/microwave_door | 3 |
| device/binary_state | 9 |

语言还包含左、右、前、后、中间、两者之间、顶部、底部等选择词。它们不一定是新的最终谓词：很多 `libero_spatial` 任务最终仍只是 `On(subject, region)`，空间词负责选择正确主体或目标区域。因此第一版必须把选择词解析为**实体约束**，而不是误当成独立成功条件。

## 5. 组合目标与进展状态机

一个任务可包含多个目标原子：110 个任务有 1 个，19 个有 2 个，1 个有 3 个。每个原子独立保留四值状态：

\[
v_k\in\{\text{satisfied},\text{violated},\text{uncertain},\text{unobservable}\}.
\]

整体完成只有在所有原子均为 `satisfied` 且满足时间迟滞时成立；任何原子未知时都不能宣称完成。状态机还保存“曾满足后失效”，从而覆盖物体滑落、抽屉重新弹开等后果异常。

这也解释了 task 0 的失败：真实目标只有 `Close(top_drawer)`，黑碗只是干扰物。机械臂移动黑碗即使动作—执行一致，也没有使唯一目标原子改善。任务进展层应在若干动作块后输出 `goal_no_progress`，而不是等待 400 步超时。

## 6. 第一版在线接口

每个关系适配器使用统一输出契约：

```text
RelationEvidence:
  predicate             # In / On / Open / Close / Turnon / Turnoff
  subject_track_id
  target_track_id
  state                 # adapter-specific state
  progress              # [0,1] if measurable, otherwise null
  confidence
  uncertainty_reason
  observation_timestamp
  evidence_provenance   # fixed camera / wrist / tracker / low-frequency review
```

在线路由遵循：

1. 动作—执行一致性可靠且目标关系改善：继续执行；
2. 一致性异常且物理安全风险高：执行层立即停止/回退；
3. 一致性正常但目标长期无改善：判为规划/语义进展问题，低频复核后重规划；
4. 视觉不确定：不授权危险动作，也不单凭视觉触发大幅回退；请求重新观察或使用已有安全恢复链。

## 7. 验证计划

第一阶段先验证结构而非追求完美识别：

- 单元级：130 个 BDDL 全部解析，151 个目标全部有适配器，当前 unsupported=0；
- 关系级：每个关系族至少包含成功、未完成、完成后失效和不可观测样本；
- 任务级：优先验证已知自然失败，特别是 task 0 的 `Close` 无进展；
- 系统级：严格同服务、同 seed、同采样噪声比较无监督器与有监督器，报告任务成功率、误介入率、发现延迟和恢复成本；
- 泛化级：按未见任务/未见对象拆分，不用同类测试任务的数据反向训练阈值。

## 8. 当前结论

LIBERO 第一版视觉监督不需要为 130 个任务写 130 套规则，而需要：一个通用谓词解析器、一个带集合/空间约束的实体落地器、四类关系适配器、一个支持未知状态的组合目标状态机。该结构覆盖了基准的全部目标语义，同时保留迁移到其他机器人时只重做相机标定、实体检测接口和少量机构端点标定的可能性。

对应的通用时序实现已加入 `vla_supervisor/goal_relation_monitor.py`。它已经用不依赖视觉模型的单元场景验证：多帧完成确认、组合目标必须全部满足、持续无进展、完成后退化，以及不可观测时请求重新观察而不是误报。当前仍缺的是四类关系适配器产生真实 `RelationEvidence` 的感知实现及在线接线。
