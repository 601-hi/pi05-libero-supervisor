# 通用情景视觉分析：实现方案选型

日期：2026-09-27  
性质：文献调研 + 方案决策输入。**明确不做对象/场景/任务特调**（无抽屉专用规则、无 LIBERO task ID、无固定像素区、无对象名硬编码）。  
前一版 `VISUAL_RELATION_EVIDENCE_SCHEME_DECISION_20260927.md` 把 P0 写成“抽屉开合比例”过拟合到单条失败案例，**以本文件为准**。

---

## 1. 通用问题定义

监督器需要的不是“某个场景的检测器”，而是**语言条件下的通用视觉任务状态分析**：

\[
g_t = P\big(\Phi(I_{0:t}, L)=\text{true}\big),\qquad \Delta g_t = g_t-g_{t-w}
\]

其中 \(\Phi\) 是语言解析出的**可组合谓词**，不是某个物体名。输出必须带可观测性与弃权；仿真特权信息不得进入在线输入。

### 1.1 为什么必须通用

- 项目硬约束：状态机不得含 task ID、suite 名、对象名、固定像素区、场景手写步骤；任务差异只能通过声明式关系表达。
- 目标是冻结 π0.5 的通用监督与纠偏，不是 LIBERO 专用插件；跨 suite、跨对象、跨机器人是验收条件。
- 文献一致结论（Opening Articulated Structures, RSS 2025）：针对单一机构/场景孤立优化的感知模块换视角就崩；**模块化通用原语 + 系统级组合**优于场景特化端到端。

### 1.2 谓词层（通用、可组合）

解析器输出的是**关系型**，实现必须按**关系类型**写测量原语，而不是按对象写管线：

| 谓词族（通用） | 语义 | 适用任意… |
|---|---|---|
| `in(o, r)` | 物体进入区域/容器 | 碗→柜、杯→抽屉、书→盒 |
| `on(o, s)` | 物体置于支撑面 | 盘上、桌上、架上 |
| `config(p, c)` | 可动部件处于目标构型 | 抽屉/门/盖/旋钮的开合或角度 |
| `controlled(o)` | 目标被夹爪稳定控制 | 任意可抓物 |
| `stable(pred)` | 谓词持续成立 | 任意组合目标 |

`close the top drawer` = `config(drawer, closed) ∧ stable`  
`put the bowl in the drawer` = `in(bowl, drawer)`  
`put the bowl in the drawer and close it` = `in ∧ config(closed) ∧ stable`  
`turn on the stove` = `config(burner_knob, on)`

同一套原语服务全部句式；**没有任何“if drawer”**。

---

## 2. 通用视觉分析架构（推荐主线）

```text
                    任务语言 L
                        │
            ┌───────────▼───────────┐
            │  谓词解析（已有规则层） │  GoalRelation / 声明式子目标
            └───────────┬───────────┘
                        │ 实体指称 + 关系假设
                        ▼
        ┌──────────────────────────────────────┐
        │ A. 低频语义假设（通用 VLM，开放词汇）  │
        │    实体候选 / 关系假设 / 阶段 / 目标态 │
        │    结构化输出 + 弃权，不直接控机器人   │
        └──────────────────┬───────────────────┘
                           │ 共识初始化（操作前）
                           ▼
        ┌──────────────────────────────────────┐
        │ B. 身份生命周期（已有，通用状态机）    │
        │    锁定 → 跟踪 → 遮挡 / 丢失 / 冲突   │
        └──────────────────┬───────────────────┘
                           │ 已确认实体掩膜/轨迹
                           ▼
        ┌──────────────────────────────────────┐
        │ C. 通用测量原语（主证据，连续、可校准）│
        │    C1 空间关系（in / on / 相对位姿）   │
        │    C2 构型状态（config：DOF 参数化）   │
        │    C3 控制/接触（controlled）          │
        │    输出：状态量 + 误差界 + 可观测性    │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │ D. 谓词求值 + 时序进展（已有监视器）   │
        │    g_t / Δg / goal_stable / no_progress│
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │ E. 低频 VLM 复核（冲突/阶段切换时）    │
        │    ROI + 前后帧，只消歧不授权          │
        └──────────────────────────────────────┘
```

**分工原则**

- **A/E（VLM）**：开放语义覆盖、假设与复核；FailBench 证明其自报置信不可靠，故永不单独形成控制证据。
- **B（生命周期）**：与机器人/任务名无关，继续复用。
- **C（测量原语）**：只写**关系类型的几何/运动学定义**，不写场景对象分支。
- **D（进展）**：已有 `goal_progress` / `semantic_consequence` 契约原样接入。

这对应项目文档中的路线 C，但测量内核是**关系类型级通用原语**，不是场景适配器表。

---

## 3. 通用测量原语（按关系类型，不按对象）

### 3.1 C1 空间关系原语 — `in` / `on` / 相对位姿

**定义（场景无关）**

对已锁定实体 \(o\) 与参照实体/区域 \(r\)：

- **包含 `in(o,r)`**：\(o\) 掩膜落入 \(r\) 腐蚀掩膜（或由深度/支撑平面定义的 3D 区域）的比例 \( \mathrm{IoA} \)；有深度时加高度序/法向一致性。
- **支撑 `on(o,s)`**：\(o\) 底部与 \(s\) / 支撑平面对齐 + 接触带稳定；平面可由场景内静态面 RANSAC 得到（不绑定桌子）。
- **相对位姿**：归一化相对位移/朝向（图像对角线或物体尺度归一），供 `left_of/behind/...` 扩展。

**误差**：跟踪质量门（面积、质心跳变、可见性）→ 误差界；遮挡或身份不可靠 → `unknown`。

**文献**：RelatiViT（ICLR 2024，通用空间关系）；Jund et al. ICRA 2018（关系度量学习、未知物体泛化）；Robo2VLM（NeurIPS 2025，空间/目标关系 VQA 自动构造）；Scalise IROS 2019（in/on 成功检测）。

**通用性**：同一函数服务“碗在抽屉”“杯在盘”“书在架”；输入只是两个身份 + 一个关系类型。

### 3.2 C2 构型状态原语 — `config(p, c)`（对应 `mechanism_state`，但是通用的）

**定义**：对任意“可动相对固定”的部件对 \((p, b)\)，估计**低维相对构型** \(q_t\)，而不是某个物体的开合比例。

1. 跟踪 \(p\) 与 \(b\) 的掩膜/稀疏点轨迹（已有 SAM2 / 可加点跟踪）。
2. 估计相对运动子空间维数与类型：1D 棱柱 / 1D 旋转 / 刚性（无相对运动）/ 未知。
3. 将相对位移投影到估计的自由度方向，得 \(q_t\)；用序列内极值或语言指定的语义极值（closed/open, off/on）归一化到构型语义标签。
4. 输出：`config_label`（离散语义）+ `q_t`（连续）+ 误差界 + `unknown`（DOF 不可辨识、遮挡、双解）。

**关键：DOF 由运动/几何估计，不靠对象类别表。**  
棱柱抽屉、推拉门、旋钮、上掀盖落在同一原语；语义标签（closed/open/on/off）由语言目标与极值约定，不写死部件名。

**文献**：Category-Independent Articulated Tracking（IROS 2022，因子图+位姿跟踪）；RPMArt（IROS 2024，噪声点云关节+可供点）；Online Articulation（AuRo 2026，视觉先验+本体感觉在线融合）；FlowBot++（逐点运动流+关节参数）；GPS（CVPR 2026，部件几何主结构）；FormNet（IROS 2021，单帧关节残差流）。

**通用性**：与 RSS 2025 的“感知瓶颈在通用机构估计而非末端控制”一致；**禁止** `if object == drawer` 的分支。

### 3.3 C3 控制/接触原语 — `controlled(o)`

复用已有共同运动、接触/附着、跨视角 `possible_control`；继续遵守“候选概率≠附着概率”。  
通用定义：目标相对夹爪的相对位姿稳定 + 独立接触/世界运动证据，不依赖对象名。

### 3.4 时序 — `stable(·)`

已有生命周期与 `goal_progress` 窗口/确认机制即可；关系谓词连续成立 \(K\) 个不重叠窗才标 `goal_stable`。

---

## 4. 语义层如何保持通用

### 4.1 操作前共识初始化（不靠单一开放词汇框）

1. 通用 VLM（Qwen3-VL / InternVL / Florence / RoboPoint 类）提出实体与关系假设。
2. 类无关掩膜提议 + 静止背景/共同运动候选（已有）。
3. **多证据一致才锁定**；冲突 → `unknown`，不进入危险授权。
4. 锁定后走已有生命周期（禁止操作中换目标）。

FailBench：全图裸问 VLM 不可靠；**先 ROI 再问**更好。故假设与复核都裁剪到候选区域。

### 4.2 低频 VLM 复核（通用问答，不是场景规则）

- 输入：语言 + 相关实体 ROI + 前后帧（可选腕部）。
- 输出：已有 `goal_vlm_output` 结构（achieved/not/unknown + 框 + 阶段 + 理由）。
- 触发：阶段切换、测量冲突、长时间无进展、重规划前。
- 权限：只进 `SemanticConsequenceEvidence`；与测量冲突 → `ambiguous`；默认 shadow。

### 4.3 可选增强：通用关系/状态 VQA 头（仍保持通用）

若纯几何原语在细语义（on/off 旋钮、复杂容器）不足，可在**通用数据**上微调一个状态问答头，而不是写规则：

- 数据：Robo2VLM 式自动 VQA、OSCaR / VidOSC / HowToChange（开放世界物体状态变化）、LIBERO 成功示范 + 自然失败（仅离线标签）。
- 目标：成对进展排序 / 状态分类 + 弃权（RoboReward 负例配方：时间截断、反事实重标）。
- **不得**按 LIBERO 对象名过拟合；评测须含未见对象/布局/关系。

---

## 5. 文献如何约束“通用”

| 结论 | 来源 | 对架构的含义 |
|---|---|---|
| VLM 判成败上限 ~0.77，模糊时偏报成功 | FailBench 2026 | VLM 只作假设/复核，主证据必须可校准 |
| 模块化 > 端到端；感知是瓶颈 | Opening Articulated Structures, RSS 2025 | 通用原语组合，不训场景专家 |
| 置信门控进展优于裸奖励 | RARM 2026 | 确认改善必须过误差界 |
| 成对进展监督优于单帧二分类 | Robot Critics 2026 | 若训头，用排序/进展而非 yes/no |
| 开放世界物体状态变化 | VidOSC CVPR 2024, OSCaR NAACL 2024, HowToChange | 状态语义要跨物体共享表示 |
| 通用空间关系 | RelatiViT ICLR 2024 | in/on/相对位姿用统一关系模型 |
| 机构构型可类别无关估计 | IROS 2022 factor graph, RPMArt, FlowBot++ | config 原语按 DOF，不按物体类 |

---

## 6. 三条路线在“通用”尺度下重新对照

| | A 端到端 VLM 判目标 | B 场景化测量表 | **C 通用原语 + VLM 假设/复核** |
|---|---|---|---|
| 跨对象/任务 | 理论上广 | 差（每场景一把适配器） | **广（按关系类型实现）** |
| 可校准/弃权 | 差 | 好但会过拟合 | **好** |
| 是否含对象/场景分支 | 无 | 有 | **无** |
| 工程复用 | 低（黑盒） | 低 | **高（接现有生命周期/进展）** |
| 文献支持 | FailBench 否决作主证据 | RSS25 否决孤立特化 | **RSS25 + FailBench + RARM 共同支持** |

**决策：采用 C，且 C 的测量层严格实现为“关系类型原语”，拒绝任何对象分支。**  
前一版的“抽屉开合比例”只是 `config` 原语在棱柱 DOF 上的一个**评测用例**，不是实现规格。

---

## 7. 实施顺序（仍是 MVP，但通用）

| 阶段 | 交付 | 通用性验收（必须） |
|---|---|---|
| G0 | 谓词求值器 + 测量契约（状态量/误差界/unknown） | 单元测试用合成实体，不引用任务名 |
| G1 | C1 空间关系原语（in/on + 归一化相对位姿） | 至少 2 类容器/支撑、未见物体配置上评估 |
| G2 | C2 构型原语（DOF 估计 + 语义极值标签） | 棱柱 + 旋转两类构型；禁止按对象名分支 |
| G3 | 共识初始化 + 生命周期接入 | 错误锁定率/弃权率分列；操作中不换目标 |
| G4 | 接入 progress / no_progress / 带谓词的重规划提示 | 组合目标（in ∧ config）不得部分满足即宣称完成 |
| G5 | 低频 VLM ROI 复核 + 投票弃权 | 相对纯测量的冲突消解；不单独控机 |
| G6 | 同服务配对端到端（多 suite） | 报 rescued/harmed/preserved/unresolved；**留出未见关系** |

G1/G2 可在已有 seed36 帧上离线验证；冻结阈值后再碰 seed38。  
端到端成功标准是**任务成功率与误伤**，不是某类检测 mAP。

---

## 7.5 常见 YOLO 方案是否适用

**结论：不能作主证据链，最多作可选的快速 ROI 提议器。**

| 变体 | 能给什么 | 对本项目的问题 |
|---|---|---|
| YOLOv8/v11 闭集检测/分割 | 已知类框/掩膜 + 置信度 | 类别固定，换对象/场景要重训 → 违反通用与禁对象特调；框级置信度不校准 |
| YOLO-World / YOLOE 开放词汇 | 文本查询框 | 与已否决的 Grounding DINO / Florence-2 同类；机械臂入框、对象混淆、负对照也出框；仍无 in/on、构型、进展 |
| YOLO + SAM 掩膜 | 更好的分割 | 身份保持有用（项目已用 SAM2），但**检测器语义错误会稳定传给跟踪** |
| 微调 YOLO 判“抽屉开/关” | 单类状态分类器 | 典型场景特调；不能泛化到门/盖/旋钮；与 `config(p,c)` 通用原语冲突 |

YOLO 解决的是“**图里有什么框**”，本项目要的是“**语言目标是否成立、是否在改善、能否弃权**”。即便检测完美，仍拿不到：

- `in` / `on` 关系（要两实体几何，不是两个框）
- `config` 构型连续量（开合/角度，不是类别）
- 误差界与 `unknown`（YOLO 置信度 ≠ 校准概率）
- 时序进展 `Δg`

**允许的边缘角色**（可选，非必须）：

1. 操作前把候选区域裁给 VLM（FailBench：ROI 裁剪有帮助）；
2. 与类无关掩膜/运动候选一起做**共识初始化的一票**，不能单票锁定；
3. 若某子集必须报闭集检测指标，可另训 YOLO 作**诊断基线**，不得进授权路径。

主链仍是：开放词汇假设 + 共识锁定 + SAM2 生命周期 + **关系类型测量原语**。

---

## 8. 明确禁止（防再次过拟合）

1. 禁止 `if object_name == ...`、`if task_id == ...`、固定像素 ROI。
2. 禁止为单条失败视频写专用规则或专用阈值。
3. 禁止把仿真关节/成功谓词作在线输入（只作离线标注）。
4. 禁止把单一开放词汇框或 VLM 自报分数当校准证据。
5. 禁止用 π0.5 latent / 反事实动作敏感性作目标解码。
6. 评测必须报告未见对象/布局/关系；只报 LIBERO 内数字不得宣称通用。

---

## 9. 与项目现有模块的对接

| 通用组件 | 对接已有代码 |
|---|---|
| 谓词解析 | `goal_relations.py`（继续声明式，可扩组合） |
| 身份生命周期 | `target_track_lifecycle.py` + `target_track_quality.py` |
| 测量契约 | `goal_measurement.py`（校准域、误差界、拒绝条件） |
| 空间/构型原语 | 新增 `vla_supervisor/relation_primitives.py`（纯测量，无状态机） |
| 进展 | `goal_progress.py` / `goal_progress_adapter.py` |
| 语义后果 | `semantic_consequence.py` + `CalibratedSemanticConsequenceAdapter` |
| VLM 复核 | `goal_vlm_output.py`（严格解析，calibrated=false） |
| 候选入口 | `visual_consequence.candidate_provider`（由共识初始化填充） |

新增代码必须：无任务名、可单测、失败返回 `unknown` 而非零。

---

## 10. 关键文献（通用分析向）

**VLM 成败与进展（不可作唯一证据）**
1. FailBench, arXiv:2609.03611, 2026  
2. RoboReward, arXiv:2601.00675, 2026  
3. Robot Critics that Sweat the Small Stuff, arXiv:2606.21572, 2026  
4. Robo-Dopamine / 2.0, arXiv:2512.23703 / 2608.15680  
5. RARM, arXiv:2606.22027, 2026  
6. SOLE-R1, arXiv:2603.28730, 2026  

**通用空间/物体状态**
7. RelatiViT: Can Transformers Capture Spatial Relations?, ICLR 2024, arXiv:2403.00729  
8. Learning Object State Changes in Videos: Open-World (VidOSC/HowToChange), CVPR 2024, arXiv:2312.11782  
9. OSCaR: Object State Captioning and State Change Representation, NAACL 2024, arXiv:2402.17128  
10. SPOC: Spatially-Progressing Object State Change Segmentation, WACV 2026, arXiv:2503.11953  
11. OSSA: Object State-Sensitive Neurorobotic Task Planning, ICANN 2024, arXiv:2406.09988  
12. Robo2VLM, NeurIPS 2025  

**通用机构构型（DOF 级，非对象级）**
13. Opening Articulated Structures in the Real World, RSS 2025, arXiv:2402.17767  
14. Category-Independent Articulated Object Tracking with Factor Graphs, IROS 2022, arXiv:2205.03721  
15. RPMArt, IROS 2024, arXiv:2403.16023  
16. Online Estimation and Manipulation of Articulated Objects, AuRo 2026, arXiv:2601.01438  
17. FlowBot++, arXiv:2306.12893  
18. GPS: Revisiting Articulated Parts Perception, CVPR 2026, arXiv:2606.08103  

---

## 11. 一句话决策

**做通用语言条件视觉状态分析：语言只产生可组合谓词；测量层只按关系类型（空间 in/on、构型 config、控制 controlled、时序 stable）实现几何/运动原语；VLM 仅负责开放词汇假设与低频复核；全程误差界与弃权。任何对象名、任务 ID、场景规则都视为设计违规。**
