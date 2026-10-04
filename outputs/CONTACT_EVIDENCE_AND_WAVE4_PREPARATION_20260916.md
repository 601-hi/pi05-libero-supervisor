# 接触证据覆盖审计与Wave4准备节点

## 已完成

- 新增严格覆盖审计器，不把原始几何残差或阈值事件自动转换成概率。
- Wave3共20条：动作四态覆盖20/20，背景置信度覆盖20/20，存在附着建立事件20/20；完整状态机所需的校准逐步字段覆盖0/20。
- Wave1+Wave2生成40条Pass A物理盲标清单和40条Pass B后置语义清单。
- Pass A不含任务语言；Pass B必须引用冻结后的物理标注，不能反向修改受控对象。
- 形成Wave4前瞻协议候选与完整消融、指标和数据治理要求。
- 服务器完整回归149/149通过，全程未加载GPU模型。

## 当前缺口

1. Wave1/2匿名双视角视频仍需按协议完成人工物理事件标注。
2. 需要在这些开发标注上校准夹爪—候选接近、独立运动和固定受阻概率。
3. 腕部速度/面积事件需要校准为逐步附着概率或保持结构化离散证据，不能混称概率。
4. π0.5语义表征可读性实验需要GPU；在此之前语义字段保持缺失并触发弃权。
5. 所有模型和阈值冻结后才能生成Wave4，Wave3不再承担测试作用。

## 后续无卡完成项：匿名视频与概率校准框架

- 40/40条Wave1/2私有sidecar均成功映射到匿名开发ID。
- CPU生成40个固定—腕部并排盲审视频，共5882帧、448×224、约25MB；逐视频帧数与索引一致，文件名只有匿名ID。
- 视频只显示固定RGB、腕部RGB、视觉帧号、动作索引和闭爪事件，不显示语言、结果、奖励、专家分数或算法候选。
- 审阅包`CONTACT_BLIND_REVIEW_WAVE12_V1.tar.gz`大小约24MB，SHA-256为`04ec29e242ad7c3f8b431eb48fa0b5f06f7ba7d4f22a7d5bff5129ab29194f91`。
- 标注模板已为每个闭爪事件预建字段；校验器对未填写模板返回非零状态并报告368项缺失/非法字段，防止半成品进入训练。
- 新增纯NumPy分组逻辑概率校准器：只允许Wave1/2，按episode分组交叉验证，输出AUC、Brier、log-loss和10-bin ECE；任何Wave3/Wave4输入直接拒绝。
- 概率模型只校准证据，不移动状态机已预注册阈值；完整回归153/153通过。

## 文件

- `cross_suite_generalization/audit_contact_evidence_coverage.py`
- `cross_suite_generalization/build_contact_annotation_manifests.py`
- `outputs/CONTACT_EVIDENCE_COVERAGE_WAVE3.json`
- `outputs/CONTACT_PHYSICAL_ANNOTATION_MANIFEST_WAVE12.json`
- `outputs/CONTACT_SEMANTIC_ANNOTATION_MANIFEST_WAVE12.json`
- `outputs/CONTACT_EVENT_ANNOTATION_PROTOCOL_WAVE12_20260916.md`
- `outputs/WAVE4_CONTACT_GRASP_PREREGISTRATION_V1_20260916.json`
