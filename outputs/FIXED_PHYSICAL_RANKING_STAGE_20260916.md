# 固定视角运动候选物理排序阶段报告（2026-09-16）

## 目的

固定相机首先估计静止背景，并提出相对背景独立运动的候选物体；这一层不要求预先知道任务语义目标。此前至多两个候选已经提高了召回率，但跨视角桥接会被初始帧颜色/纹理相似度误导。本阶段检验能否用抓取事件后的物理响应对集合内候选排序。

## 冻结规则

对每个固定视角候选，读取夹爪闭合事件后40帧窗口中的最大位移；在上游冻结候选集合中选择该值最大的候选。该分数不使用任务成功标签、目标类别或标注身份。

## 开发集结果（Wave2）

- 候选集合包含完整真实物体：17/20。
- 闭合后最大位移排序：15/17（88.2%）。
- 整段最大净位移排序：14/17（82.4%）。
- 接入固定—腕部完整桥接后的成对正确：14/20。

## Wave3回顾性诊断

- 候选集合包含完整真实物体：12/17个可观察案例。
- 闭合后最大位移排序在这12例中为12/12。
- 固定视角正确数由旧v2的8/20提高到12/20。
- 完整固定—腕部成对正确由7/20提高到10/20。
- 剩余失败可拆为：上游候选未包含真值5例；腕部附着选择即使在正确固定物体条件下仍失败2例。

## 解释限制

Wave3已经在规则形成过程中被查看，因此12/12与10/20均为回顾性诊断，不能作为独立前瞻泛化结果。下一次正式测试需要先冻结代码和阈值，再采集Wave4或使用未查看的独立场景。

## 产物

- `cross_suite_generalization/compare_fixed_proposal_rankers.py`
- `cross_suite_generalization/evaluate_cross_view_identity_bridge.py`
- `outputs/FIXED_PROPOSAL_RANKER_AUDIT_WAVE2.json`
- `outputs/FIXED_PROPOSAL_RANKER_DIAGNOSTIC_WAVE3.json`
- `outputs/CROSS_VIEW_IDENTITY_BRIDGE_PHYSICAL_WAVE2.json`
- `outputs/CROSS_VIEW_IDENTITY_BRIDGE_PHYSICAL_WAVE3_DIAGNOSTIC.json`
