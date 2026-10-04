# 因果物体身份留出验证协议 v1

日期：2026-09-13  
状态：预测前冻结；下一步需要 GPU 执行 wave 1。

## 研究问题

在不读取任务结果、仿真状态或未来帧的前提下，多候选 SAM2 跟踪与因果运动锁定能否识别当前真正被操纵的刚体对象，并在不确定时保持拒判？

## 数据划分

冻结视觉清单原有 110 个唯一 episode。开发试点使用了五种目标语言各自的第一条 episode，已按完整 episode key 精确排除。剩余 105 条分为：

- 90 条 `rigid_object_transport`：本协议的主要留出池；
- 15 条 `articulated_environment`：关闭抽屉，保留给独立铰接状态专家，不与刚体身份准确率混算。

90 条刚体 episode 使用固定盐值对 `(suite, goal language, task id, episode index)` 做 SHA256 排序，分成 20/20/20/20/10 五波。排序不读取 success/failure、动作质量或既有监督器分数。

## 泄漏隔离

公开清单只含匿名 ID、任务语言、机制族、哈希顺序和波次。自动检查确认其样本记录不含 `success`、`failure`、`outcome`、服务器路径或 `.npz`。真实 sidecar 路径和事后结果只保存在私有审计映射；GPU 外层负责读取 RGB，视觉模型不可访问私有元数据。

## 冻结感知与锁定参数

- 模型：`facebook/sam2.1-hiera-small`
- 自动掩码：`points_per_side=24`、`pred_iou_thresh=0.72`、`stability_score_thresh=0.82`
- 候选面积：图像的 0.2%–8%
- 每条 episode 最多 12 个候选
- 因果锁定：归一化累计运动至少 0.03，领先第二名至少 0.02，连续 3 帧
- 无充分证据时输出 unknown；首次锁定后身份保持黏性

这些参数来自五条开发试点，不允许根据 wave 1 结果修改后再重报 wave 1。若未来修改，必须升协议版本并在下一未打开波次验证。

## 主要指标

1. 可观察 episode 上的正确锁定率；
2. 全部 episode 的错误锁定率；
3. 全部 episode 的 unknown/拒判率；
4. 从人工标注“身份首次可观察帧”到正确锁定的延迟；
5. 按波次报告，不把同一波次重复用于调参和最终结论。

错误锁定比拒判更危险，因此不能以提高覆盖率为理由强制选择候选。人工标注只看匿名可视化，不看 episode 成败；标签是“实际操纵候选 ID / 不可观察”，不是“任务成功/失败”。

## 工程状态

- 冻结清单构建器：`cross_suite_generalization/build_causal_identity_holdout.py`
- GPU 批处理器：`cross_suite_generalization/run_causal_identity_holdout_gpu.py`
- 独立评估器：`cross_suite_generalization/evaluate_causal_identity_holdout.py`
- wave 1 dry-run：20/20 sidecar 可读取
- 项目回归测试：74/74 通过

冻结清单 SHA256：

- public：`c1a149aa6a54720e29a0032dc90676e2fc5fbd9d93eee603ef2abce70f758993`
- private audit：`990c3fcc5a814fd99d68adb1fed726514690068d8bd1ccdc7e06df81c201d4e2`
- blind annotation template：`898da72aa7cb158234ee19e004b55a89c691eccdd933cc213afba66157af6cd0`

## 下一条 GPU 命令

```bash
cd /root/gpufree-data/supervisor-tools-v1
HF_HOME=/root/gpufree-data/hf-cache \
HF_ENDPOINT=https://hf-mirror.com \
/root/gpufree-data/vision-env/bin/python \
  cross_suite_generalization/run_causal_identity_holdout_gpu.py \
  --public-manifest outputs/CAUSAL_IDENTITY_HOLDOUT_PUBLIC_V1.json \
  --private-map outputs/CAUSAL_IDENTITY_HOLDOUT_PRIVATE_AUDIT_V1.json \
  --output-dir outputs/causal_identity_holdout_wave1 \
  --wave 1 \
  --family rigid_object_transport
```

运行 wave 1 后先完成盲化身份审阅和冻结评估，再决定是否将同一参数无修改地推进到 wave 2；不能根据结果挑选最漂亮的 episode。
