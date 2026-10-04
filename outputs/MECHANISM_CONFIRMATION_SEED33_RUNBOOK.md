# 新数据机制监督器确认实验运行说明

## 数据治理

- 校准数据：LIBERO-10 odd tasks 1/3/5/7/9，新seed 33，每任务4条正常轨迹；只用于设备正常阈值。
- 确认数据：此前未打开的even tasks 0/2/4/6/8，每任务2条正常轨迹及每种scale 0.25/0.50/0.75各2条异常轨迹。
- 机制边界、4-of-7主规则及3-of-5次规则已经冻结。
- 确认评测禁止读取视觉sidecar。视觉被保存只为后续另立实验；一旦据此修改视觉方法，本批不能作为视觉最终holdout。

重要修正：scale是干预元数据而不是最终异常标签。采集后必须同时生成任务成功、完成时间、路径/恢复/振荡指标和视觉后果标签。机制监督器可先报告“干预窗口可检测性”作为诊断，但主要结论应改为对 degraded、task-critical 和 safety-critical 后果的识别。未造成可测退化的scale轨迹作为benign hard negative。

## GPU开机后

终端A启动策略服务器：

```bash
cd /root/gpufree-data/vla-workspace/openpi
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /root/gpufree-data/openpi-data/openpi-assets/checkpoints/pi05_libero
```

出现 `server listening on 0.0.0.0:8000` 后，终端B运行：

```bash
cd /root/gpufree-data/supervisor-tools-v1
bash cross_suite_generalization/run_mechanism_confirmation_seed33.sh \
  2>&1 | tee outputs/mechanism_confirmation_seed33_collection.log
```

脚本拒绝覆盖已有trace，并在端口8000没有监听时直接退出。预计共采集60条episode：20条正常校准、10条正常确认、30条异常确认。所有even-task确认轨迹保存逐步主相机和腕部相机sidecar。

## 无卡阶段已做的检查

只执行 `bash -n`、路径和CLI参数检查，不启动策略服务器或采集。正式采集完成后，先进行episode数量、step连续性、visual index对齐、固定噪声hash及扰动窗口完整性审计，再运行冻结确认评测。
