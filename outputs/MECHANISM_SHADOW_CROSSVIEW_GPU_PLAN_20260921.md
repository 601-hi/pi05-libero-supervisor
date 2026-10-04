# 机制 shadow 第二轮最小 GPU 计划

日期：2026-09-21  
前提：第一轮12条轨迹和冻结腕部候选评分已经完成；本计划不重跑π0.5，不读取任务成败来选择样本或参数。

## 目的

一次GPU启动完成两个尚不能由CPU补齐的视觉量：

1. 对5个“唯一且超过冻结候选质量门”的腕部掩膜，从首次闭爪一直跟踪到轨迹末尾，用于检测后续丢失；
2. 在固定相机同一闭爪帧生成候选并进行闭爪前15帧、后40帧双向跟踪，用于建立独立的世界运动证据。

两项输出都不包含任务语言、reward或episode outcome。即使二者一致，也只输出`possible_control`，不能推出抓的是目标物体或任务成功。

## GPU任务A：腕部候选跟踪至结尾

```bash
cd /root/gpufree-data/supervisor-tools-v1
HF_HOME=/root/gpufree-data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/gpufree-data/vision-env/bin/python \
  cross_suite_generalization/run_mechanism_shadow_selected_wrist_full_gpu.py \
  --public outputs/mechanism_shadow_sam2_inputs_20260921/public.json \
  --private outputs/mechanism_shadow_sam2_inputs_20260921/private.json \
  --scores outputs/MECHANISM_SHADOW_FROZEN_WRIST_SCORE_20260921.json \
  --candidate-dir outputs/mechanism_shadow_sam2_bidirectional_20260921 \
  --output-dir outputs/mechanism_shadow_selected_wrist_full_20260921
```

中断后可增加`--resume`；已有最终文件时拒绝覆盖。

## GPU任务B：固定相机闭爪双向候选

```bash
HF_HOME=/root/gpufree-data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/gpufree-data/vision-env/bin/python \
  cross_suite_generalization/run_wrist_close_bidirectional_gpu.py \
  --public outputs/mechanism_shadow_sam2_inputs_20260921/public.json \
  --private outputs/mechanism_shadow_sam2_inputs_20260921/private.json \
  --features outputs/mechanism_shadow_sam2_inputs_20260921/features.json \
  --camera-key agent_images \
  --output-dir outputs/mechanism_shadow_fixed_bidirectional_20260921
```

## GPU后CPU处理

- 使用Wave2冻结稳定性阈值诊断全程腕部稳定、丢失和恢复；
- 从固定视角相对轨迹构造背景独立运动组；
- 用`fuse_mechanism_shadow_crossview.py`输出保守跨视角状态；
- 最后才打开任务语言和成败，审计成功轨迹假报警及三个失败机制；
- 所有可行动结论继续保持shadow，未经过新的baseline/control成对闭环前不得控制机器人。
