# GPU 恢复后的最小运行手册

## 阶段 A：安装候选补丁并启动服务

先备份远端三个源文件，再把 `patched/` 中对应文件覆盖到 openpi。运行 `py_compile`。启动
`pi05_libero` server；首次视觉旁路调用会额外 JIT，预期明显慢于后续调用。

## 阶段 B：只做 1+1 条动作不变性 smoke test

固定 `seed=7`、`sampling_noise_seed=20260904`、task0、1 episode、无扰动：

- A：不传 `visual_latent_out_path`；
- B：传 `visual_latent_out_path`；
- 两者 trace 使用不同文件名。

运行 `compare_action_invariance.py --off A.jsonl --on B.jsonl`。只有三个 max diff 均为 0 才继续。
再运行 `verify_visual_latent_artifacts.py` 检查 B。

## 阶段 C：采集矩阵

按 seed18/19/20/21 采集正常与 scale025/050/075 条件，沿用既有固定噪声、扰动日程和 episode 数。
每个条件独立 trace 及 latent 目录。先验证每个 batch，再加入训练清单。

## 阶段 D：冻结特征与训练小头

严格按 `EXPERIMENT_MATRIX.json`：固定随机投影、训练标准化只看 seed18/19；seed20 选择 head、margin、阈值；
锁定配置后只运行一次 seed21。最终与双专家、normal-only、旧 v2 做 paired episode comparison。

## 停止条件

动作不变性失败、NPZ/JSONL 对齐失败、正常 FP 超过 1/23，任一发生均立即停止并诊断，不继续扩大采集。
