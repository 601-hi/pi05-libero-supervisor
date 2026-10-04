# 稀疏 episode 调度的环境 reset 可复现性修复（2026-10-02）

## 问题

GPU v12 为节省费用只调度 episode `0,2`。评测器原来只对被执行的 episode 调用 `env.reset()`，因此编号2实际经历的是第二次 reset；完整 `0,1,2,3,4` 批次中的编号2则经历第三次 reset。`initial_states[episode_idx]` 不能消除这一差异，因为 LIBERO 在 `set_init_state` 之前的 reset 仍会消耗已播种的环境随机状态。

这使得 v10 与 v12 的所谓 episode 2 并非同一环境轨迹。证据包括 action78 的末端位置和关节状态明显不同；v10 在 action78 仍有5.48 mm环境间隙并录入安全历史点，v12 同位置对应的实际状态已不同，环境距离为负并拒绝历史点。因此，v12 得到 action414 的交接和重复计划，不能直接归因于回退器相对 v10 退化。

固定 pi0.5 扩散噪声只固定策略采样，不会自动固定环境 reset 序列。这一边界现已明确写入评测契约。

## 修复

新增轻量模块 `examples/libero/episode_schedule.py`。稀疏调度 `0,2` 时：

1. episode0 消耗第一次 reset 并执行；
2. 跳过 episode1的策略推理和仿真步骤，但仍消耗第二次 reset；
3. episode2消耗第三次 reset再执行。

索引必须严格递增且不得重复。默认连续调度行为不变。这个修复只影响评测调度，不进入策略或监督器输入，也没有用测试任务信息调参。

## 验证

- LIBERO Python 3.8 对 `monitoring_main.py` 与新模块语法检查通过；
- 调度、新颖性与历史点准备定向测试：23 passed；
- 可运行 CPU 回归：401 passed；
- 另有6个测试文件因系统Python缺少可选视觉、MuJoCo或openpi依赖而按既有口径排除，不属于失败测试。
- 新增 `outputs/audit_sparse_episode_reproducibility_20261002.py`。它对旧 v10/v12 episode2 的前82步给出：噪声哈希相同，但动作从第0步、末端从第1步、关节从第0步即不同，最大动作差0.439、末端差55.7 mm、关节差0.267 rad，明确否证“固定扩散噪声即可视为同一轨迹”。

## 下一次最小 GPU 验证

仍只运行 episode `0,2`，但新调度会补偿被跳过的 reset。首先比较 v10 与新轨迹在 action0--81 的动作、末端、关节和接触证据；只有 episode2环境前缀真正复现，才评价回退交接、新颖性门和25步独占窗口。若仍不一致，就必须把环境随机状态显式序列化，而不能继续用 episode编号假定相同。
