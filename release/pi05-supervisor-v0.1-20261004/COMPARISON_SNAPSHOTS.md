# 对比研究快照清单

三个快照均为压缩 NPZ，包含完整 MuJoCo flattened state、mocap、actuator control、恢复准备计划和元数据。它们用于因果对比，不是训练数据。

| 文件 | 动作点 | 用途 | 已知结果 |
|---|---:|---|---|
| `artifacts/snapshots/fault_action0081.npz` | 81 | 所有主要回退版本共享的原始故障起点 | 可复现第一轮回退与后续重规划 |
| `artifacts/snapshots/second_start_failure_action0420.npz` | 420 | 固定第二轮起点的确定性失败对照 | 两次运行前183个提案逐字段相同，最终约5.08 cm处被碰撞门终止 |
| `artifacts/snapshots/second_start_success_action0405.npz` | 405 | 最终成功运行的第二轮起点 | 第二轮达到9.5165 cm后重规划并完成任务 |

## 推荐比较

1. 从 action 81 比较不同回退器版本，回答控制结构是否改善恢复。
2. 从 action 420 重复运行，检查恢复器本身的确定性和必败模式。
3. 从 action 405 重放最终版本，检查9.5 cm交接与重规划是否可复现。
4. 不应把 action 405 与 action 420 的结果差异全部归因于回退器；两者是第一次重规划后形成的不同物理状态，恰好用于研究重规划初态敏感性。

所有文件的 SHA-256 位于同目录 `SHA256SUMS`。

