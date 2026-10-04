# 回退 MVP GPU v12 定向配对（2026-10-02）

## 实验目的

只运行 LIBERO-90 task 0 的 episode 0、2，并在同一策略服务生命周期中依次运行 strict 与 MVP。两组固定初态、环境随机种子与 pi0.5 采样噪声。episode 0 用于验证重复计划的有界拒绝；episode 2 原计划用于验证变化计划获准连续执行 25 个策略动作后，是否仍会被回退阶段规则误打断。

## 结果

| episode | strict | MVP | 因果前缀 | MVP 新颖性结果 |
|---:|---|---|---|---|
| 0 | 失败，113 步 | 失败，82 步 | 前 82 步动作与末端位姿逐值一致 | repeated × 3，层级 0/1/2 后安全停止 |
| 2 | 失败，540 步 | 失败，414 步 | 前 82 步动作与末端位姿逐值一致 | repeated × 3，层级 0/1/2 后安全停止 |

汇总：strict 0/2，MVP 0/2；6 次重规划全部判为 repeated，0 次 changed；两条 MVP 均以 `repeated_replan_no_deeper_checkpoint` 有界停止。GPU 服务在实验结束后已清理。

## 关键诊断

episode 0 的三级拒绝再次证明层级边界修复有效。episode 2 却没有复现 v10 的变化计划：v10 在 action 106 重规划，计划相对 L2 为 0.447、逐步方向余弦中位数为 0.955，因此判为 changed；v12 到 action 414 才重规划，三个候选的相对 L2 仅 0.169--0.188，方向余弦约 0.982--0.990，累计方向余弦约 0.989--0.998，均高度复现原失败动作。

因此，本轮没有实际进入新增的 25 步“新颖性独占窗口”，不能据此评价窗口能否提升任务成功率。后续逐状态审计还发现，v12 的稀疏 `0,2` 调度跳过了 episode1 应消耗的环境 reset；所以 v10 与 v12 的 episode2 环境前缀并不相同。本轮只能证明 v12 内部 strict/MVP 配对有效，不能把它与 v10 的交接时间差解释成回退器退化。

## 决策

不为了得到正结果而直接放宽新颖性门。把 17%--19% 的幅值变化、几乎相同的方向轨迹称为新计划，会违背该门的因果目的。先修复稀疏调度的 reset 序列并验证 v10 环境前缀能否复现；只有对照条件一致后，才比较交接点并验证 25 步窗口。详见 `SPARSE_EPISODE_RESET_REPRODUCIBILITY_CPU_20261002.md`。

## 证据

- `outputs/rollback_mvp_gpu_v12_20261002/audit.json`
- `outputs/rollback_mvp_gpu_v12_20261002/strict.log`
- `outputs/rollback_mvp_gpu_v12_20261002/mvp.log`
- 远端完整轨迹：`/root/gpufree-data/libero-traces/goal_relation_rollback_mvp_v12_ep0_ep2_paired`
