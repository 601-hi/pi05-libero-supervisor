# 机构关系监督器在线 GPU 运行手册（2026-09-27）

## 现在已经完成什么

离线视觉已证明：对 LIBERO-90 task 0，`open top drawer` 的状态感知查询加 SAM2 面积进展量，能在冻结的小样本回放中保持 2/2 成功轨迹，并发现 4/4 长期不关抽屉的失败轨迹。CPU 运行时已经把该证据接入统一监督、时序融合与 `restore_goal_relation` 恢复路由。

新增的 `ValidatedArticulationProvider` 是在线感知与控制系统之间的防火墙。后端只提供面积、置信度、可观测性、身份可靠性和标定身份；缺字段、非法数值、旧帧、标定不匹配和后端异常均变成 `unknown`，不能触发控制。

## 为什么现在不能直接打开控制

离线回放没有验证三件事：

1. DINO 与 SAM2 和 π0.5 同时驻留 GPU 时是否显存足够；
2. 在线逐帧状态与离线批量跟踪结果是否一致；
3. 感知延迟是否会改变策略查询时序，从而破坏基线/控制的因果配对。

因此下一次有卡开机必须按“资源与回放等价性 → 在线影子 → 小规模控制”推进，不能跳过影子阶段。

## 冻结实验

机器可读方案见 `ARTICULATION_ONLINE_PAIRED_GPU_PROTOCOL_20260927.json`。固定 task 0 的五个初始状态，并为每对指定独立且固定的 π0.5 采样噪声。选择是在结果揭盲前完成的，不能根据哪条失败再增删样本。

### 阶段 A：资源与回放等价性

- 加载 π0.5、Grounding DINO 和 SAM2；记录峰值显存及单帧/单步延迟；
- 用已有 NPZ 帧逐帧喂给在线后端；
- 比较在线面积序列和已冻结离线面积序列；
- 若 OOM，不降低科学标准去偷偷换结果，而是改成 DINO 只在初始化/身份复核时加载，或把视觉进程与策略进程分时运行后重新验证延迟边界。

冻结命令（在 supervisor-tools-v1 根目录执行）：

```bash
HF_HOME=/root/gpufree-data/hf-cache \
/root/gpufree-data/vision-env/bin/python \
scripts/validate_articulation_online_parity.py \
  --predictions \
  outputs/task0_close_relation_20260927/sam2_counterstate/predictions.json \
  outputs/task0_close_relation_20260927/sam2_counterstate_more/predictions.json \
  --output outputs/task0_close_relation_20260927/online_image_parity.json
```

冻结准入为：每条轨迹进展曲线相关系数不低于0.85、平均绝对进展误差不高于0.10、最终时序事件类型一致，并且每帧身份可靠。任一记录失败则整体失败，不允许看完结果后调门槛进入控制。

### 阶段 B：在线影子

- baseline 与 shadow 使用相同初始状态、环境 seed、sampling-noise seed；
- shadow 可以记录假想报警，但不得清空动作块、改变重规划频率或执行恢复；
- 检查假想事件发生前，动作块、末端状态和噪声哈希必须逐项一致；
- 成功轨迹出现可操作关系报警，立即停止，不进入控制阶段。

### 阶段 C：在线控制小试验

只有 A、B 均通过才开放 `articulation_control_enabled`。事件仍须经过统一时序融合、干预预算和安全恢复；视觉关系监督器不直接输出动作，也不能宣布任务完成。

最终按配对 episode 报告：

- `rescued`：基线失败、控制成功；
- `harmed`：基线成功、控制失败；
- `preserved_success`：两者均成功；
- `unresolved_failure`：两者均失败；
- `abstained`：证据不可靠而未介入。

首轮五对只用于验证框架闭环，不能宣称跨任务泛化。后续 Open/Close 场景必须用关系本体中的预注册任务扩展，不能围绕已见测试结果逐个特调。

## GPU 阶段尚需实现的唯一关键部件

现已准备好隔离进程版本：`scripts/run_articulation_vision_sidecar.py` 在 Python 3.12 `vision-env` 中承载 DINO/SAM2；LIBERO Python 3.8 主循环通过本机 TCP 客户端接收测量。超时、断连或畸形结果自动弃权且不重试。监控主程序已有 `frozen_four_state_articulation` 与显式 `shadow/control` 开关。服务端默认 `calibrated=false`；阶段A通过后才允许显式加入 `--calibrated`。全量CPU回归为472项通过。
