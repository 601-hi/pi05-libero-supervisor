# 机制 shadow：双向 SAM2 最小 GPU 运行单

日期：2026-09-21  
角色：预注册的 GPU 补算，不更改候选生成、评分公式或控制策略。

## 为什么必须补算

第一次 SAM2 结果只从闭爪帧向后跟踪。冻结腕部候选评分的主要证据是闭爪前后运动统计的变化，权重为 0.55；缺失闭爪前轨迹时不得计算完整分数。此次只补齐同一批12条冻结轨迹的15帧闭爪前和40帧闭爪后跟踪，不运行 π0.5，不改变任务、seed、候选阈值或轨迹结果。

校准概率的语义严格限定为“掩膜是可接受的受控物体候选”的概率，不是抓取成功、物理附着或目标身份概率，不能直接触发释放、退避或重规划。

## GPU 命令

```bash
cd /root/gpufree-data/supervisor-tools-v1

HF_HOME=/root/gpufree-data/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/root/gpufree-data/vision-env/bin/python \
  cross_suite_generalization/run_wrist_close_bidirectional_gpu.py \
  --public outputs/mechanism_shadow_sam2_inputs_20260921/public.json \
  --private outputs/mechanism_shadow_sam2_inputs_20260921/private.json \
  --features outputs/mechanism_shadow_sam2_inputs_20260921/features.json \
  --output-dir outputs/mechanism_shadow_sam2_bidirectional_20260921
```

若进程中断且只有 `predictions.partial.json`，原命令末尾增加 `--resume`。若已经有 `predictions.json`，脚本按设计拒绝覆盖。

## 输出契约与 CPU 评分

GPU完成后先验证，不通过则禁止评分：

```bash
/usr/bin/python3 scripts/validate_mechanism_shadow_bidirectional.py \
  --predictions outputs/mechanism_shadow_sam2_bidirectional_20260921/predictions.json \
  --public outputs/mechanism_shadow_sam2_inputs_20260921/public.json
```

验证通过后应用冻结协议：

```bash
PYTHONPATH=. /usr/bin/python3 \
  cross_suite_generalization/score_frozen_wrist_candidates.py \
  --predictions outputs/mechanism_shadow_sam2_bidirectional_20260921/predictions.json \
  --protocol outputs/WRIST_PHYSICAL_EVIDENCE_FROZEN_CANDIDATE_20260917.json \
  --output outputs/MECHANISM_SHADOW_FROZEN_WRIST_SCORE_20260921.json
```

冻结时间定义：闭爪前最多15帧（计算时排除闭爪帧）；闭爪后先跳过5帧，再用25帧计算变化；可见度仍按协议使用闭爪后第5至19帧。二者是两个不同统计量，不能混为同一个窗口。

## 运行前检查（已完成）

- 公开、私有、特征清单均为12条且匿名ID一一对应；
- 10条存在闭爪转换，2条按协议输出无候选；
- 本地 SAM2 缓存存在，运行强制离线；
- 目标输出目录不存在，不会覆盖旧结果；
- 数据盘可用约10 GiB，高于6 GiB安全线；
- 无策略服务、采集器或旧 SAM2 进程运行；
- 当前无卡实例没有 CUDA，因此此运行单尚未执行。
