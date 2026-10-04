# pi05-supervisor-v0.1-20261004

本目录冻结 2026-10-04 首次视频确认成功时的实际运行版本。

本次故障是 π0.5 在无动作缩放、无人工推移和无随机故障注入条件下自然形成的任务失败。为了固定因果起点，成功实验从自然失败时保存的 action-81 快照启动。运行采用 `articulation_isolated_control`：两次回退均由视觉关系监督的 `object_failure` 触发，而不是由双专家触发；具体回退动作则由恢复状态机和机器人学候选规划器产生。

## 内容

- `server/monitoring_main.py`：LIBERO 在线执行、监督、回退和重规划入口。
- `server/vla_supervisor/`：服务器实际使用的监督器 Python 包。
- `server/websocket_policy_server.py`：支持显式固定采样噪声的策略服务协议实现。
- `server/fault_snapshot.py`：完整 MuJoCo 状态与恢复计划的原子保存/恢复。
- `server/run_staged_gate_snapshot_gpu_v30_20261004.sh`：基础运行脚本。
- `server/run_staged_gate_snapshot_gpu_v32_progress_replan_20261004.sh`：9.5 cm 交接配置包装器。
- `artifacts/snapshots/fault_action0081.npz`：原始故障状态。
- `artifacts/snapshots/second_start_failure_action0420.npz`：确定性失败的第二轮起点。
- `artifacts/snapshots/second_start_success_action0405.npz`：最终成功运行的第二轮起点。
- `artifacts/results/v37_full_episode_supervision_recovery_success.mp4`：推荐展示的视频，由产生自然故障快照的严格配对前缀与 action-81 快照恢复段拼接；它不是一次未中断的单进程录像。
- `artifacts/results/v37_second_rollback_095_replan_success.mp4`：从 action-81 故障快照启动的原始、未经剪辑的成功实验视频，用于保留直接实验凭证。
- `SHA256SUMS`：关键文件完整性校验。
- `RESULT.json`：最小机器可读结果摘要。

## 依赖边界

本目录不是 OpenPI 或 LIBERO 的完整镜像。运行时仍需用户自行按上游许可安装 OpenPI、π0.5 LIBERO checkpoint、LIBERO、robosuite 与 MuJoCo。OpenPI 基线仓库当时的 HEAD 为：

`15a9616a00943ada6c20a0f158e3adb39df2ccac`

服务器 OpenPI 工作树包含本项目修改，因此应以 `SHA256SUMS` 中冻结文件为准。

## 结果边界

该版本证明单个困难场景的完整恢复链可以工作，但不声称已经获得总体成功率、跨任务泛化或真机安全保证。

双专家此前已在正常和动作缩放异常数据上完成条件动力学似然建模与独立评测，但没有取得本次成功案例的控制权。其结果支持继续研究执行一致性异常，不足以证明自然任务失败检测。未来应由双专家负责高频动作—响应一致性与卡滞/受阻证据，视觉负责目标身份、物体后果和任务关系进展，再由融合器对一致、分歧、重叠和域外证据分别路由。完整说明见仓库根目录 `DETECTION_EXPERT_ROLES.md`。

这是面向单场景的最小闭环冻结版，不是最终通用监督器。第一轮恢复用于失败且无进展/低进展状态，强调任务无关的安全撤离；第二轮恢复用于已经形成部分任务进展后的失败，需要保护已完成状态，因此允许按统一方法对不同场景进行少量参数微调。

当前安全候选评分读取 MuJoCo 物体和几何真值，仅用于仿真机制验证。真机版本必须以电机电流/力矩、触觉或力传感器、激光雷达标定、深度信息标定、占据地图或其他实测碰撞距离证据替换，不得把仿真 oracle 当作可部署输入。

本冻结版本属于项目作者的原创研究成果。任何使用、修改、展示或学术衍生工作都必须注明作者和仓库来源；第三方组件仍归其各自权利人所有。

