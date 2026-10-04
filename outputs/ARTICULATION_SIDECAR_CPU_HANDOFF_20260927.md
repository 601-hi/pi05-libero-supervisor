# 机构关系视觉 Sidecar CPU 交接节点（2026-09-27）

## 解决的问题

SAM2/DINO位于Python 3.12 `vision-env`，LIBERO控制循环位于Python 3.8环境，无法安全地在同一解释器直接导入。本阶段采用本机进程隔离，不改动两个既有环境。

## 已实现

- `vla_supervisor/articulation_provider.py`：验证感知字段、标定ID、身份可靠性、时序新鲜度；失败即弃权。
- `vla_supervisor/articulation_sidecar.py`：256像素固定相机JPEG的单请求TCP客户端，默认250毫秒超时；无重试，不允许视觉延迟复制机器人动作。
- `scripts/run_articulation_vision_sidecar.py`：vision-env中的DINO初始化与SAM2逐帧图像提示服务；默认未标定。
- `remote_stage_20260923/monitoring_main.py`及服务器正式监控脚本：新增`frozen_four_state_articulation`模式和独立的视觉控制权限开关。
- `scripts/validate_articulation_online_parity.py`：在五条冻结轨迹上比较逐帧在线实现与旧视频跟踪的进展曲线及最终事件。
- 五对在线配对身份、阈值、输入防火墙和停止规则已经预注册。

## 验证

- 新增provider/IPC/运行时定向测试13项通过；
- 服务器全项目472项通过；
- LIBERO Python 3.8监控脚本、vision-env Python 3.12服务端和等价性检查器均通过语法/CLI检查；
- 尚未加载GPU模型，因此没有把sidecar声明为视觉有效，也没有打开`calibrated`或控制权限。

## 下一边界

下一次有卡开机只先执行资源共存和五条冻结轨迹等价性。若失败，停在感知实现修复；若通过，再进入在线shadow。不得直接运行控制组。
