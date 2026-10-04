# 数据盘清理记录（2026-09-25）

## 清理前

`/root/gpufree-data`：147 GB，总使用138 GB，可用9.2 GB，占用94%。

## 已删除

1. `/root/gpufree-data/checkpoints/pi05_libero_lora`
   - 约19.1 GB；
   - 早期 `l40s_100step_smoke_v1` 的 step 99/499 checkpoint；
   - 当前监督器主线不再使用；不可直接恢复，只能重新训练。
2. `/root/gpufree-data/cache/huggingface/datasets`
   - 约35.3 GB；
   - 早期 LoRA 使用的 LIBERO parquet/Arrow 处理缓存；
   - 可由源数据重新生成。
3. `/root/gpufree-data/cache/huggingface/lerobot`
   - 约34.9 GB；
   - 同一 LIBERO 训练集的 LeRobot 下载缓存；
   - 可重新下载。
4. `/root/gpufree-data/tmp/libero-wheels`
   - 约1.66 GB；
   - 已安装的 torch 1.11/cu113、torchvision 0.12 等安装包；
   - 删除前已确认 LIBERO venv 中相应包可导入，可重新下载。

每个删除目标均通过 `readlink -f` 验证为预期绝对路径，且无训练进程占用。

## 明确保留

- `/root/gpufree-data/openpi-data/openpi-assets/checkpoints/pi05_libero` 正式策略checkpoint；
- `/root/gpufree-data/libero-traces` 当前监督器、视觉和配对轨迹；
- `/root/gpufree-data/supervisor-tools-v1` 代码、冻结产物和文档；
- DINO、Florence-2、SAM2视觉模型缓存；
- DROID跨场景研究数据及环境；
- openpi和LIBERO运行环境。

## 清理后

数据盘占用约36%，可用约94 GB。

