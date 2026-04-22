# Diffusion Policy Scripts

本目录包含 Push-T 的 diffusion 训练与演示脚本:

- `train_dp.py`: 训练低维 Diffusion Policy (支持 DDPM / DDIM)
- `play_dp.py`: 加载 checkpoint 并在 Isaac Lab 环境中演示
- `dataset.py`: `npz` 演示数据窗口采样与归一化
- `policy.py`: 低维 diffusion policy 推理/训练逻辑
- `model.py`: 条件 UNet1D 主干
- `utils.py`: 归一化与辅助工具

完整项目说明、任务语义、命令示例和常见问题请查看仓库根目录:

- `README.md`
