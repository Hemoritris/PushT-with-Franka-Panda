# Push-T (Isaac Lab) Project Guide

这是一个基于 Isaac Lab Manager Based框架的 Franka Panda Push-T 项目，当前项目包括:

- 键盘遥操作采集演示数据
- 基于低维观测训练 Diffusion Policy (DDPM/DDIM)
- 在仿真中回放和评估策略

项目已经将动作语义统一为物理坐标，便于和机械臂工作空间直接对应。


## 1. Project Scope

当前仓库关注的是 `Diffusion Policy` 工作流:

- 环境任务: `Template-Franka-Panda-Push-T-EePos-v0`
- 观测: `t_block_xy + t_block_quat + ee_xy` (8维)
- 动作: `ee_xy` 物理坐标 (2维, 单位米)
- 末端固定: `z=0.13`, 姿态朝下

说明:

- 训练/推理脚本位于 `scripts/diffusion/`
- 遥操作采集脚本位于 `scripts/teleop/`
- 任务配置和 MDP 逻辑位于 `source/push_T/push_T/tasks/manager_based/franka_panda/`


## 2. Directory Layout

```text
push_T/
├── assets/                                  # T block / target 资产
├── demonstrations/                          # 采集的数据 (如 push_t_demos.npz)
├── logs/                                    # 训练输出
├── sim2sim/                                 # MuJoCo 跨环境验证
│   ├── model/
│   │   ├── push_t_scene.xml                 # Push-T MuJoCo 场景
│   │   └── assets/                          # Panda / T block / target 的 MuJoCo 资产
│   ├── mujoco_push_t_env.py                 # MuJoCo 低维环境封装
│   ├── validate_policy.py                   # DDPM / DDIM 跨环境验证入口
│   └── README.md                            # sim2sim 使用说明
├── scripts/
│   ├── teleop/keyboard_teleop.py            # 键盘遥操作采集
│   ├── diffusion/
│   │   ├── dataset.py                       # 数据集与窗口采样
│   │   ├── model.py                         # 1D conditional UNet
│   │   ├── policy.py                        # Diffusion policy (DDPM/DDIM)
│   │   ├── train_dp.py                      # 训练入口
│   │   └── play_dp.py                       # 演示入口
│   ├── list_envs.py                         # 打印已注册任务
│   ├── random_agent.py                      # 随机动作调试
│   └── zero_agent.py                        # 零动作调试
└── source/push_T/push_T/tasks/manager_based/franka_panda/
    ├── franka_panda_env_cfg.py              # 场景/obs/action/reward/终止
    └── mdp/
        ├── actions.py                       # 2D action -> IK (固定z/姿态)
        ├── resets.py                        # reset时 ee x/y随机, z固定
        └── rewards.py                       # 奖励与成功判定
```


## 3. Setup

### 3.1 Verified Hardware / Software Versions

本项目在当前机器上的已验证环境如下:

- GPU: `NVIDIA GeForce RTX 4090` 
- NVIDIA Driver: `580.126.09`
- CUDA Toolkit (`nvcc`): `12.8`
- PyTorch: `2.7.0+cu128`
- Python: `3.11.13`
- Isaac Lab: `0.45.9`

说明:

- 若在容器内 `nvidia-smi` 不可用，但 `/proc/driver/nvidia/gpus/*/information` 可读，仍可确认 GPU 型号。
- `PyTorch + cu128` 需和驱动/CUDA 运行时兼容，建议保持同一主版本链路。

### 3.2 Dependency Checklist

推荐在 Isaac Lab 环境中运行 (例如 conda env `isaaclab`)。

核心依赖:

- `torch` / `torchvision` (CUDA 版本需匹配)
- `numpy`
- `gymnasium`
- `pygame` (键盘遥操作 UI)
- `diffusers` (DDPM/DDIM scheduler + 训练工具)
- `einops` (UNet 张量重排)
- `isaaclab` / `isaaclab_tasks` / `isaacsim` (由 Isaac Lab 环境提供)

当前环境中已安装版本:

- `numpy==1.26.0`
- `gymnasium==1.2.0`
- `pygame==2.6.1`
- `diffusers==0.31.0`
- `einops==0.8.1`

如需补齐 Python 包，可执行:

```bash
pip install numpy==1.26.0 gymnasium==1.2.0 pygame==2.6.1 diffusers==0.31.0 einops==0.8.1
```

### 3.3 Install task package

在仓库根目录执行:

```bash
pip install -e source/push_T
```

安装后可确保 `push_T.tasks` 在所有脚本中可被导入。


## 4. Task Definition (Current)

关键配置文件:

- `source/push_T/push_T/tasks/manager_based/franka_panda/franka_panda_env_cfg.py`
- `source/push_T/push_T/tasks/manager_based/franka_panda/mdp/actions.py`
- `source/push_T/push_T/tasks/manager_based/franka_panda/mdp/resets.py`
- `source/push_T/push_T/tasks/manager_based/franka_panda/mdp/rewards.py`

### 4.1 Observation

策略观测 8 维:

- `t_block_xy` (2)
- `t_block_quat` (4)
- `ee_xy` (2)

### 4.2 Action

策略动作 2 维:

- `action = [ee_x, ee_y]` (物理坐标, 米)

动作进入环境后:

- 在 `actions.py` 中进行工作空间安全裁剪
- 与固定 `z=0.13` 以及固定末端姿态拼接
- 通过 IK 控制器执行

工作空间范围:

- `x in [0.15, 0.70]`
- `y in [-0.40, 0.40]`

### 4.3 Reset and Success

- reset 时随机 `x/y`, 固定 `z=0.13`
- 成功阈值:
  - 位置阈值 `0.02`
  - 旋转阈值 `0.07`


## 5. Teleoperation Data Collection

脚本:

```bash
python scripts/teleop/keyboard_teleop.py \
  --task Template-Franka-Panda-Push-T-EePos-v0 \
  --save_path ./demonstrations
```

控制:

- `W/S`: X 正/负方向
- `A/D`: Y 正/负方向
- `Shift + 方向键`: 小步进
- `U`: 手动保存当前累计轨迹
- `ESC`: 退出并保存

默认步长:

- `POSE_STEP = 0.005` 米 (见 `keyboard_teleop.py`)


## 6. Demonstration Data

默认使用文件:

- `demonstrations/push_t_demos.npz`

训练脚本期望 `npz` 包含 3 个数组:

- `states`: shape `[N, 8]`, `float32`
- `actions`: shape `[N, 2]`, `float32`
- `traj_lengths`: shape `[num_traj]`, `int64`

字段语义:

- `states[:, 0:2]` -> `t_block_xy` (米)
- `states[:, 2:6]` -> `t_block_quat` (`qx, qy, qz, qw`)
- `states[:, 6:8]` -> `ee_xy` (米)
- `actions[:, 0:2]` -> 目标 `ee_xy` (米)

轨迹切分规则:

- 所有轨迹在时间维拼接后组成 `states/actions` 的第 0 维长度 `N`
- `traj_lengths[i]` 表示第 `i` 条轨迹的步数
- 必须满足: `sum(traj_lengths) == N`

### 6.1 Current Dataset Statistics

仓库演示数据 `demonstrations/push_t_demos.npz`详情:

- 轨迹条数: `204`
- 总步数: `68154`

轨迹长度分布:

- 最短: `38`
- 最长: `893`
- 平均: `334.09`
- 中位数: `316`

数值范围:

- `actions.x`: `[0.1500, 0.7000]`
- `actions.y`: `[-0.3743, 0.4000]`
- `states.ee_x`: `[0.1499, 0.6570]`
- `states.ee_y`: `[-0.3117, 0.3641]`
- `states.t_x`: `[0.3385, 0.5869]`
- `states.t_y`: `[-0.1463, 0.1409]`


### 6.2 Notes

- 目录模式下优先读取 `demonstrations/push_t_demos.npz`。
- 当前示例数据可能存在轨迹质量不算很高，多余动作较多的问题，因为键盘控制机械臂运动是一个离散控制过程，并且结合机械臂特性很难精准控制，所以使用此数据训练出来的模型也可能会学到此特征，即多余动作较多，属于正常现象。


## 7. Train Diffusion Policy

### 7.1 DDPM Training

```bash
python scripts/diffusion/train_dp.py \
  --demo_path ./demonstrations \
  --output_path ./logs/diffusion_policy/pushT_ddpm \
  --scheduler_type ddpm \
  --diffusion_steps 100 \
  --num_inference_steps 100 \
  --epochs 200 \
  --batch_size 256 \
  --device cuda:0
```

### 7.2 DDIM Training (faster inference oriented)

```bash
python scripts/diffusion/train_dp.py \
  --demo_path ./demonstrations \
  --output_path ./logs/diffusion_policy/pushT_ddim \
  --scheduler_type ddim \
  --diffusion_steps 100 \
  --num_inference_steps 16 \
  --epochs 200 \
  --batch_size 256 \
  --device cuda:0
```

训练输出:

- `best_model.pt`: 最佳验证模型
- `model.pt`: 最佳模型别名 (用于演示最方便)
- 常用目录命名:
  - `logs/diffusion_policy/pushT_ddpm/`
  - `logs/diffusion_policy/pushT_ddim/`


## 8. Play / Evaluate

### 8.1 High-quality (DDPM)

```bash
python scripts/diffusion/play_dp.py \
  --checkpoint ./logs/diffusion_policy/pushT_ddpm/model.pt \
  --task Template-Franka-Panda-Push-T-EePos-v0 \
  --num_envs 1 \
  --steps 2000 \
  --num_inference_steps 100 \
  --execute_horizon 8 \
  --deterministic_sampling \
  --sync_planning
```

### 8.2 Faster (DDIM)

```bash
python scripts/diffusion/play_dp.py \
  --checkpoint ./logs/diffusion_policy/pushT_ddim/model.pt \
  --task Template-Franka-Panda-Push-T-EePos-v0 \
  --num_envs 1 \
  --steps 2000 \
  --num_inference_steps 16 \
  --execute_horizon 8 \
  --deterministic_sampling \
  --sync_planning
```

说明:

- `play_dp.py` 会从 checkpoint 自动识别并加载 DDPM 或 DDIM scheduler


## 9. Important Runtime Arguments

`scripts/diffusion/play_dp.py` 常用参数:

- `--num_inference_steps`: 推理扩散步数
- `--execute_horizon`: 每次规划后连续执行多少步动作
- `--deterministic_sampling`: 确定性采样
- `--sync_planning`: 强制同步规划 (质量优先)
- `--clamp_action`: 在 `env.step` 前再次按工作空间裁剪
- `--action_ema_alpha`: 执行动作 EMA 平滑系数 (1.0 表示关闭)
- `--max_action_delta`: 每步动作增量限制 (<=0 关闭)


## 10. Debug Utilities

列出任务:

```bash
python scripts/list_envs.py
```

零动作:

```bash
python scripts/zero_agent.py --task Template-Franka-Panda-Push-T-EePos-v0 --num_envs 1
```

随机动作:

```bash
python scripts/random_agent.py --task Template-Franka-Panda-Push-T-EePos-v0 --num_envs 1
```


## 11. Common Issues


1. 演示卡顿明显

- 先检查是否回退到 CPU (`CUDA unavailable`)
- 质量优先: `DDPM + 100 steps`
- 实时优先: 重新训练 `DDIM`，不要直接把 DDPM 推理步数强行降太低
- 因为整个演示过程采用推理+控制模式，如果使用DDPM模型可能会出现计算跟与控制交替进行，导致动作信号断断续续的情况，表现为动作卡顿，缓冲策略优化有限；DDIM模型很好的缓解了卡顿情况，可以用于实时控制。

2. 训练报动作范围错误

- `dataset.py` 检测到动作不在物理坐标范围
- 需要使用当前 teleop 脚本重新采集演示数据


## 12. MuJoCo sim2sim Validation

仓库现已补充 `sim2sim/` 目录，用于将 Isaac Lab 中训练得到的策略迁移到 MuJoCo 中做跨环境验证。

当前 MuJoCo 验证链路包括：

- `sim2sim/model/push_t_scene.xml`：Push-T 场景
- `sim2sim/mujoco_push_t_env.py`：MuJoCo 低维环境封装
- `sim2sim/validate_policy.py`：加载 `DDPM / DDIM` checkpoint 并 rollout

对齐原则：

- 观测仍为 `t_block_xy + t_block_quat + ee_xy`（8维）
- 动作仍为 `panda_hand` 的物理平面坐标 `[ee_x, ee_y]`（2维）
- `panda_hand` 高度固定为 `0.13m`
- 夹爪始终闭合

示例：

```bash
python sim2sim/validate_policy.py \
  --checkpoint logs/diffusion_policy/pushT_ddim/best_model.pt \
  --model_xml sim2sim/model/push_t_scene.xml \
  --episodes 1 \
  --steps_per_episode 300 \
  --execute_horizon 8 \
  --deterministic_sampling \
  --clamp_action \
  --realtime
```

更多说明请见：`sim2sim/README.md`


## 13. Notes

- 当前主线逻辑演示数据 + Diffusion Policy。
- 若后续要接入真实机械臂，请优先保持:
  - 动作语义一致 (物理坐标)
  - 观测语义一致 (训练/推理完全同构)
  - scheduler 与推理参数与训练配置匹配


## 14. References

1. C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," arXiv preprint arXiv:2303.04137 [cs.RO], 2023, doi: 10.48550/arXiv.2303.04137.

2. Reference repository:

- https://github.com/real-stanford/diffusion_policy
- https://github.com/google-deepmind/mujoco
- https://github.com/google-deepmind/mujoco_menagerie
3. Learning Resources:

- Diffusion Policy project page:
  https://diffusion-policy.cs.columbia.edu/
- arXiv paper page:
  https://arxiv.org/abs/2303.04137
- 【最适合入门的diffusion policy】https://www.bilibili.com/video/BV1MtXHYUE6M?vd_source=4d2ca99b593d2f2b9c234c77d695c78c
- 【入门机器人Diffusion Policy】https://www.bilibili.com/video/BV1eGbceREsx?vd_source=4d2ca99b593d2f2b9c234c77d695c78c
- 【扩散模型 - Diffusion Model【李宏毅2023】】https://www.bilibili.com/video/BV14c411J7f2?vd_source=4d2ca99b593d2f2b9c234c77d695c78c

