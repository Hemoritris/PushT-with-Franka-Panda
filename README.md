# Franka Panda Push-T

IsaacLab 环境下基于强化学习的 Franka Panda 机械臂推动 T 型块任务。

## 任务描述

训练机械臂将桌面上的 T 形方块推动到目标位置并对齐角度。

## 项目结构

```
push_T/
├── assets/                    # USD 资源文件
│   ├── block_T.usd           # T 型块模型
│   └── target.usd            # 目标标记模型
├── source/push_T/push_T/tasks/manager_based/franka_panda/
│   ├── franka_panda_env_cfg.py   # 环境配置
│   ├── agents/rsl_rl_ppo_cfg.py  # PPO 算法配置
│   └── mdp/rewards.py            # 自定义奖励函数
├── scripts/rsl_rl/
  ├── train.py             # 训练脚本
  └── play.py              # 演示脚本

```

## 环境配置

### 依赖

- IsaacLab
- Python 3.10+
- PyTorch
- IsaacLab_rl
- rsl_rl_lib

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --num_envs | 4096 | 并行环境数 |
| --max_iterations | 5000 | 训练轮数 |
| --headless | False | 无头模式运行 |
| --video | False | 录制视频 |

## 任务配置

### 奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| reaching_object | -2.0 | 指尖到 T 型块距离 |
| object_to_target_pos | -20.0 | T 型块到目标距离 |
| object_to_target_rot | -1.0 | T 型块旋转对齐 |
| success_bonus | 10000.0 | 成功完成奖励 |
| action_rate_penalty | -0.01 | 动作平滑惩罚 |

### 成功判定

- 位置阈值: 0.12 m
- 旋转阈值: 0.50 rad
- Episode 长度: 20 秒

### PPO 配置

| 参数 | 值 |
|------|-----|
| learning_rate | 5.0e-4 |
| entropy_coef | 0.005 |
| gamma | 0.99 |
| lam | 0.95 |
| actor_hidden_dims | [256, 256, 128] |
| critic_hidden_dims | [256, 256, 128] |