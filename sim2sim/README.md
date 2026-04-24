# sim2sim（MuJoCo 跨环境验证）

本目录提供一套基于 MuJoCo 的 `sim2sim` 验证流程，用于将 Isaac Lab 中训练得到的 Franka Panda Push-T `Diffusion Policy` 迁移到 MuJoCo 中进行跨环境测试。

当前实现目标是：

- 保持与 Isaac Lab 一致的低维任务接口；
- 保持动作语义为 `panda_hand` 的平面物理坐标 `[ee_x, ee_y]`；
- 固定 `panda_hand.z = 0.13m`，夹爪始终闭合；
- 使用 MuJoCo 场景对 `DDPM` / `DDIM` checkpoint 进行回放验证。

## 目录说明

- `model/push_t_scene.xml`：Push-T MuJoCo 场景
- `model/panda.xml`：标准 Panda MJCF 模型
- `model/assets/Tcube.stl`：T 块视觉 mesh
- `model/assets/target.stl`：目标区域视觉 mesh
- `mujoco_push_t_env.py`：MuJoCo 低维环境封装
- `validate_policy.py`：加载 checkpoint 并在 MuJoCo 中 rollout

## 任务接口对齐

### 观测

MuJoCo 环境输出的策略观测为 8 维：

- `t_block_xy`：T 块平面位置 `(2,)`
- `t_block_quat`：T 块世界系四元数 `(4,)`
- `ee_xy`：`panda_hand` 平面位置 `(2,)`

### 动作

策略动作为 2 维：

- `action = [ee_x, ee_y]`

环境内部会自动：

- 将动作裁剪到工作空间范围
- 固定 `panda_hand.z = 0.13`
- 固定手部姿态
- 保持夹爪闭合

### 成功判定

成功条件与 Isaac Lab 对齐：

- 位置误差 `< 0.02m`
- 旋转误差 `< 0.07rad`

## 资产与碰撞

- `Tcube.stl` 与 `target.stl` 用于视觉显示
- 为了先保证验证链路稳定，T 块碰撞目前使用两个 box 近似
- 目标区域仅用于可视化与目标姿态参考，不参与碰撞

这意味着当前版本更适合作为 **策略迁移验证**，而不是高保真接触动力学对比。

## 运行方式

所有命令都应当在 **仓库根目录** 下执行，统一使用相对路径，不依赖本机绝对路径。

### 1. DDPM 评估

```bash
python sim2sim/validate_policy.py \
  --checkpoint logs/diffusion_policy/pushT_ddpm/best_model.pt \
  --model_xml sim2sim/model/push_t_scene.xml \
  --episodes 5 \
  --steps_per_episode 1200 \
  --execute_horizon 8 \
  --deterministic_sampling \
  --clamp_action
```

### 2. DDIM 评估

```bash
python sim2sim/validate_policy.py \
  --checkpoint logs/diffusion_policy/pushT_ddim/best_model.pt \
  --model_xml sim2sim/model/push_t_scene.xml \
  --episodes 5 \
  --steps_per_episode 1200 \
  --execute_horizon 8 \
  --deterministic_sampling \
  --clamp_action
```

### 3. 实时演示

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

### 4. 关闭 viewer

如果只想统计结果，不打开可视化窗口，可添加：

```bash
--headless
```

## 依赖

当前 MuJoCo 验证脚本依赖：

- `mujoco`
- `torch`
- `numpy`
- `diffusers`
- `einops`

如果缺少依赖，可在当前 Python 环境中安装。

## 便携性说明

- 本目录所有脚本均按仓库相对路径组织
- 默认命令不使用绝对路径
- 若外部传入的是仓库内绝对路径，环境封装会尽量转换为相对路径，以提升跨机器可移植性

## 当前局限

- 目前主要用于 `sim2sim` 验证，不追求完全一致的接触动力学
- T 块碰撞近似为 box，而非完整凹形 mesh
- 若需要更高保真，可在后续引入 convex decomposition 或更精细的碰撞几何
