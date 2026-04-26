# sim2sim（MuJoCo 跨环境验证）

本目录提供 MuJoCo 版 Push-T 验证流程，用于将 Isaac Lab 中训练的 Franka Panda `Diffusion Policy` 在 MuJoCo 中回放评估。

当前目标：

- 保持与 Isaac Lab 一致的低维接口；
- 动作语义保持 `ee_xy` 物理坐标；
- `panda_hand.z` 固定为 `0.13m`，夹爪保持闭合；
- 在 MuJoCo 中验证 `DDPM` / `DDIM` checkpoint 的可运行性与稳定性。

## 目录说明

- `model/push_t_scene.xml`：Push-T 场景（含 T 块碰撞参数）
- `model/panda.xml`：Panda 机器人 MJCF
- `model/assets/Tcube.stl`：T 块视觉 mesh
- `model/assets/Tcube_col_01.stl`：T 块 CoACD 碰撞凸包 1
- `model/assets/Tcube_col_02.stl`：T 块 CoACD 碰撞凸包 2
- `model/assets/target.stl`：目标区域视觉 mesh
- `mujoco_push_t_env.py`：MuJoCo 低维环境封装
- `validate_policy.py`：checkpoint rollout 验证入口

## 接口对齐

### 观测（8维）

- `t_block_xy` `(2,)`
- `t_block_quat` `(4,)`
- `ee_xy` `(2,)`

### 动作（2维）

- `action = [ee_x, ee_y]`

环境内部会自动：

- 裁剪动作到工作空间 `x∈[0.15,0.70], y∈[-0.40,0.40]`
- 固定 `panda_hand.z = 0.13`
- 固定手部姿态
- 夹爪闭合

### 成功判定

- 位置误差 `< 0.02m`
- 旋转误差 `< 0.07rad`



## 控制稳定化

`mujoco_push_t_env.py` 当前默认包含两层稳定化：

- 动作层：EMA + 单步增量限制
- 关节命令层：IK 解后再做 EMA + 单步增量限制

IK 默认参数：

- `ik_damping=0.25`
- `ik_step_size=0.40`
- `ik_control_iterations=16`
- `ik_orientation_weight=0.35`
- `ik_pos_weight_z=2.0`


## 运行方式

所有命令在仓库根目录执行。

### 1. DDPM 评估

```bash
python sim2sim/validate_policy.py \
  --checkpoint logs/diffusion_policy/pushT_ddpm/best_model.pt \
  --model_xml sim2sim/model/push_t_scene.xml \
  --episodes 5 \
  --steps_per_episode 1200 \
  --deterministic_sampling
```

### 2. DDIM 评估

```bash
python sim2sim/validate_policy.py \
  --checkpoint logs/diffusion_policy/pushT_ddim/best_model.pt \
  --model_xml sim2sim/model/push_t_scene.xml \
  --episodes 5 \
  --steps_per_episode 1200 \
  --deterministic_sampling
```


### 3. 无窗口统计

```bash
python sim2sim/validate_policy.py \
  --checkpoint logs/diffusion_policy/pushT_ddim/best_model.pt \
  --headless
```

## 依赖

- `mujoco`
- `torch`
- `numpy`
- `diffusers`
- `einops`

