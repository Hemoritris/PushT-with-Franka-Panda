# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda Push-T 自定义重置逻辑."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_ee_xy_random_fixed_z(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.15, 0.7),
    y_range: tuple[float, float] = (-0.4, 0.4),
    z_fixed: float = 0.13,
    action_term_name: str = "arm_action",
    ik_iterations: int = 2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """将机器人重置到随机平面位置：x/y 随机，z 固定.

    逻辑：
    1) 先回到默认关节状态，避免上个 episode 残留状态影响；
    2) 采样目标 ee (x, y, z_fixed)；
    3) 通过当前 action term 的 IK 控制器求解并写回关节状态。
    """
    if env_ids is None or len(env_ids) == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device
    env_ids = env_ids.to(device=device, dtype=torch.long)

    # 回默认关节
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

    # 采样目标 ee 位置（仅平面随机）
    num = env_ids.numel()
    target_xy = torch.empty((num, 2), device=device)
    target_xy[:, 0].uniform_(x_range[0], x_range[1])
    target_xy[:, 1].uniform_(y_range[0], y_range[1])
    target_pos = torch.cat([target_xy, torch.full((num, 1), z_fixed, device=device)], dim=-1)

    # 获取动作项（FixedDownIKAction）
    action_term = env.action_manager._terms.get(action_term_name, None)
    if action_term is None:
        return

    # 为当前 env_ids 写入随机目标动作（物理坐标）
    action_term._raw_actions[env_ids] = target_xy
    action_term._processed_actions[env_ids] = target_pos

    # 为 IK 控制器构造全量 command，避免影响未重置的环境实例
    full_command = action_term._ik_controller._command.clone()
    full_command[env_ids, 0:3] = target_pos
    if hasattr(action_term, "_ee_ori_fixed"):
        full_command[env_ids, 3:7] = action_term._ee_ori_fixed[env_ids]
    action_term._ik_controller.set_command(full_command)

    # 迭代 IK
    ik_iterations = max(int(ik_iterations), 1)
    for i in range(ik_iterations):
        ee_pos_curr, ee_quat_curr = action_term._compute_frame_pose()
        jacobian = action_term._compute_frame_jacobian()
        curr_joint_pos = action_term._asset.data.joint_pos[:, action_term._joint_ids]
        joint_pos_des = action_term._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, curr_joint_pos)

        joint_pos_des_env = joint_pos_des[env_ids]
        joint_vel_env = torch.zeros_like(joint_pos_des_env)
        asset.write_joint_state_to_sim(
            joint_pos_des_env, joint_vel_env, joint_ids=action_term._joint_ids, env_ids=env_ids
        )
        asset.set_joint_position_target(joint_pos_des_env, joint_ids=action_term._joint_ids, env_ids=env_ids)
        asset.set_joint_velocity_target(joint_vel_env, joint_ids=action_term._joint_ids, env_ids=env_ids)

        # 非最后一次迭代时，推进一次前向更新雅可比/末端位姿
        if i < ik_iterations - 1:
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)
