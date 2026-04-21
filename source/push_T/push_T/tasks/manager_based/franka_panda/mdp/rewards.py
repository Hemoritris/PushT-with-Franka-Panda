# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda Push-T 任务的自定义奖励函数 - 末端位置控制版本."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def end_effector_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取末端执行器位置."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_state_w[:, asset_cfg.body_ids, :3].squeeze(1)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """末端执行器到物体的距离."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    ee_pos = robot.data.body_state_w[:, ee_cfg.body_ids, :3].squeeze(1)
    obj_pos = obj.data.root_state_w[:, :3]

    return torch.norm(ee_pos - obj_pos, dim=1)


def gaussian_reward(distance: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """高斯核奖励：靠近时奖励快速增加，远距离时平滑衰减。

    Reward = exp(-distance^2 / (2 * sigma^2))
    """
    return torch.exp(-distance ** 2 / (2 * sigma ** 2))


def exp_decay_reward(distance: torch.Tensor, k: float = 5.0) -> torch.Tensor:
    """指数衰减奖励：靠近时奖励快速增加。

    Reward = exp(-k * distance)
    """
    return torch.exp(-k * distance)


def object_target_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """物体到目标的距离 (位置)."""
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    obj_pos = obj.data.root_state_w[:, :3]
    target_pos = target.data.root_state_w[:, :3]

    return torch.norm(obj_pos - target_pos, dim=1)


def object_target_rotation_diff(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """物体到目标的旋转角度差异 (以弧度为单位)."""
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    obj_quat = obj.data.root_state_w[:, 3:7]
    target_quat = target.data.root_state_w[:, 3:7]

    relative_quat = quat_mul(obj_quat, quat_conjugate(target_quat))

    angle = 2.0 * torch.atan2(
        torch.norm(relative_quat[:, 1:], dim=1),
        torch.abs(relative_quat[:, 0])
    )

    return angle


def is_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.2,
) -> torch.Tensor:
    """判断物体是否成功到达目标位置并对齐."""
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    obj_pos = obj.data.root_state_w[:, :3]
    target_pos = target.data.root_state_w[:, :3]
    pos_distance = torch.norm(obj_pos - target_pos, dim=1)

    obj_quat = obj.data.root_state_w[:, 3:7]
    target_quat = target.data.root_state_w[:, 3:7]
    relative_quat = quat_mul(obj_quat, quat_conjugate(target_quat))
    rot_distance = 2.0 * torch.atan2(
        torch.norm(relative_quat[:, 1:], dim=1),
        torch.abs(relative_quat[:, 0])
    )

    pos_success = pos_distance < position_threshold
    rot_success = rot_distance < rotation_threshold

    return pos_success & rot_success


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float,
) -> torch.Tensor:
    """判断物体高度是否低于最小阈值."""
    obj: RigidObject = env.scene[asset_cfg.name]
    obj_pos = obj.data.root_state_w[:, 2]
    return (obj_pos < minimum_height)