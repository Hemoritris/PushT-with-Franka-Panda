# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda Push-T 的最小 MDP 函数集合.

- 观测相关函数（xy 提取）
- 成功判定函数（供 termination 与最小奖励共用）
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def root_pos_xy_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取刚体根坐标的平面位置 (x, y)."""
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.root_state_w[:, :2]


def end_effector_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取末端执行器平面位置 (x, y)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_state_w[:, asset_cfg.body_ids, :2].squeeze(1)


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
