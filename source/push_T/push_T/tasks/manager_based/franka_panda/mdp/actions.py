# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""自定义Action拦截器 - 2D 平面控制 + 固定末端姿态/高度."""

import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class FixedDownIKAction(DifferentialInverseKinematicsAction):
    """自定义动作拦截器：RL 只输出平面 2D 目标，z 固定.

    - DP 网络输出 2 维动作 (x, y)，直接使用世界坐标系物理位置（米）
    - z 始终固定为常量，姿态固定朝下
    """

    # 类级别的固定朝下姿态 (w, x, y, z)
    EE_ORIENTATION_DOWN = torch.tensor([0.0, 1.0, 0.0, 0.0])

    # 位置范围限制（世界坐标系，基座在 z=0）
    X_RANGE = (0.15, 0.7)
    Y_RANGE = (-0.4, 0.4)
    Z_FIXED = 0.13

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 父类默认是 6D，改为我们需要的 2D 输入和 3D 目标位置。
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_ori_fixed = self.EE_ORIENTATION_DOWN.clone().unsqueeze(0).repeat(self.num_envs, 1).to(self.device)

    @property
    def action_dim(self) -> int:
        return 2

    def process_actions(self, actions: torch.Tensor):
        # 输入动作为物理坐标（米）：[x, y]
        self._raw_actions[:] = actions
        x_min, x_max = self.X_RANGE
        y_min, y_max = self.Y_RANGE

        # 仅做安全硬裁剪，避免 IK 接收超工作空间指令。
        target_x = actions[:, 0].clamp(x_min, x_max)
        target_y = actions[:, 1].clamp(y_min, y_max)
        target_z = torch.full_like(target_x, fill_value=self.Z_FIXED)

        # Debug: 打印动作到物理位置
        # if not hasattr(self, '_debug_counter'):
        #     self._debug_counter = 0
        # self._debug_counter += 1
        # if self._debug_counter % 500 == 1:
        #     raw_action_np = actions[0].cpu().numpy()
        #     pos_np = torch.stack([target_x[0], target_y[0], target_z[0]]).cpu().numpy()
        #     print(
        #         f"[DEBUG Actions] raw_physical_xy={raw_action_np} "
        #         f"-> physical_pos(x,y,z)={pos_np}"
        #     )

        self._processed_actions[:] = torch.stack([target_x, target_y, target_z], dim=-1)

        # 拼接固定姿态: [x, y, z, qw, qx, qy, qz]
        full_actions = torch.cat([self._processed_actions, self._ee_ori_fixed], dim=-1)

        # 设置到 IK 控制器
        self._ik_controller.set_command(full_actions)
