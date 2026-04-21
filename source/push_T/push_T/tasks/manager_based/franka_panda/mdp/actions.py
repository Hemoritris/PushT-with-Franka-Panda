# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""自定义Action拦截器 - 固定末端执行器姿态并限制位置范围."""

import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class FixedDownIKAction(DifferentialInverseKinematicsAction):
    """自定义动作拦截器：RL输出3D位置，限制位置范围，固定姿态朝下.

    - RL网络输出3维(x,y,z) - 目标位置
    - 限制位置范围在安全区域内
    - 姿态固定为朝下
    """

    # 类级别的固定朝下姿态 (w, x, y, z)
    EE_ORIENTATION_DOWN = torch.tensor([0.0, 1.0, 0.0, 0.0])

# 位置范围限制（世界坐标系，基座在z=0）
    X_RANGE = (0.15, 0.7)
    Y_RANGE = (-0.4, 0.4)
    Z_RANGE = (0.13, 0.5)

    def __init__(self, cfg, env):
        # 调用父类初始化
        super().__init__(cfg, env)
        # 父类会用6D创建_raw_actions等，但我们需要3D
        # 重新创建3D的动作张量
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 3, device=self.device)
        # 预计算固定朝下姿态（每个环境一个副本）
        self._ee_ori_fixed = self.EE_ORIENTATION_DOWN.clone().unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        # 用于 EMA 平滑的缓存
        self._prev_applied_pos = None
        self.smoothing_alpha = 1.0  # 无平滑，直接应用目标位置

    @property
    def action_dim(self) -> int:
        # 向 RL 算法声明：本环境只需要 3 维动作
        return 3

    def process_actions(self, actions: torch.Tensor):
        # 限制动作范围在 [-1, 1]
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        self._raw_actions[:] = actions

        # 将 [-1, 1] 线性映射到目标物理范围
        # 动作顺序: action[0]=前后(x), action[1]=左右(y), action[2]=上下(z)
        x_min, x_max = self.X_RANGE
        y_min, y_max = self.Y_RANGE
        z_min, z_max = self.Z_RANGE

        mapped_x = (actions[:, 0] + 1.0) / 2.0 * (x_max - x_min) + x_min
        mapped_y = (actions[:, 1] + 1.0) / 2.0 * (y_max - y_min) + y_min
        mapped_z = (actions[:, 2] + 1.0) / 2.0 * (z_max - z_min) + z_min

        # Debug: 打印动作到物理位置的映射
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 500 == 1:
            action_np = actions[0].cpu().numpy()
            pos_np = torch.stack([mapped_x[0], mapped_y[0], mapped_z[0]]).cpu().numpy()
            print(f"[DEBUG Actions] action={action_np} -> physical_pos(x,y,z)={pos_np}")

        # 当前目标位置
        current_target_pos = torch.stack([mapped_x, mapped_y, mapped_z], dim=-1)

        # EMA 平滑：减少高频抖动
        if self._prev_applied_pos is None:
            self._prev_applied_pos = current_target_pos.clone()

        # smoothed_pos = alpha * target + (1 - alpha) * last_pos
        smoothed_pos = self.smoothing_alpha * current_target_pos + (1 - self.smoothing_alpha) * self._prev_applied_pos
        self._processed_actions[:] = smoothed_pos
        self._prev_applied_pos[:] = smoothed_pos

        # 拼接固定姿态: [x, y, z, qw, qx, qy, qz]
        full_actions = torch.cat([self._processed_actions, self._ee_ori_fixed], dim=-1)

        # 设置到 IK 控制器
        self._ik_controller.set_command(full_actions)
