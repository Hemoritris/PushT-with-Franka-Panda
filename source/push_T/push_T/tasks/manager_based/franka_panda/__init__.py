# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda Push-T 环境子模块 - 末端位置控制版本."""

import gymnasium as gym

##
# 注册 Gym 环境
##

gym.register(
    id="Template-Franka-Panda-Push-T-EePos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_panda_env_cfg:FrankaPandaEePosEnvCfg",
    },
)
