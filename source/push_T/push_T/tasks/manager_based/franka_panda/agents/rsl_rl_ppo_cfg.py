# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda Push-T 任务的 RL 配置."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO 运行器配置.

    用于BC预训练模型的RL微调。
    """
    num_steps_per_env = 48
    max_iterations = 5000
    save_interval = 100
    experiment_name = "franka_panda_push_t"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,  # BC预训练模型降低初始探索
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,  # BC微调用较低熵系数
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-5,  # BC微调用较小学习率
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )