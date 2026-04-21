# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""播放BC预训练模型脚本 - 支持原始BC模型和RSL-RL转换模型."""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play BC pretrained model.")
parser.add_argument("--task", type=str, default="Template-Franka-Panda-Push-T-EePos-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--model", type=str, required=True, help="Model path (.pt file) - BC or RSL-RL format.")
parser.add_argument("--rsl_rl", action="store_true", default=False, help="Use RSL-RL format (from convert_bc_to_rsl_rl.py).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

import push_T.tasks

from rsl_rl.modules.actor_critic import ActorCritic


class BCPolicy(nn.Module):
    """BC策略网络 - 与训练时一致."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(prev_dim, action_dim)

    def forward(self, obs):
        features = self.feature_net(obs)
        actions = torch.tanh(self.action_net(features))
        return actions


class RSLRLActorCriticWrapper(nn.Module):
    """包装RSL-RL ActorCritic，适配play接口."""

    def __init__(self, ac, obs_normalizer):
        super().__init__()
        self.ac = ac
        self.obs_normalizer = obs_normalizer

    def forward(self, obs):
        # RSL-RL的actor直接处理归一化后的观测
        return self.ac.actor(obs)


def load_model(model_path, use_rsl_rl=False):
    """加载模型，自动检测格式."""
    print(f"[INFO] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    if use_rsl_rl or "model_state_dict" in checkpoint:
        # RSL-RL 格式
        print("  格式: RSL-RL (from convert_bc_to_rsl_rl.py)")
        obs_dim = checkpoint['model_state_dict']['actor.0.weight'].shape[1]
        action_dim = checkpoint['model_state_dict']['actor.6.weight'].shape[0]

        # 构建 ActorCritic
        hidden_dims = [256, 256, 128]
        ac = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=hidden_dims,
            critic_hidden_dims=hidden_dims,
            activation="elu",
            init_noise_std=0.1,
            noise_std_type="scalar",
        ).to("cpu")

        # 加载权重
        ac.load_state_dict(checkpoint['model_state_dict'], strict=False)
        ac.eval()

        # 获取 obs_normalizer
        obs_norm_state_dict = checkpoint.get('obs_norm_state_dict', None)

        print(f"  obs_dim: {obs_dim}, action_dim: {action_dim}")
        return ac, obs_norm_state_dict, True

    else:
        # 原始 BC 格式
        print("  格式: BC (from train_bc.py)")

        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        hidden_dims = checkpoint.get('hidden_dims', [256, 256, 128])

        # 检查是否使用归一化
        state_mean = checkpoint.get('state_mean', None)
        state_std = checkpoint.get('state_std', None)
        action_mean = checkpoint.get('action_mean', None)
        action_std = checkpoint.get('action_std', None)

        use_normalization = state_mean is not None and state_std is not None and np.any(state_std != 1.0)
        print(f"  obs_dim: {obs_dim}, action_dim: {action_dim}")
        print(f"  使用归一化: {use_normalization}")

        # 构建 BC 策略网络
        bc_policy = BCPolicy(obs_dim, action_dim, hidden_dims)
        bc_policy.load_state_dict(checkpoint['policy_state_dict'])
        bc_policy.eval()

        return bc_policy, (state_mean, state_std, action_mean, action_std), use_normalization


def main():
    model, norm_info, is_rsl_rl = load_model(args_cli.model, args_cli.rsl_rl)

    # 创建环境
    env_cfg_class = None
    import push_T.tasks
    for item in dir(push_T.tasks):
        obj = getattr(push_T.tasks, item)
        if isinstance(obj, type) and "Franka" in item and "EePos" in item and "EnvCfg" in item:
            env_cfg_class = obj
            break

    if env_cfg_class is None:
        from push_T.tasks.manager_based.franka_panda.franka_panda_env_cfg import FrankaPandaEePosTeleopEnvCfg
        env_cfg_class = FrankaPandaEePosTeleopEnvCfg

    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")

    dt = env.unwrapped.step_dt

    print("[INFO] Starting policy playback...")

    obs_dict, _ = env.reset()
    step_count = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.no_grad():
                # 从dict中提取policy观测
                if isinstance(obs_dict, dict):
                    policy_obs = obs_dict["policy"]
                else:
                    policy_obs = obs_dict

                obs_np = policy_obs[0].cpu().numpy() if args_cli.num_envs == 1 else policy_obs.cpu().numpy()

                if is_rsl_rl:
                    # RSL-RL 格式：使用内置的obs_normalizer
                    # obs_normalizer期望 (N, obs_dim) 的tensor
                    obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0)
                    action = model.actor(obs_tensor).numpy().squeeze(0)
                    # BC模型有tanh，RSL-RL的actor没有tanh，需要手动加
                    action = np.tanh(action)
                else:
                    # BC 格式
                    state_mean, state_std, action_mean, action_std = norm_info

                    if norm_info and np.any(state_std != 1.0):
                        # 使用归一化
                        obs_normalized = (obs_np - state_mean) / state_std
                        obs_tensor = torch.from_numpy(obs_normalized).float()
                    else:
                        # 不使用归一化
                        obs_tensor = torch.from_numpy(obs_np).float()

                    action = model(obs_tensor).numpy()

                    # 反归一化动作
                    if norm_info and np.any(state_std != 1.0):
                        action = action * action_std + action_mean

                # 限制动作范围 [-1, 1]
                action = np.clip(action, -1.0, 1.0)

                # 转换为tensor
                action_tensor = torch.from_numpy(action).float().unsqueeze(0)

            # step环境
            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)
            step_count += 1

            # 处理episode结束
            if terminated or truncated:
                obs_dict, _ = env.reset()
                step_count = 0
                print("[INFO] Episode reset")

            if step_count % 100 == 0:
                print(f"[INFO] Step: {step_count}, Action: {action[:3]}")

            # 实时模式
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted at step {step_count}")
    finally:
        env.close()
        print(f"[INFO] Done. Total steps: {step_count}")


if __name__ == "__main__":
    main()
    simulation_app.close()
