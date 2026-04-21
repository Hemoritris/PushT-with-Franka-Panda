# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""将 BC 模型转换为 RSL-RL ActorCritic 格式，用于继续 RL 训练."""

import argparse
import os

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic

parser = argparse.ArgumentParser(description="Convert BC policy to RSL-RL format.")
parser.add_argument("--bc_model", type=str, required=True, help="BC 模型路径 (.pt 文件).")
parser.add_argument("--output", type=str, default=None, help="输出路径 (可选, 默认保存到 logs/rsl_rl/).")
parser.add_argument("--device", type=str, default="cuda:0", help="设备.")
parser.add_argument("--experiment_name", type=str, default="franka_panda_push_t", help="实验名称.")
parser.add_argument("--run_name", type=str, default="bc_pretrained", help="运行名称.")
args_cli = parser.parse_args()


def convert_bc_to_rsl_rl():
    """将 BC 模型权重转换为 RSL-RL ActorCritic 格式."""
    print(f"加载 BC 模型: {args_cli.bc_model}")
    bc_checkpoint = torch.load(args_cli.bc_model, weights_only=False)

    bc_state_dict = bc_checkpoint['policy_state_dict']
    obs_dim = bc_checkpoint['obs_dim']
    action_dim = bc_checkpoint['action_dim']
    hidden_dims = bc_checkpoint.get('hidden_dims', [256, 256, 128])

    # 归一化参数
    state_mean = bc_checkpoint.get('state_mean', None)
    state_std = bc_checkpoint.get('state_std', None)
    action_mean = bc_checkpoint.get('action_mean', None)
    action_std = bc_checkpoint.get('action_std', None)

    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  隐藏层: {hidden_dims}")
    if state_mean is not None:
        print(f"  归一化参数: 已保存")
    else:
        print(f"  归一化参数: 无")

    # 构建 RSL-RL ActorCritic
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    ac = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation="elu",
        init_noise_std=0.1,  # 使用较小的初始噪声
        noise_std_type="scalar",
    ).to(device)

    print(f"\nRSL-RL ActorCritic state_dict keys:")
    for key, v in ac.state_dict().items():
        print(f"  {key}: {v.shape}")

    rsl_rl_state_dict = {}

    # actor 层映射
    # RSL actor.0 (obs_dim→256) <- BC feature_net.0
    # RSL actor.2 (256→256) <- BC feature_net.2
    # RSL actor.4 (256→128) <- BC feature_net.4
    # RSL actor.6 (128→action_dim) <- BC action_net
    actor_mapping = {
        'actor.0': 'feature_net.0',
        'actor.2': 'feature_net.2',
        'actor.4': 'feature_net.4',
        'actor.6': 'action_net',
    }

    for rsl_key, bc_key_base in actor_mapping.items():
        weight_key = bc_key_base + '.weight'
        bias_key = bc_key_base + '.bias'
        if weight_key in bc_state_dict:
            # 直接复制权重（BC训练时没有归一化，所以不需要反归一化）
            rsl_rl_state_dict[rsl_key + '.weight'] = bc_state_dict[weight_key]
            rsl_rl_state_dict[rsl_key + '.bias'] = bc_state_dict[bias_key]
            print(f"  {rsl_key} <- {bc_key_base}: {bc_state_dict[weight_key].shape}")
        else:
            print(f"  [WARN] {weight_key} not found in BC state_dict")

    # critic: 用 BC 的前三层初始化 critic 隐藏层
    rsl_rl_state_dict['critic.0.weight'] = bc_state_dict['feature_net.0.weight'].clone()
    rsl_rl_state_dict['critic.0.bias'] = bc_state_dict['feature_net.0.bias'].clone()
    rsl_rl_state_dict['critic.2.weight'] = bc_state_dict['feature_net.2.weight'].clone()
    rsl_rl_state_dict['critic.2.bias'] = bc_state_dict['feature_net.2.bias'].clone()
    rsl_rl_state_dict['critic.4.weight'] = bc_state_dict['feature_net.4.weight'].clone()
    rsl_rl_state_dict['critic.4.bias'] = bc_state_dict['feature_net.4.bias'].clone()

    # critic 输出层用小随机初始化
    nn.init.orthogonal_(ac.critic[6].weight, gain=0.01)
    nn.init.constant_(ac.critic[6].bias, 0.0)

    # 加载权重
    ac.load_state_dict(rsl_rl_state_dict, strict=False)

    # 构建 obs_normalizer state_dict (RSL-RL EmpiricalNormalization 需要的格式)
    # BC训练时没有归一化，所以这里初始化为单位归一化（RSL-RL会从头学习）
    obs_mean = torch.zeros(obs_dim).to(device)
    obs_var = torch.ones(obs_dim).to(device)
    obs_std = torch.ones(obs_dim).to(device)
    sample_count = bc_checkpoint.get('sample_count', 1000)
    obs_norm_state_dict = {"_mean": obs_mean.unsqueeze(0), "_var": obs_var.unsqueeze(0), "_std": obs_std.unsqueeze(0), "count": torch.tensor([sample_count]).to(device)}

    # 验证
    print(f"\n验证加载后的 actor 输出:")
    test_input = torch.randn(1, obs_dim).to(device)
    with torch.no_grad():
        action = ac.actor(test_input)
        value = ac.critic(test_input)
    print(f"  测试输入: {test_input.shape}")
    print(f"  Actor 输出: {action.shape}, Critic 输出: {value.shape}")

    # 保存 RSL-RL checkpoint
    # 格式: logs/rsl_rl/{experiment_name}/{run_name}/model_{iter}.pt
    log_dir = os.path.join("logs", "rsl_rl", args_cli.experiment_name, args_cli.run_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, "model_0.pt")

    rsl_rl_checkpoint = {
        "model_state_dict": ac.state_dict(),
        "optimizer_state_dict": {},
        "iter": 0,
        "infos": {
            "bc_pretrained": True,
            "bc_model": args_cli.bc_model,
        },
        "obs_norm_state_dict": obs_norm_state_dict,
        "privileged_obs_norm_state_dict": obs_norm_state_dict,  # 与 obs_norm_state_dict 相同
    }

    # 保存归一化参数（如果有的话）
    if state_mean is not None:
        rsl_rl_checkpoint['norm_mean'] = state_mean
        rsl_rl_checkpoint['norm_std'] = state_std
        rsl_rl_checkpoint['action_mean'] = action_mean
        rsl_rl_checkpoint['action_std'] = action_std

    torch.save(rsl_rl_checkpoint, checkpoint_path)
    print(f"\nRSL-RL checkpoint 已保存: {checkpoint_path}")


if __name__ == "__main__":
    convert_bc_to_rsl_rl()