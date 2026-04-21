# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""行为克隆训练脚本 - 用专家演示数据预训练策略."""

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

parser = argparse.ArgumentParser(description="行为克隆训练.")
parser.add_argument("--demo_path", type=str, default="./demonstrations/", help="演示数据路径.")
parser.add_argument("--output_path", type=str, default="./bc_models/", help="模型保存路径.")
parser.add_argument("--epochs", type=int, default=100, help="训练轮数.")
parser.add_argument("--batch_size", type=int, default=256, help="批次大小.")
parser.add_argument("--lr", type=float, default=1e-4, help="学习率.")
parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数.")
parser.add_argument("--device", type=str, default="cuda:0", help="训练设备.")
parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例.")
parser.add_argument("--action_noise", type=float, default=0.01, help="动作噪声标准差 (0=不使用).")
parser.add_argument("--obs_noise", type=float, default=0.01, help="观测噪声标准差 (0=不使用).")
parser.add_argument("--random_sampling", action="store_true", default=False, help="随机采样而非顺序.")
args_cli = parser.parse_args()


class BCPolicy(nn.Module):
    """行为克隆策略网络 - 匹配RSL-RL actor架构."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, obs):
        features = self.feature_net(obs)
        actions = torch.tanh(self.action_net(features))
        return actions


def apply_augmentation(states, actions, obs_noise=0.0, action_noise=0.0):
    """对观测和动作添加噪声进行数据增强."""
    if obs_noise > 0:
        noise = torch.randn_like(states) * obs_noise
        states = states + noise
    if action_noise > 0:
        noise = torch.randn_like(actions) * action_noise
        actions = actions + noise
    return states, actions


def load_demonstrations(demo_path):
    """加载演示数据."""
    demo_files = [f for f in os.listdir(demo_path) if f.endswith('.npz')]
    if len(demo_files) == 0:
        raise FileNotFoundError(f"在 {demo_path} 中没有找到演示文件")

    latest_file = sorted(demo_files)[-1]
    demo_file = os.path.join(demo_path, latest_file)
    print(f"加载演示数据: {demo_file}")
    data = np.load(demo_file, allow_pickle=True)

    states = data['states']
    actions = data['actions']
    traj_lengths = data['traj_lengths']

    if states.dtype == object:
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)

    print(f"  轨迹数: {len(traj_lengths)}")
    print(f"  总步数: {len(states)}")
    print(f"  观测维度: {states.shape[1] if len(states.shape) > 1 else states.shape}")
    print(f"  动作维度: {actions.shape[1] if len(actions.shape) > 1 else actions.shape}")

    return states, actions, traj_lengths


def train_bc():
    """训练行为克隆策略."""
    os.makedirs(args_cli.output_path, exist_ok=True)

    states, actions, traj_lengths = load_demonstrations(args_cli.demo_path)
    obs_dim = states.shape[1]
    action_dim = actions.shape[1]
    print(f"\n观测维度: {obs_dim}, 动作维度: {action_dim}")

    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")

    # 归一化参数（为了一致性仍然保存，但训练时使用原始数据）
    # 这样下游脚本可以选择是否使用归一化
    state_mean = np.zeros(obs_dim)
    state_std = np.ones(obs_dim)

    action_mean = np.zeros(action_dim)
    action_std = np.ones(action_dim)

    print(f"  [不使用归一化] 观测和动作使用原始数据训练")
    if args_cli.obs_noise > 0 or args_cli.action_noise > 0:
        print(f"  [数据增强] 观测噪声 std={args_cli.obs_noise}, 动作噪声 std={args_cli.action_noise}")
    else:
        print(f"  [无数据增强]")

    # 直接使用原始数据（不归一化）
    states_tensor = torch.from_numpy(states).float()
    actions_tensor = torch.from_numpy(actions).float()

    # 划分训练集和验证集
    dataset = TensorDataset(states_tensor, actions_tensor)
    val_size = int(len(dataset) * args_cli.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"  训练集: {train_size} 步, 验证集: {val_size} 步")

    train_loader = DataLoader(train_dataset, batch_size=args_cli.batch_size, shuffle=True, num_workers=args_cli.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args_cli.batch_size, shuffle=False, num_workers=args_cli.num_workers)

    policy = BCPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    print(f"\n开始训练: {args_cli.epochs} 轮, 批次大小: {args_cli.batch_size}")
    print("=" * 60)

    best_val_loss = float('inf')
    for epoch in range(args_cli.epochs):
        # 训练阶段
        policy.train()
        train_loss = 0.0
        num_batches = 0
        for batch_states, batch_actions in train_loader:
            # 在训练循环中移动到GPU
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            # 数据增强：添加噪声
            if args_cli.obs_noise > 0 or args_cli.action_noise > 0:
                batch_states, batch_actions = apply_augmentation(
                    batch_states, batch_actions,
                    obs_noise=args_cli.obs_noise,
                    action_noise=args_cli.action_noise
                )

            pred_actions = policy(batch_states)
            loss = criterion(pred_actions, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / num_batches

        # 验证阶段
        policy.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                pred_actions = policy(batch_states)
                loss = criterion(pred_actions, batch_actions)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args_cli.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    print("=" * 60)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")

    os.makedirs(args_cli.output_path, exist_ok=True)
    model_path = os.path.join(args_cli.output_path, "bc_policy.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_dims': [256, 256, 128],
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std,
        'sample_count': len(states),
    }, model_path)
    print(f"模型已保存: {model_path}")

    # 最终验证
    policy.eval()
    with torch.no_grad():
        final_val_loss = 0.0
        final_batches = 0
        for batch_states, batch_actions in val_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            pred_actions = policy(batch_states)
            loss = criterion(pred_actions, batch_actions)
            final_val_loss += loss.item()
            final_batches += 1
    print(f"最终验证损失: {final_val_loss / final_batches:.6f}")


if __name__ == "__main__":
    train_bc()