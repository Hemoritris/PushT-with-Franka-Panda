# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation script for Franka Panda Push-T - End-Effector Position Control."""

import argparse
import sys
import os
import time

import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard EE teleoperation for Franka Panda Push-T.")
parser.add_argument("--task", type=str, default="Template-Franka-Panda-Push-T-EePos-v0", help="Name of the task.")
parser.add_argument("--save_path", type=str, default="./demonstrations/", help="Path to save demonstrations.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pygame
import gymnasium as gym
import torch

import push_T.tasks  # noqa: F401


# Key mapping for EE position control (3D)
# W/S - X direction (forward/backward in world frame)
# A/D - Y direction (left/right)
# UP/DOWN - Z direction (up/down)
KEY_POSE_MAP = {
    pygame.K_w: (1, 0, 0),     # +X (forward)
    pygame.K_s: (-1, 0, 0),    # -X (backward)
    pygame.K_a: (0, 1, 0),     # +Y (left)
    pygame.K_d: (0, -1, 0),    # -Y (right)
    pygame.K_UP: (0, 0, 1),    # +Z (up)
    pygame.K_DOWN: (0, 0, -1), # -Z (down)
}

POSE_STEP = 0.005  # EE position step size in meters


def main():
    """Run teleoperation with EE position control."""
    from push_T.tasks.manager_based.franka_panda.franka_panda_env_cfg import FrankaPandaEePosTeleopEnvCfg

    env_cfg = FrankaPandaEePosTeleopEnvCfg()
    env_cfg.scene.num_envs = 1

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")

    save_dir = args_cli.save_path
    os.makedirs(save_dir, exist_ok=True)

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Franka Panda EE Teleop - W/S=X, A/D=Y, UP/DOWN=Z - ESC to quit")
    font = pygame.font.Font(None, 24)

    # Reset
    obs, _ = env.reset()
    print("\n" + "="*60)
    print("FRANKA PANDA PUSH-T TELEOPERATION (EE Position Control)")
    print("="*60)
    print("Controls:")
    print("  W/S - Move EE in X (forward/backward)")
    print("  A/D - Move EE in Y (left/right)")
    print("  UP/DOWN - Move EE in Z (up/down)")
    print("  SHIFT + direction - Fine control (smaller steps)")
    print("  ESC - Quit and save")
    print("="*60 + "\n")

    # Data storage
    episode_observations = []
    episode_actions = []
    episode_count = 0
    total_saved_episodes = 0
    current_ep_obs = []
    current_ep_acts = []
    episode_start_time = time.time()

    # 获取初始末端执行器位置
    # obs["policy"] 形状: [1, 13] - EePos配置
    # [t_block_pos(3), t_block_quat(4), target_pos(3), ee_pos(3)] = 13
    policy_obs = obs["policy"][0].cpu().numpy()
    ee_pos = policy_obs[-3:]  # 最后3个是end_effector_pos
    print(f"[INFO] 初始EE位置: {ee_pos}")
    print(f"[INFO] 观测维度: {len(policy_obs)}")

    # EE position: [x, y, z] - 3维位置
    # 姿态由 FixedDownIKAction 自动固定为朝下
    current_pos = ee_pos.copy().astype(np.float32)

    try:
        while simulation_app.is_running():
            pygame.time.wait(16)  # ~60 FPS

            # Continuous key handling
            keys = pygame.key.get_pressed()
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

            # Calculate step size
            step = POSE_STEP * (0.2 if shift_held else 1.0)

            # Update EE position based on pressed keys
            for key, direction in KEY_POSE_MAP.items():
                if keys[key]:
                    current_pos += np.array(direction) * step

            # 限制EE位置范围，防止失控（与 FixedDownIKAction 中的范围一致）
            current_pos[2] = np.clip(current_pos[2], 0.13, 0.5)  # Z范围
            current_pos[0] = np.clip(current_pos[0], 0.15, 0.7)  # X范围
            current_pos[1] = np.clip(current_pos[1], -0.4, 0.4)  # Y范围

            # Apply action - 将绝对位置转换为 [-1, 1] 范围的action（与FixedDownIKAction配合）
            # X_RANGE = (0.15, 0.7), Y_RANGE = (-0.4, 0.4), Z_RANGE = (0.13, 0.5)
            action_normalized = np.zeros(3, dtype=np.float32)
            action_normalized[0] = (current_pos[0] - 0.15) / (0.7 - 0.15) * 2.0 - 1.0  # X
            action_normalized[1] = (current_pos[1] - (-0.4)) / (0.4 - (-0.4)) * 2.0 - 1.0  # Y
            action_normalized[2] = (current_pos[2] - 0.13) / (0.5 - 0.13) * 2.0 - 1.0  # Z
            action_tensor = torch.from_numpy(action_normalized).float().unsqueeze(0)

            # Record - 提取policy观测的实际数据
            policy_obs = obs["policy"][0].cpu().numpy()  # [obs_dim]
            current_ep_obs.append(policy_obs)
            current_ep_acts.append(action_normalized.copy())  # 记录normalized action

            # Step
            obs, reward, terminated, truncated, info = env.step(action_tensor)

            # Update display
            elapsed = time.time() - episode_start_time
            screen.fill((30, 30, 30))
            texts = [
                f"Episodes: {episode_count} | Current: {len(current_ep_obs)} steps",
                f"EE pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]",
                f"obs pos: [{policy_obs[10]:.3f}, {policy_obs[11]:.3f}, {policy_obs[12]:.3f}]",
                f"obs t_pos: [{policy_obs[0]:.3f}, {policy_obs[1]:.3f}, {policy_obs[2]:.3f}]",
                f"obs t_rot: [{policy_obs[3]:.3f}, {policy_obs[4]:.3f}, {policy_obs[5]:.3f}, {policy_obs[6]:.3f}]",
                f"obs target_pos: [{policy_obs[7]:.3f}, {policy_obs[8]:.3f}, {policy_obs[9]:.3f}]",
                f"Obs dim: {len(policy_obs)} | Action dim: 3",
                "",
                "W/S = X, A/D = Y, UP/DOWN = Z",
                "U = save all episodes",
                "ESC = quit & save"
            ]
            for i, text in enumerate(texts):
                screen.blit(font.render(text, True, (200, 200, 200)), (20, 20 + i * 25))
            pygame.display.flip()

            # Handle keyboard events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # 保存并退出
                        if len(episode_observations) > 0:
                            # 保存当前未完成的episode
                            if len(current_ep_obs) > 10:
                                episode_observations.append(current_ep_obs)
                                episode_actions.append(current_ep_acts)
                            _save_demonstrations(episode_observations, episode_actions, save_dir)
                            print(f"[INFO] 已保存 {len(episode_observations)} 个演示轨迹")
                        pygame.quit()
                        env.close()
                        simulation_app.close()
                        return
                    elif event.key == pygame.K_u:
                        # 按U键手动保存当前轨迹
                        if len(episode_observations) > 0:
                            # 先保存当前未完成的episode
                            if len(current_ep_obs) > 10:
                                episode_observations.append(current_ep_obs)
                                episode_actions.append(current_ep_acts)
                            _save_demonstrations(episode_observations, episode_actions, save_dir)
                            print(f"[INFO] 已保存 {len(episode_observations)} 个轨迹")
                            # 清空数据
                            episode_observations = []
                            episode_actions = []
                            current_ep_obs = []
                            current_ep_acts = []

            # Auto-reset on episode end
            if terminated or truncated:
                episode_count += 1
                total_saved_episodes += 1
                # 保存当前episode数据
                episode_observations.append(current_ep_obs)
                episode_actions.append(current_ep_acts)
                current_ep_obs = []
                current_ep_acts = []
                episode_start_time = time.time()
                obs, _ = env.reset()
                # 获取重置后的EE位置
                policy_obs = obs["policy"][0].cpu().numpy()
                ee_pos = policy_obs[-3:]
                current_pos[:] = ee_pos
                print(f"[INFO] Episode {episode_count} 结束, 累计保存: {total_saved_episodes}")

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if len(episode_observations) > 0:
            _save_demonstrations(episode_observations, episode_actions, save_dir)
        else:
            print("[WARNING] Less than 10 steps, skipping save")

        pygame.quit()
        env.close()
        print("[INFO] Done")


def _save_demonstrations(episode_observations, episode_actions, save_dir):
    """Save demonstrations to npz file - 每条轨迹分开保存."""
    # 将每条episode的观察和动作堆叠
    all_obs = []
    all_act = []
    traj_lengths = []

    for ep_obs, ep_acts in zip(episode_observations, episode_actions):
        if len(ep_obs) > 10:  # 只保存长度大于10的轨迹
            all_obs.extend(ep_obs)
            all_act.extend(ep_acts)
            traj_lengths.append(len(ep_obs))

    if len(all_obs) == 0:
        print("[WARNING] No valid episodes to save")
        return

    all_obs = np.array(all_obs)
    all_act = np.array(all_act)
    traj_lengths = np.array(traj_lengths)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(save_dir, f"push_t_demos_{timestamp}.npz")
    np.savez(
        filename,
        states=all_obs,
        actions=all_act,
        traj_lengths=traj_lengths,
    )
    print(f"[INFO] Saved to {filename}")
    print(f"       Episodes: {len(traj_lengths)}, Total steps: {len(all_obs)}")
    print(f"       Episode lengths: {traj_lengths}")


if __name__ == "__main__":
    main()