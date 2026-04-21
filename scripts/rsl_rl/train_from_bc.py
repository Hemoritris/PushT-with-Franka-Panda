# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL from BC pretrained model."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train RL agent with RSL-RL from pretrained checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--bc_model", type=str, required=True, help="Path to RSL-RL format checkpoint (.pt file).")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--init_entropy_coef", type=float, default=0.001, help="Initial entropy coefficient for exploration.")
parser.add_argument("--init_learning_rate", type=float, default=3e-5, help="Initial learning rate for RL fine-tuning.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""
import importlib.metadata as metadata
import platform
from packaging import version

RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
          f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
          f"\n\n\t{' '.join(cmd)}\n")
    exit(1)

"""Rest everything follows."""
import gymnasium as gym
import torch
from datetime import datetime
import omni

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import push_T.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def load_rsl_rl_checkpoint(checkpoint_path, device):
    """加载RSL-RL格式的checkpoint."""
    print(f"[INFO] Loading RSL-RL checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    model_state_dict = checkpoint['model_state_dict']
    obs_norm_state_dict = checkpoint.get('obs_norm_state_dict', None)

    print(f"[INFO] RSL-RL checkpoint loaded, iter: {checkpoint.get('iter', 0)}")

    return model_state_dict, obs_norm_state_dict


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train from BC pretrained model."""
    # override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    # 调整超参数用于BC微调
    agent_cfg.algorithm.entropy_coef = args_cli.init_entropy_coef
    agent_cfg.algorithm.learning_rate = args_cli.init_learning_rate
    print(f"[INFO] Using entropy_coef: {agent_cfg.algorithm.entropy_coef}")
    print(f"[INFO] Using learning_rate: {agent_cfg.algorithm.learning_rate}")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        log_dir += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set IO descriptors output directory
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn("IO descriptors are only supported for manager based RL environments.")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # 加载RSL-RL格式的预训练模型
    device = torch.device(agent_cfg.device)
    model_state_dict, _ = load_rsl_rl_checkpoint(args_cli.bc_model, device)

    # 加载权重到runner
    runner.alg.policy.load_state_dict(model_state_dict, strict=False)
    print(f"[INFO] Model weights loaded")

    # 在actor输出后加tanh，限制动作范围[-1, 1]
    original_forward = runner.alg.policy.actor.forward
    def actor_forward_with_tanh(obs):
        return torch.tanh(original_forward(obs))
    runner.alg.policy.actor.forward = actor_forward_with_tanh
    print(f"[INFO] Added tanh to actor output")

    print(f"[INFO] RSL-RL pretrained model loaded, starting RL fine-tuning from iter 0")

    # dump the configuration
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # save BC model info
    bc_info = {
        "bc_model_path": args_cli.bc_model,
        "bc_pretrained": True,
    }
    dump_pickle(os.path.join(log_dir, "params", "bc_info.pkl"), bc_info)

    # 添加debug输出wrapper
    class DebugWrapper(gym.Wrapper):
        def __init__(self, env, print_every=100):
            super().__init__(env)
            self.step_count = 0
            self.print_every = print_every
            self.first_reset = True

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            if self.first_reset:
                print("[DEBUG] Initial obs:")
                if isinstance(obs, dict) and "policy" in obs:
                    p_obs = obs["policy"][0].cpu().numpy()
                    print(f"  EE pos: [{p_obs[10]:.3f}, {p_obs[11]:.3f}, {p_obs[12]:.3f}]")
                    print(f"  T pos: [{p_obs[0]:.3f}, {p_obs[1]:.3f}, {p_obs[2]:.3f}]")
                    print(f"  Target pos: [{p_obs[7]:.3f}, {p_obs[8]:.3f}, {p_obs[9]:.3f}]")
                self.first_reset = False
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.step_count += 1
            if self.step_count % self.print_every == 0 and isinstance(obs, dict) and "policy" in obs:
                p_obs = obs["policy"][0].cpu().numpy()
                action_np = action[0].cpu().numpy() if action.dim() == 2 else action.cpu().numpy()
                print(f"[Step {self.step_count}] Action: [{action_np[0]:.3f}, {action_np[1]:.3f}, {action_np[2]:.3f}]")
                print(f"  EE pos: [{p_obs[10]:.3f}, {p_obs[11]:.3f}, {p_obs[12]:.3f}]")
            return obs, reward, terminated, truncated, info

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()