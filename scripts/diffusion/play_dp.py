#!/usr/bin/env python3
"""在 Isaac Lab 中演示 Push-T Diffusion Policy.

支持:
- 从 checkpoint 自动识别 DDPM/DDIM
- 同步规划与异步预取规划
- 动作平滑、增量限制和安全裁剪
"""

import argparse
import os
import sys
import threading
import time
from queue import Empty, Full, Queue
from collections import deque

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from isaaclab.app import AppLauncher

from model import ConditionalUnet1D
from policy import DiffusionUnetLowdimPolicy
from utils import LinearNormalizer

parser = argparse.ArgumentParser(description="Play diffusion policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion checkpoint (model.pt).")
parser.add_argument("--task", type=str, default="Template-Franka-Panda-Push-T-EePos-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument(
    "--execute_horizon",
    type=int,
    default=1,
    help="How many planned actions to execute before replanning. Default=1 (replan every step).",
)
parser.add_argument(
    "--clamp_action",
    action="store_true",
    help="Clamp final action to physical workspace bounds before env.step.",
)
parser.add_argument(
    "--deterministic_sampling",
    action="store_true",
    help="Use deterministic diffusion sampling.",
)
# Backward-compat flag from older scripts. Kept hidden on help.
parser.add_argument("--stochastic_sampling", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--sampling_noise_scale", type=float, default=1.0, help="Noise scale if stochastic sampling is used.")
parser.add_argument(
    "--prefetch_trigger",
    type=int,
    default=2,
    help="Trigger async replanning when pending actions <= this value (e.g. 2 means replan at step 6/8).",
)
parser.add_argument(
    "--sync_planning",
    action="store_true",
    help="禁用异步规划线程，使用同步 plan->act 循环（质量优先）。",
)
parser.add_argument(
    "--action_ema_alpha",
    type=float,
    default=1.0,
    help="EMA alpha for executed action smoothing. 1.0 disables EMA.",
)
parser.add_argument(
    "--max_action_delta",
    type=float,
    default=0.0,
    help="Per-step max abs delta for each action dim. <=0 disables delta limiting.",
)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source", "push_T"))
import push_T.tasks.manager_based.franka_panda  # noqa: F401

X_RANGE = (0.15, 0.7)
Y_RANGE = (-0.4, 0.4)


@hydra_task_config(args_cli.task, "env_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg=None):
    """加载模型并执行闭环演示."""
    if hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    model_device = env_cfg.sim.device if hasattr(env_cfg.sim, "device") else "cuda:0"
    load_device = model_device
    if str(model_device).startswith("cuda") and not torch.cuda.is_available():
        model_device = "cpu"
        load_device = "cpu"
        env_cfg.sim.device = "cpu"
        print("[WARN] CUDA unavailable, fallback to CPU.")

    if not os.path.isfile(args_cli.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, map_location=load_device, weights_only=False)
    cfg = ckpt["config"]
    obs_dim = int(cfg["obs_dim"])
    action_dim = int(cfg["action_dim"])
    horizon = int(cfg["horizon"])
    n_obs_steps = int(cfg["n_obs_steps"])
    n_action_steps = int(cfg["n_action_steps"])
    num_inference_steps = int(cfg.get("num_inference_steps", 100))
    if args_cli.num_inference_steps is not None:
        num_inference_steps = int(args_cli.num_inference_steps)

    model = ConditionalUnet1D(
        input_dim=action_dim,
        local_cond_dim=None,
        global_cond_dim=obs_dim * n_obs_steps,
        diffusion_step_embed_dim=256,
        down_dims=tuple(cfg["down_dims"]),
        kernel_size=int(cfg["kernel_size"]),
        n_groups=8,
        cond_predict_scale=True,
    ).to(model_device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    scheduler_config = ckpt.get("scheduler_config", None)
    if scheduler_config is None:
        # Backward-compat for early checkpoints that stored scheduler under "scheduler": {"config": ...}
        legacy_scheduler = ckpt.get("scheduler", None)
        if isinstance(legacy_scheduler, dict) and "config" in legacy_scheduler:
            scheduler_config = legacy_scheduler["config"]
        elif isinstance(legacy_scheduler, dict):
            scheduler_config = legacy_scheduler
        else:
            raise KeyError("Checkpoint missing 'scheduler_config' (and legacy 'scheduler').")

    scheduler_config = dict(scheduler_config)
    scheduler_class = str(scheduler_config.get("_class_name", cfg.get("scheduler_type", "DDPMScheduler")))
    if "ddim" in scheduler_class.lower():
        noise_scheduler = DDIMScheduler.from_config(scheduler_config)
    else:
        noise_scheduler = DDPMScheduler.from_config(scheduler_config)
    print(f"[INFO] loaded scheduler: {scheduler_class}")
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_local_cond=bool(cfg.get("obs_as_local_cond", False)),
        obs_as_global_cond=bool(cfg.get("obs_as_global_cond", True)),
        pred_action_steps_only=bool(cfg.get("pred_action_steps_only", False)),
        oa_step_convention=bool(cfg.get("oa_step_convention", True)),
    ).to(model_device)
    normalizer = LinearNormalizer.from_state_dict(ckpt["normalizer"]).to(model_device)
    policy.set_normalizer(normalizer)
    policy.eval()

    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="./videos/dp_play",
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    obs, _ = env.reset()
    obs_policy = obs["policy"] if isinstance(obs, dict) else obs
    obs_queue = deque(maxlen=n_obs_steps)
    current_obs = obs_policy[0].detach().to(model_device)
    for _ in range(n_obs_steps):
        obs_queue.append(current_obs)

    pending_actions = []
    execute_horizon = max(1, int(args_cli.execute_horizon))
    prefetch_trigger = max(0, int(args_cli.prefetch_trigger))
    total_reward = 0.0
    episode_count = 0
    prev_action = None
    planning_inflight = False
    episode_token = 0

    deterministic_sampling = True
    if bool(args_cli.stochastic_sampling):
        deterministic_sampling = False
    if bool(args_cli.deterministic_sampling):
        deterministic_sampling = True

    use_async_planning = (not bool(args_cli.sync_planning)) and (args_cli.num_envs == 1)
    if (not bool(args_cli.sync_planning)) and args_cli.num_envs != 1:
        print("[WARN] Async planning currently supports num_envs=1 only. Falling back to synchronous planning.")

    request_q: Queue | None = None
    result_q: Queue | None = None
    stop_event: threading.Event | None = None
    planner_thread: threading.Thread | None = None

    def _plan_action_chunk(obs_hist: torch.Tensor) -> list[torch.Tensor]:
        """执行一次扩散规划，返回动作片段（先缓存到 CPU）."""
        obs_hist = obs_hist.to(model_device)
        with torch.no_grad():
            result = policy.predict_action(
                {"obs": obs_hist},
                deterministic_sampling=deterministic_sampling,
                sampling_noise_scale=float(args_cli.sampling_noise_scale),
            )
        action_chunk = result["action"]
        use_n = min(execute_horizon, action_chunk.shape[1])
        # Keep pending actions on CPU; move to env device only when executing a step.
        return [action_chunk[0, i].detach().cpu().clone() for i in range(use_n)]

    def _clear_queue(q: Queue):
        while True:
            try:
                q.get_nowait()
            except Empty:
                break

    if use_async_planning:
        request_q = Queue(maxsize=1)
        result_q = Queue(maxsize=2)
        stop_event = threading.Event()

        def _planner_loop():
            while not stop_event.is_set():
                try:
                    req_token, obs_hist_cpu = request_q.get(timeout=0.05)
                except Empty:
                    continue
                planned = _plan_action_chunk(obs_hist_cpu)
                try:
                    result_q.put_nowait((req_token, planned))
                except Full:
                    # Keep latest result if queue is full.
                    try:
                        result_q.get_nowait()
                    except Empty:
                        pass
                    try:
                        result_q.put_nowait((req_token, planned))
                    except Full:
                        pass

        planner_thread = threading.Thread(target=_planner_loop, daemon=True)
        planner_thread.start()

    def _submit_plan_request():
        nonlocal planning_inflight
        if (not use_async_planning) or planning_inflight:
            return
        obs_hist_cpu = torch.stack(list(obs_queue), dim=0).unsqueeze(0).float().cpu()
        try:
            request_q.put_nowait((episode_token, obs_hist_cpu))
            planning_inflight = True
        except Full:
            pass

    def _drain_plan_results():
        """把异步线程产出的动作片段追加到待执行队列."""
        nonlocal planning_inflight
        if not use_async_planning:
            return
        while True:
            try:
                result_token, planned_actions = result_q.get_nowait()
            except Empty:
                break
            planning_inflight = False
            if result_token == episode_token:
                pending_actions.extend(planned_actions)

    def _wait_for_valid_plan(timeout_s: float) -> bool:
        """在超时时间内等待当前 episode 的有效规划结果."""
        nonlocal planning_inflight
        if not use_async_planning:
            return False
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            try:
                result_token, planned_actions = result_q.get(timeout=0.1)
            except Empty:
                continue
            planning_inflight = False
            if result_token != episode_token:
                continue
            if len(planned_actions) == 0:
                continue
            pending_actions.extend(planned_actions)
            return True
        return False

    if use_async_planning:
        _submit_plan_request()
        if not _wait_for_valid_plan(timeout_s=3.0):
            # Startup fallback
            obs_hist = torch.stack(list(obs_queue), dim=0).unsqueeze(0).float()
            pending_actions = _plan_action_chunk(obs_hist)
            planning_inflight = False

    try:
        with torch.no_grad():
            for step in range(args_cli.steps):
                if use_async_planning:
                    _drain_plan_results()
                    if (len(pending_actions) <= prefetch_trigger) and (not planning_inflight):
                        _submit_plan_request()
                else:
                    if len(pending_actions) == 0:
                        obs_hist = torch.stack(list(obs_queue), dim=0).unsqueeze(0).float()
                        pending_actions = _plan_action_chunk(obs_hist)

                if len(pending_actions) == 0 and use_async_planning:
                    # Quality-first: never reuse previous action.
                    # Wait for next planned chunk; if it is still unavailable, plan once synchronously.
                    got = _wait_for_valid_plan(timeout_s=1.0)
                    if not got:
                        obs_hist = torch.stack(list(obs_queue), dim=0).unsqueeze(0).float()
                        pending_actions = _plan_action_chunk(obs_hist)
                        planning_inflight = False
                if len(pending_actions) == 0:
                    obs_hist = torch.stack(list(obs_queue), dim=0).unsqueeze(0).float()
                    pending_actions = _plan_action_chunk(obs_hist)

                action = pending_actions.pop(0).unsqueeze(0).to(env.unwrapped.device)
                if prev_action is not None:
                    alpha = float(args_cli.action_ema_alpha)
                    action = alpha * action + (1.0 - alpha) * prev_action
                    max_delta = float(args_cli.max_action_delta)
                    if max_delta > 0:
                        delta = (action - prev_action).clamp(-max_delta, max_delta)
                        action = prev_action + delta
                if args_cli.clamp_action:
                    action[:, 0] = action[:, 0].clamp(X_RANGE[0], X_RANGE[1])
                    action[:, 1] = action[:, 1].clamp(Y_RANGE[0], Y_RANGE[1])
                prev_action = action.detach().clone()
                obs, reward, terminated, truncated, _ = env.step(action)

                if isinstance(reward, torch.Tensor):
                    total_reward += reward.float().mean().item()
                else:
                    total_reward += float(reward)

                obs_policy = obs["policy"] if isinstance(obs, dict) else obs
                current_obs = obs_policy[0].detach().to(model_device)
                obs_queue.append(current_obs)

                if bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item()):
                    episode_count += 1
                    episode_token += 1
                    obs, _ = env.reset()
                    obs_policy = obs["policy"] if isinstance(obs, dict) else obs
                    current_obs = obs_policy[0].detach().to(model_device)
                    obs_queue.clear()
                    for _ in range(n_obs_steps):
                        obs_queue.append(current_obs)
                    pending_actions.clear()
                    prev_action = None
                    planning_inflight = False
                    if use_async_planning:
                        _clear_queue(request_q)
                        _clear_queue(result_q)
                        _submit_plan_request()

                if (step + 1) % 200 == 0:
                    print(f"Step {step+1:4d}/{args_cli.steps}")
    finally:
        if planner_thread is not None:
            stop_event.set()
            planner_thread.join(timeout=1.0)

    print(f"Done. Episodes: {episode_count}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
