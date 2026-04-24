from __future__ import annotations

"""MuJoCo 中的 Push-T 策略验证脚本。

该脚本负责：
1. 加载 Isaac Lab 训练得到的 DDPM / DDIM checkpoint；
2. 构建 MuJoCo Push-T 环境；
3. 在 MuJoCo 中回放并统计成功率。

"""

import argparse
import sys
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from mujoco_push_t_env import MujocoPushTEnv


PROJECT_ROOT = Path(__file__).parent.parent
DIFFUSION_DIR = PROJECT_ROOT / "scripts" / "diffusion"
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

X_RANGE = (0.15, 0.70)
Y_RANGE = (-0.40, 0.40)


def load_policy(checkpoint_path: Path, device_arg: str, num_inference_steps: int | None):
    """加载 diffusion policy checkpoint，并恢复模型 / scheduler / normalizer。"""
    try:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'diffusers'. Install it in the Python environment used for MuJoCo validation."
        ) from exc

    from model import ConditionalUnet1D
    from policy import DiffusionUnetLowdimPolicy
    from utils import LinearNormalizer

    # 若命令行要求 CUDA 但当前环境不可用，则自动回退到 CPU。
    requested_device = device_arg
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
        print("[WARN] CUDA unavailable, falling back to CPU.")
    device = torch.device(requested_device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    obs_dim = int(config["obs_dim"])
    action_dim = int(config["action_dim"])
    horizon = int(config["horizon"])
    n_obs_steps = int(config["n_obs_steps"])
    n_action_steps = int(config["n_action_steps"])
    if obs_dim != 8 or action_dim != 2:
        raise ValueError(f"Unexpected checkpoint dimensions: obs_dim={obs_dim}, action_dim={action_dim}.")

    model = ConditionalUnet1D(
        input_dim=action_dim,
        local_cond_dim=None,
        global_cond_dim=obs_dim * n_obs_steps,
        diffusion_step_embed_dim=256,
        down_dims=tuple(config["down_dims"]),
        kernel_size=int(config["kernel_size"]),
        n_groups=8,
        cond_predict_scale=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # 新旧 checkpoint 的 scheduler 字段命名略有差异，这里做兼容处理。
    scheduler_config = checkpoint.get("scheduler_config")
    if scheduler_config is None:
        legacy_scheduler = checkpoint.get("scheduler", None)
        if isinstance(legacy_scheduler, dict) and "config" in legacy_scheduler:
            scheduler_config = legacy_scheduler["config"]
        elif isinstance(legacy_scheduler, dict):
            scheduler_config = legacy_scheduler
        else:
            raise KeyError("Checkpoint missing scheduler configuration.")
    scheduler_config = dict(scheduler_config)
    scheduler_class_name = str(scheduler_config.get("_class_name", config.get("scheduler_type", "DDPMScheduler")))
    if "ddim" in scheduler_class_name.lower():
        scheduler = DDIMScheduler.from_config(scheduler_config)
    else:
        scheduler = DDPMScheduler.from_config(scheduler_config)

    # 允许命令行覆盖 checkpoint 中保存的推理步数。
    resolved_inference_steps = int(config.get("num_inference_steps", 100))
    if num_inference_steps is not None:
        resolved_inference_steps = int(num_inference_steps)

    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=resolved_inference_steps,
        obs_as_local_cond=bool(config.get("obs_as_local_cond", False)),
        obs_as_global_cond=bool(config.get("obs_as_global_cond", True)),
        pred_action_steps_only=bool(config.get("pred_action_steps_only", False)),
        oa_step_convention=bool(config.get("oa_step_convention", True)),
    ).to(device)
    normalizer = LinearNormalizer.from_state_dict(checkpoint["normalizer"]).to(device)
    policy.set_normalizer(normalizer)
    policy.eval()

    return policy, device, config


def postprocess_action(
    action: np.ndarray,
    prev_action: np.ndarray | None,
    use_clamp: bool,
    ema_alpha: float,
    max_action_delta: float,
) -> np.ndarray:
    """对策略输出动作做平滑、增量限制和工作空间裁剪。"""
    processed = action.astype(np.float32, copy=True)
    if prev_action is not None:
        processed = ema_alpha * processed + (1.0 - ema_alpha) * prev_action
        if max_action_delta > 0:
            delta = np.clip(processed - prev_action, -max_action_delta, max_action_delta)
            processed = prev_action + delta
    if use_clamp:
        processed[0] = np.clip(processed[0], *X_RANGE)
        processed[1] = np.clip(processed[1], *Y_RANGE)
    return processed


def main():
    """命令行入口：在 MuJoCo 中执行若干局策略验证。"""
    parser = argparse.ArgumentParser(description="Validate a Push-T diffusion policy in MuJoCo.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the diffusion checkpoint.")
    parser.add_argument(
        "--model_xml",
        type=Path,
        default=Path("sim2sim") / "model" / "push_t_scene.xml",
        help="MuJoCo scene XML for Push-T validation.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps_per_episode", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execute_horizon", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--deterministic_sampling", action="store_true")
    parser.add_argument("--sampling_noise_scale", type=float, default=1.0)
    parser.add_argument("--clamp_action", action="store_true")
    parser.add_argument("--action_ema_alpha", type=float, default=1.0)
    parser.add_argument("--max_action_delta", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy, device, config = load_policy(args.checkpoint, args.device, args.num_inference_steps)
    env = MujocoPushTEnv(args.model_xml, max_steps=args.steps_per_episode, seed=args.seed)

    viewer_context = nullcontext(None)
    if not args.headless:
        # 非 headless 模式下启动 MuJoCo 被动 viewer，用于实时演示。
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=False,
            show_right_ui=False,
        )

    successes = 0
    episode_lengths: list[int] = []

    print(
        "[INFO] Loaded checkpoint | "
        f"obs_dim={config['obs_dim']} act_dim={config['action_dim']} "
        f"horizon={config['horizon']} n_obs_steps={config['n_obs_steps']} "
        f"n_action_steps={config['n_action_steps']}"
    )
    print(f"[INFO] MuJoCo scene: {args.model_xml}")
    print(f"[INFO] Control dt: {env.control_dt:.4f}s | Episodes: {args.episodes}")

    with viewer_context as viewer:
        abort = False
        for episode_index in range(args.episodes):
            # 每局使用不同 seed，便于复现实验同时保证初始状态可区分。
            observation, reset_info = env.reset(seed=args.seed + episode_index)
            obs_queue: deque[np.ndarray] = deque(
                [observation.copy() for _ in range(int(config["n_obs_steps"]))],
                maxlen=int(config["n_obs_steps"]),
            )
            pending_actions: list[np.ndarray] = []
            prev_action: np.ndarray | None = None
            episode_success = False
            episode_reward = 0.0
            executed_steps = 0
            info = {
                "success": bool(reset_info["success"]),
                "position_distance": float(reset_info["position_distance"]),
                "rotation_distance": float(reset_info["rotation_distance"]),
            }

            if viewer is not None:
                viewer.sync()

            for step_index in range(args.steps_per_episode):
                if viewer is not None and not viewer.is_running():
                    abort = True
                    break

                loop_start = time.perf_counter()
                if not pending_actions:
                    # 当缓存动作耗尽时，使用最近 n_obs_steps 帧观测重新规划。
                    obs_hist = torch.from_numpy(np.stack(obs_queue, axis=0)).unsqueeze(0).to(device=device).float()
                    with torch.no_grad():
                        result = policy.predict_action(
                            {"obs": obs_hist},
                            deterministic_sampling=args.deterministic_sampling,
                            sampling_noise_scale=float(args.sampling_noise_scale),
                        )
                    action_plan = result["action"][0].detach().cpu().numpy()
                    use_n = min(max(1, int(args.execute_horizon)), action_plan.shape[0])
                    pending_actions = [action_plan[action_index].astype(np.float32, copy=True) for action_index in range(use_n)]

                action = postprocess_action(
                    action=pending_actions.pop(0),
                    prev_action=prev_action,
                    use_clamp=args.clamp_action,
                    ema_alpha=float(args.action_ema_alpha),
                    max_action_delta=float(args.max_action_delta),
                )
                prev_action = action.copy()
                observation, reward, terminated, truncated, info = env.step(action)
                obs_queue.append(observation.copy())
                episode_reward += float(reward)
                executed_steps = step_index + 1

                if viewer is not None:
                    viewer.sync()
                    if args.realtime:
                        # `--realtime` 下尽量按控制周期睡眠，方便目视观察。
                        elapsed = time.perf_counter() - loop_start
                        time.sleep(max(0.0, env.control_dt - elapsed))

                if terminated or truncated:
                    episode_success = bool(info["success"])
                    episode_lengths.append(step_index + 1)
                    break
            else:
                episode_lengths.append(args.steps_per_episode)

            if len(episode_lengths) < episode_index + 1:
                episode_lengths.append(executed_steps)
            successes += int(episode_success)
            print(
                f"[EP {episode_index + 1:02d}] success={episode_success} "
                f"steps={episode_lengths[-1]} "
                f"pos_err={info['position_distance']:.4f} "
                f"rot_err={info['rotation_distance']:.4f} "
                f"reward={episode_reward:.2f} "
                f"reset_pos_err={reset_info['position_distance']:.4f}"
            )

            if abort:
                break

    total_episodes = len(episode_lengths)
    if total_episodes > 0:
        print(
            "[SUMMARY] "
            f"episodes={total_episodes} "
            f"success_rate={successes / total_episodes:.3f} "
            f"avg_steps={float(np.mean(episode_lengths)):.1f}"
        )
    env.close()


if __name__ == "__main__":
    main()
