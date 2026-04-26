#!/usr/bin/env python3
"""训练 Push-T 低维 Diffusion Policy.

当前脚本支持:
- DDPM / DDIM 两种调度器
- EMA 权重评估与保存
- 早停与 best checkpoint 导出
"""

import argparse
import os
from copy import deepcopy

import torch
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

from dataset import PushTLowdimDataset
from model import ConditionalUnet1D
from policy import DiffusionUnetLowdimPolicy


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True).float() for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser(description="Train diffusion policy from demonstrations.")
    # demo_path 可传目录或具体 npz 文件路径
    parser.add_argument("--demo_path", type=str, default="./demonstrations")
    # 默认输出到当前命名规范的 DDPM 目录
    parser.add_argument("--output_path", type=str, default="./logs/diffusion_policy/pushT_ddpm")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--scheduler_type", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", choices=["linear", "squaredcos_cap_v2"])
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample"])
    parser.add_argument("--down_dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulate_every", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--ema_min_decay", type=float, default=0.0)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=0.75)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    train_dataset = PushTLowdimDataset(
        demo_path=args.demo_path,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        seed=args.seed,
        val_ratio=args.val_ratio,
        is_validation=False,
    )
    val_dataset = train_dataset.get_validation_dataset()
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Train/val dataset is empty. Check horizons and demonstrations.")

    obs_dim = int(train_dataset.states.shape[1])
    action_dim = int(train_dataset.actions.shape[1])
    print(
        f"[INFO] windows | train={len(train_dataset)} val={len(val_dataset)} "
        f"| obs_dim={obs_dim} act_dim={action_dim}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = ConditionalUnet1D(
        input_dim=action_dim,
        local_cond_dim=None,
        global_cond_dim=obs_dim * args.n_obs_steps,
        diffusion_step_embed_dim=256,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=8,
        cond_predict_scale=True,
    ).to(device)
    if args.scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.diffusion_steps,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule=args.beta_schedule,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule=args.beta_schedule,
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type=args.prediction_type,
        )
    num_inference_steps = args.diffusion_steps if args.num_inference_steps is None else int(args.num_inference_steps)
    print(
        f"[INFO] scheduler={args.scheduler_type} | "
        f"train_timesteps={args.diffusion_steps} | inference_steps={num_inference_steps}"
    )
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=args.horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=args.n_action_steps,
        n_obs_steps=args.n_obs_steps,
        num_inference_steps=num_inference_steps,
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        oa_step_convention=True,
    ).to(device)
    normalizer = train_dataset.get_normalizer(mode="limits").to(device)
    policy.set_normalizer(normalizer)

    optimizer = torch.optim.AdamW(
        policy.model.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    total_train_batches = len(train_loader) * args.epochs
    num_training_steps = max(total_train_batches // max(args.gradient_accumulate_every, 1), 1)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    use_ema = not args.disable_ema
    ema = None
    if use_ema:
        ema = EMAModel(
            parameters=policy.model.parameters(),
            decay=args.ema_max_decay,
            min_decay=args.ema_min_decay,
            update_after_step=args.ema_update_after_step,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
        )

    best_val = float("inf")
    best_epoch = 0
    best_model_state = None
    stale_epochs = 0
    best_ckpt_path = os.path.join(args.output_path, "best_model.pt")
    global_step = 0

    def _build_ckpt(model_state_dict, epoch: int, best_val_noise_mse: float):
        return {
            "model_state_dict": model_state_dict,
            "normalizer": normalizer.state_dict(),
            "scheduler_config": dict(noise_scheduler.config),
            "config": {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "horizon": args.horizon,
                "n_obs_steps": args.n_obs_steps,
                "n_action_steps": args.n_action_steps,
                "scheduler_type": args.scheduler_type,
                "num_inference_steps": num_inference_steps,
                "down_dims": args.down_dims,
                "kernel_size": args.kernel_size,
                "obs_as_local_cond": False,
                "obs_as_global_cond": True,
                "pred_action_steps_only": False,
                "oa_step_convention": True,
            },
            "best_val_noise_mse": float(best_val_noise_mse),
            "best_epoch": int(epoch),
            "use_ema": use_ema,
            "ema": {
                "max_decay": args.ema_max_decay,
                "min_decay": args.ema_min_decay,
                "inv_gamma": args.ema_inv_gamma,
                "power": args.ema_power,
                "update_after_step": args.ema_update_after_step,
                "use_warmup": True,
            },
        }

    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            raw_loss = policy.compute_loss(batch)
            loss = raw_loss / max(args.gradient_accumulate_every, 1)
            loss.backward()

            do_step = ((batch_idx + 1) % max(args.gradient_accumulate_every, 1) == 0) or (
                batch_idx == len(train_loader) - 1
            )
            if do_step:
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if ema is not None:
                    ema.step(policy.model.parameters())
                global_step += 1

            train_loss += raw_loss.item()
            if args.max_train_steps is not None and batch_idx >= (args.max_train_steps - 1):
                break

        train_loss /= max(min(len(train_loader), args.max_train_steps or len(train_loader)), 1)

        eval_policy = policy
        if ema is not None:
            eval_policy = deepcopy(policy)
            ema.copy_to(eval_policy.model.parameters())
            eval_policy.to(device)
        eval_policy.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = _to_device(batch, device)
                val_loss += eval_policy.compute_loss(batch).item()
                if args.max_val_steps is not None and batch_idx >= (args.max_val_steps - 1):
                    break

        val_loss /= max(min(len(val_loader), args.max_val_steps or len(val_loader)), 1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | Train Noise MSE: {train_loss:.6f} | "
            f"Val Noise MSE: {val_loss:.6f} | LR: {current_lr:.7f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            if ema is not None:
                best_eval = deepcopy(policy)
                ema.copy_to(best_eval.model.parameters())
                best_model_state = {k: v.detach().cpu().clone() for k, v in best_eval.model.state_dict().items()}
            else:
                best_model_state = {k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items()}
            torch.save(_build_ckpt(best_model_state, best_epoch, best_val), best_ckpt_path)
            print(f"  [INFO] new best saved: {best_ckpt_path} (epoch={best_epoch}, val={best_val:.6f})")
        else:
            stale_epochs += 1

        if stale_epochs >= args.early_stop_patience:
            print(f"[INFO] Early stopping at epoch {epoch} (no val improvement for {stale_epochs} epochs).")
            break

    if best_model_state is None:
        raise RuntimeError("Training finished without valid checkpoint state.")

    ckpt = _build_ckpt(best_model_state, best_epoch, best_val)
    best_alias_path = os.path.join(args.output_path, "model.pt")
    torch.save(ckpt, best_alias_path)
    print(f"[INFO] best epoch: {best_epoch} | best val noise mse: {best_val:.6f}")
    print(f"[INFO] saved best alias: {best_alias_path}")
    print(f"[INFO] saved best checkpoint: {best_ckpt_path}")

    if args.save_last:
        if ema is not None:
            last_eval = deepcopy(policy)
            ema.copy_to(last_eval.model.parameters())
            last_state = {k: v.detach().cpu().clone() for k, v in last_eval.model.state_dict().items()}
        else:
            last_state = {k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items()}
        last_ckpt_path = os.path.join(args.output_path, "last_model.pt")
        torch.save(_build_ckpt(last_state, epoch, val_loss), last_ckpt_path)
        print(f"[INFO] saved last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
