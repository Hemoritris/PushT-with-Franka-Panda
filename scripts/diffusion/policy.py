"""低维 Diffusion Policy 实现（兼容 DDPM / DDIM）.

包含:
- 条件掩码生成
- 训练损失计算
- 推理阶段的条件采样与动作切片
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from model import ConditionalUnet1D
from utils import LinearNormalizer


class LowdimMaskGenerator:
    """低维条件掩码生成器（官方风格）."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ):
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.max_n_obs_steps = int(max_n_obs_steps)
        self.fix_obs_steps = bool(fix_obs_steps)
        self.action_visible = bool(action_visible)

    @torch.no_grad()
    def __call__(self, shape: tuple[int, int, int], device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
        batch_size, horizon, dim = shape
        assert dim == (self.action_dim + self.obs_dim)

        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        is_action_dim = torch.zeros_like(mask)
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full((batch_size,), fill_value=self.max_n_obs_steps, device=device, dtype=torch.long)
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(batch_size,),
                generator=rng,
                device=device,
                dtype=torch.long,
            )

        steps = torch.arange(0, horizon, device=device).reshape(1, horizon).expand(batch_size, horizon)
        obs_mask = (steps.T < obs_steps).T.reshape(batch_size, horizon, 1).expand(batch_size, horizon, dim)
        obs_mask = obs_mask & is_obs_dim

        mask = obs_mask
        if self.action_visible:
            action_steps = torch.maximum(obs_steps - 1, torch.tensor(0, device=device, dtype=torch.long))
            action_mask = (steps.T < action_steps).T.reshape(batch_size, horizon, 1).expand(batch_size, horizon, dim)
            action_mask = action_mask & is_action_dim
            mask = mask | action_mask
        return mask


class DiffusionUnetLowdimPolicy(nn.Module):
    """低维 Diffusion 策略."""

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: Union[DDPMScheduler, DDIMScheduler],
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = True,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = True,
    ):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_local_cond = bool(obs_as_local_cond)
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.pred_action_steps_only = bool(pred_action_steps_only)
        self.oa_step_convention = bool(oa_step_convention)
        if num_inference_steps is None:
            num_inference_steps = int(noise_scheduler.config.num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)

    def _is_ddim_scheduler(self) -> bool:
        """判断当前 scheduler 是否为 DDIM."""
        class_name = getattr(self.noise_scheduler.config, "_class_name", self.noise_scheduler.__class__.__name__)
        return "ddim" in str(class_name).lower()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def _step_mean(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """DDPM 确定性均值步（不注入随机噪声）."""
        t = int(timestep)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=sample.device, dtype=sample.dtype)
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(1.0 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif pred_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        if self.noise_scheduler.config.clip_sample:
            clip_range = getattr(self.noise_scheduler.config, "clip_sample_range", 1.0)
            pred_original_sample = pred_original_sample.clamp(-clip_range, clip_range)

        pred_original_sample_coeff = (torch.sqrt(alpha_prod_t_prev) * current_beta_t) / (1.0 - alpha_prod_t)
        current_sample_coeff = torch.sqrt(current_alpha_t) * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        return pred_prev_sample

    @torch.no_grad()
    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        init_trajectory: Optional[torch.Tensor] = None,
        deterministic_sampling: bool = False,
        sampling_noise_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """扩散反向采样.

        - DDIM 分支使用 `eta` 控制随机性
        - DDPM 分支支持确定性与随机两种模式
        """
        if init_trajectory is None:
            trajectory = torch.randn_like(condition_data)
        else:
            trajectory = init_trajectory.to(condition_data.device).to(condition_data.dtype).clone()
            if trajectory.shape != condition_data.shape:
                raise ValueError(
                    f"init_trajectory shape mismatch: expected {condition_data.shape}, got {trajectory.shape}"
                )

        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=condition_data.device)
        for t in self.noise_scheduler.timesteps:
            # 每个扩散步都先写回条件位，防止条件被噪声覆盖
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            if self._is_ddim_scheduler():
                # DDIM: eta=0 为确定性推理
                eta = 0.0 if deterministic_sampling else max(0.0, min(1.0, float(sampling_noise_scale)))
                trajectory = self.noise_scheduler.step(
                    model_output,
                    int(t.item()),
                    trajectory,
                    generator=generator,
                    eta=eta,
                ).prev_sample
            else:
                # DDPM 分支
                if deterministic_sampling:
                    trajectory = self._step_mean(model_output, int(t.item()), trajectory)
                else:
                    if abs(float(sampling_noise_scale) - 1.0) < 1e-8:
                        trajectory = self.noise_scheduler.step(
                            model_output,
                            int(t.item()),
                            trajectory,
                            generator=generator,
                        ).prev_sample
                    else:
                        mean = self._step_mean(model_output, int(t.item()), trajectory)
                        if int(t.item()) > 0:
                            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
                                device=trajectory.device, dtype=trajectory.dtype
                            )
                            alpha_prod_t = alphas_cumprod[int(t.item())]
                            alpha_prod_t_prev = alphas_cumprod[int(t.item()) - 1]
                            beta_t = 1.0 - (alpha_prod_t / alpha_prod_t_prev)
                            variance = beta_t * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
                            noise = torch.randn_like(trajectory, generator=generator)
                            trajectory = mean + torch.sqrt(variance.clamp_min(1e-20)) * float(sampling_noise_scale) * noise
                        else:
                            trajectory = mean

        # 最后再覆盖一次条件，确保严格满足条件输入
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        init_trajectory: Optional[torch.Tensor] = None,
        deterministic_sampling: bool = False,
        sampling_noise_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """根据观测历史预测动作序列，并切出可执行窗口."""
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        batch_size, _, obs_dim = nobs.shape
        assert obs_dim == self.obs_dim

        horizon = self.horizon
        action_dim = self.action_dim
        obs_steps = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros((batch_size, horizon, obs_dim), device=device, dtype=dtype)
            local_cond[:, :obs_steps] = nobs[:, :obs_steps]
            cond_data = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :obs_steps, :].reshape(batch_size, -1)
            shape = (batch_size, horizon, action_dim)
            if self.pred_action_steps_only:
                shape = (batch_size, self.n_action_steps, action_dim)
            cond_data = torch.zeros(shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros((batch_size, horizon, action_dim + obs_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :obs_steps, action_dim:] = nobs[:, :obs_steps]
            cond_mask[:, :obs_steps, action_dim:] = True

        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            init_trajectory=init_trajectory,
            deterministic_sampling=deterministic_sampling,
            sampling_noise_scale=sampling_noise_scale,
        )

        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
            start = 0
            end = action.shape[1]
        else:
            # receding-horizon: 从预测轨迹中取当前可执行片段
            start = obs_steps
            if self.oa_step_convention:
                start = obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
            "naction_pred": naction_pred,
            "start": start,
            "end": end,
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., action_dim:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            result["obs_pred"] = obs_pred
            result["action_obs_pred"] = obs_pred[:, start:end]
        return result

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """训练损失: 对随机时间步加入噪声并回归目标."""
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        local_cond = None
        global_cond = None
        trajectory = action

        if self.obs_as_local_cond:
            local_cond = obs.clone()
            local_cond[:, self.n_obs_steps :, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)
            if self.pred_action_steps_only:
                start = self.n_obs_steps
                if self.oa_step_convention:
                    start = self.n_obs_steps - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape, device=trajectory.device)

        noise = torch.randn_like(trajectory)
        batch_size = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
            dtype=torch.long,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).type(loss.dtype)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1)
        return loss.mean()
