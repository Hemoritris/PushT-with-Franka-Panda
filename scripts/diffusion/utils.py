"""Diffusion 脚本公共工具.

主要提供:
- 单字段线性归一化器
- 多字段归一化容器（obs/action）
"""

from typing import Dict

import numpy as np
import torch


class SingleFieldLinearNormalizer:
    """单字段归一化器（按特征维做仿射变换）."""

    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, input_stats: Dict[str, torch.Tensor]):
        self.scale = scale
        self.offset = offset
        self.input_stats = input_stats

    @staticmethod
    def fit(
        data: torch.Tensor | np.ndarray,
        mode: str = "limits",
        output_min: float = -1.0,
        output_max: float = 1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ) -> "SingleFieldLinearNormalizer":
        """根据数据统计量拟合 scale/offset."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.float().reshape(-1, data.shape[-1])
        input_min, _ = data.min(dim=0)
        input_max, _ = data.max(dim=0)
        input_mean = data.mean(dim=0)
        input_std = data.std(dim=0)

        if mode == "limits":
            if fit_offset:
                input_range = input_max - input_min
                ignore_dim = input_range < range_eps
                input_range = input_range.clone()
                input_range[ignore_dim] = output_max - output_min
                scale = (output_max - output_min) / input_range
                offset = output_min - scale * input_min
                offset = offset.clone()
                offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            else:
                output_abs = min(abs(output_min), abs(output_max))
                input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
                ignore_dim = input_abs < range_eps
                input_abs = input_abs.clone()
                input_abs[ignore_dim] = output_abs
                scale = output_abs / input_abs
                offset = torch.zeros_like(input_mean)
        elif mode == "gaussian":
            safe_std = input_std.clone()
            safe_std[safe_std < range_eps] = 1.0
            scale = 1.0 / safe_std
            offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        input_stats = {
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        }
        return SingleFieldLinearNormalizer(scale=scale, offset=offset, input_stats=input_stats)

    def normalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(device=self.scale.device, dtype=self.scale.dtype)
        src_shape = x.shape
        x = x.reshape(-1, self.scale.shape[0])
        x = x * self.scale + self.offset
        return x.reshape(src_shape)

    def unnormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(device=self.scale.device, dtype=self.scale.dtype)
        src_shape = x.shape
        x = x.reshape(-1, self.scale.shape[0])
        x = (x - self.offset) / self.scale
        return x.reshape(src_shape)

    def to(self, device: torch.device | str) -> "SingleFieldLinearNormalizer":
        self.scale = self.scale.to(device)
        self.offset = self.offset.to(device)
        self.input_stats = {k: v.to(device) for k, v in self.input_stats.items()}
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "scale": self.scale.detach().cpu(),
            "offset": self.offset.detach().cpu(),
            "input_stats": {k: v.detach().cpu() for k, v in self.input_stats.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "SingleFieldLinearNormalizer":
        return cls(scale=state["scale"], offset=state["offset"], input_stats=state["input_stats"])


class LinearNormalizer:
    """多字段归一化器容器（例如 obs 与 action）."""

    def __init__(self):
        self.fields: Dict[str, SingleFieldLinearNormalizer] = {}

    def fit(self, data: Dict[str, torch.Tensor | np.ndarray], mode: str = "limits") -> "LinearNormalizer":
        for key, value in data.items():
            self.fields[key] = SingleFieldLinearNormalizer.fit(value, mode=mode)
        return self

    def normalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.fields[k].normalize(v) for k, v in data.items()}

    def unnormalize(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: self.fields[k].unnormalize(v) for k, v in data.items()}

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        return self.fields[key]

    def to(self, device: torch.device | str) -> "LinearNormalizer":
        for value in self.fields.values():
            value.to(device)
        return self

    def state_dict(self) -> Dict:
        return {k: v.state_dict() for k, v in self.fields.items()}

    @classmethod
    def from_state_dict(cls, state: Dict) -> "LinearNormalizer":
        obj = cls()
        obj.fields = {k: SingleFieldLinearNormalizer.from_state_dict(v) for k, v in state.items()}
        return obj
