"""Diffusion 低维数据集定义.

作用:
- 从扁平化 npz 演示数据构建序列窗口
- 输出与官方 diffusion-policy 接口一致的 (obs, action) 样本
"""

import copy
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import LinearNormalizer


def _resolve_demo_file(path: str) -> str:
    """解析 demo_path.

    支持两种输入:
    1) 直接传入 .npz 文件
    2) 传入目录，则自动选择 demo 文件
       - 优先使用固定文件名 `push_t_demos.npz`
       - 否则回退到字典序最后一个 .npz
    """
    if os.path.isfile(path):
        return path
    preferred = os.path.join(path, "push_t_demos.npz")
    if os.path.isfile(preferred):
        return preferred
    demo_files = [f for f in os.listdir(path) if f.endswith(".npz")]
    if not demo_files:
        raise FileNotFoundError(f"No .npz demonstration file found in: {path}")
    latest = sorted(demo_files)[-1]
    return os.path.join(path, latest)


def _get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def _create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    pad_before = min(max(int(pad_before), 0), sequence_length - 1)
    pad_after = min(max(int(pad_after), 0), sequence_length - 1)
    indices = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        end_idx = int(episode_ends[i])
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    if len(indices) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


class SequenceSampler:
    """按照 horizon 从多条轨迹中采样序列窗口."""
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        episode_mask: Optional[np.ndarray] = None,
    ):
        self.arrays = arrays
        self.sequence_length = int(sequence_length)
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        self.indices = _create_indices(
            episode_ends=episode_ends,
            sequence_length=self.sequence_length,
            episode_mask=episode_mask,
            pad_before=pad_before,
            pad_after=pad_after,
        )

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result: Dict[str, np.ndarray] = {}
        for key, array in self.arrays.items():
            sample = array[buffer_start_idx:buffer_end_idx]
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros((self.sequence_length,) + array.shape[1:], dtype=array.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            else:
                data = sample
            result[key] = data
        return result


class PushTLowdimDataset(Dataset):
    """Push-T 低维数据集."""

    X_RANGE = (0.15, 0.7)
    Y_RANGE = (-0.4, 0.4)

    def __init__(
        self,
        demo_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        seed: int = 42,
        val_ratio: float = 0.1,
        is_validation: bool = False,
    ):
        super().__init__()
        # 读取演示
        demo_file = _resolve_demo_file(demo_path)
        data = np.load(demo_file, allow_pickle=True)
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        traj_lengths = data["traj_lengths"].astype(np.int64)
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions length mismatch in demonstrations.")

        # 统一观测/动作语义到当前任务定义
        self.states = self._build_obs_features(states)
        self.actions = self._build_action_features(actions)
        self.traj_lengths = traj_lengths
        self.episode_ends = np.cumsum(traj_lengths)
        n_episodes = len(traj_lengths)
        val_mask = _get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        self.train_mask = train_mask
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        # 与官方窗口定义一致:
        # 前端补 n_obs_steps-1, 后端补 n_action_steps-1
        self.pad_before = self.n_obs_steps - 1
        self.pad_after = self.n_action_steps - 1

        episode_mask = val_mask if is_validation else train_mask
        arrays = {"obs": self.states, "action": self.actions}
        self.sampler = SequenceSampler(
            arrays=arrays,
            episode_ends=self.episode_ends,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
        )
        self.is_validation = is_validation

    @staticmethod
    def _build_obs_features(states: np.ndarray) -> np.ndarray:
        """构建 diffusion 观测: [t_block_xy(2), t_block_quat(4), ee_xy(2)]."""
        obs_dim = states.shape[1]
        if obs_dim != 8:
            raise ValueError(
                f"Unsupported states dim={obs_dim}. Expected exactly 8 "
                f"[t_block_xy(2), t_block_quat(4), ee_xy(2)]."
            )
        return states.astype(np.float32)

    @staticmethod
    def _build_action_features(actions: np.ndarray) -> np.ndarray:
        """构建 diffusion 动作: ee_xy 物理坐标（米）."""
        action_dim = actions.shape[1]
        if action_dim != 2:
            raise ValueError(
                f"Unsupported actions dim={action_dim}. Expected exactly 2 "
                f"[ee_x, ee_y] in meters."
            )
        out = actions.astype(np.float32)

        # New pipeline expects physical x/y coordinates directly.
        # If data is still normalized [-1,1], fail fast to avoid silently training wrong semantics.
        x_min, x_max = PushTLowdimDataset.X_RANGE
        y_min, y_max = PushTLowdimDataset.Y_RANGE
        margin = 0.10
        x_ok = (out[:, 0].min() >= (x_min - margin)) and (out[:, 0].max() <= (x_max + margin))
        y_ok = (out[:, 1].min() >= (y_min - margin)) and (out[:, 1].max() <= (y_max + margin))
        if not (x_ok and y_ok):
            raise ValueError(
                "Detected action values outside expected physical workspace range. "
                "This training pipeline expects action=[ee_x, ee_y] in meters. "
                "Please re-collect demonstrations with the updated teleop script."
            )

        return out

    def get_validation_dataset(self) -> "PushTLowdimDataset":
        """构造共享底层数组的验证集视图."""
        val_set = copy.copy(self)
        arrays = {"obs": self.states, "action": self.actions}
        val_set.sampler = SequenceSampler(
            arrays=arrays,
            episode_ends=self.episode_ends,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.is_validation = True
        return val_set

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        """拟合 obs/action 归一化器."""
        normalizer = LinearNormalizer()
        normalizer.fit({"obs": self.states, "action": self.actions}, mode=mode)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        return {
            "obs": torch.from_numpy(sample["obs"]),
            "action": torch.from_numpy(sample["action"]),
        }
