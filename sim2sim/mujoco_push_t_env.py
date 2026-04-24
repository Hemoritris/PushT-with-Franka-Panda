from __future__ import annotations

"""MuJoCo 版 Push-T 低维环境封装。

设计目标：
- 复刻 Isaac Lab 中的低维任务接口；
- 观测保持为 8 维：`t_block_xy + t_block_quat + ee_xy`；
- 动作保持为 2 维：`panda_hand` 的目标平面位置 `[x, y]`；
- 末端 `panda_hand` 高度固定为 `0.13m`，夹爪保持闭合。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    """将四元数归一化，避免数值误差放大。"""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """计算四元数共轭。"""
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """计算四元数乘法。"""
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """将四元数误差转换为旋转向量，便于 IK 使用。"""
    quat = _quat_normalize(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:]
    sin_half = np.linalg.norm(vector)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quat[0], -1.0, 1.0))
    axis = vector / sin_half
    return axis * angle


def _quat_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """返回两个姿态之间的旋转差值。"""
    relative = _quat_multiply(lhs, _quat_conjugate(rhs))
    return float(np.linalg.norm(_quat_to_rotvec(relative)))


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """仅绕 z 轴旋转时的四元数表达。"""
    half = yaw * 0.5
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _lookup_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    """通过名字查询 MuJoCo 对象 id，查不到时直接报错。"""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Failed to find {obj_type.name} named {name!r} in model.")
    return obj_id


@dataclass(frozen=True)
class PushTTaskConfig:
    """Push-T 任务的关键超参数与阈值。"""
    init_arm_qpos: tuple[float, ...] = (0.0, -0.1894, -0.1107, -2.5, 0.0, 2.3775, 0.6952)
    finger_closed_qpos: float = 0.0
    gripper_ctrl: float = 0.0
    action_x_range: tuple[float, float] = (0.15, 0.70)
    action_y_range: tuple[float, float] = (-0.40, 0.40)
    hand_target_z: float = 0.13
    block_default_pos: tuple[float, float, float] = (0.30, 0.0, 0.0)
    target_default_pos: tuple[float, float, float] = (0.45, 0.0, 0.0)
    target_default_quat: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678)
    block_reset_x_range: tuple[float, float] = (0.05, 0.20)
    block_reset_y_range: tuple[float, float] = (-0.10, 0.10)
    block_reset_yaw_range: tuple[float, float] = (-1.57, 1.57)
    success_position_threshold: float = 0.02
    success_rotation_threshold: float = 0.07
    frame_skip: int = 2
    max_steps: int = 1200
    ik_damping: float = 0.10
    ik_step_size: float = 0.75
    ik_orientation_weight: float = 0.60
    ik_reset_iterations: int = 100
    ik_control_iterations: int = 30


class PandaHandIKController:
    """基于 `panda_hand` 的简易 IK 控制器。

    输入为 `panda_hand` 的目标位置和目标姿态，
    输出为 Panda 7 个关节的目标角。
    """

    def __init__(self, model: mujoco.MjModel, task_cfg: PushTTaskConfig):
        self.model = model
        self.task_cfg = task_cfg
        self.arm_joint_names = [f"joint{index}" for index in range(1, 8)]
        self.arm_actuator_names = [f"actuator{index}" for index in range(1, 8)]
        self.finger_joint_names = ["finger_joint1", "finger_joint2"]
        self.hand_body_id = _lookup_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.arm_joint_ids = np.array(
            [_lookup_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joint_names], dtype=np.int32
        )
        self.arm_actuator_ids = np.array(
            [_lookup_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.arm_actuator_names], dtype=np.int32
        )
        self.finger_joint_ids = np.array(
            [_lookup_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.finger_joint_names], dtype=np.int32
        )
        self.gripper_actuator_id = _lookup_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.arm_qpos_adr = np.array([model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids], dtype=np.int32)
        self.arm_dof_adr = np.array([model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids], dtype=np.int32)
        self.finger_qpos_adr = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in self.finger_joint_ids], dtype=np.int32
        )
        self.arm_joint_limits = model.jnt_range[self.arm_joint_ids].astype(np.float64)
        self.scratch = mujoco.MjData(model)
        self.fixed_hand_quat = self._calibrate_fixed_orientation()

    def _calibrate_fixed_orientation(self) -> np.ndarray:
        """读取初始姿态下 `panda_hand` 的朝向，作为固定末端姿态。"""
        self.scratch.qpos[:] = self.model.qpos0
        self.scratch.qvel[:] = 0
        self.scratch.qpos[self.arm_qpos_adr] = np.asarray(self.task_cfg.init_arm_qpos, dtype=np.float64)
        self.scratch.qpos[self.finger_qpos_adr] = self.task_cfg.finger_closed_qpos
        mujoco.mj_forward(self.model, self.scratch)
        return _quat_normalize(self.scratch.xquat[self.hand_body_id].copy())

    def set_ctrl(self, data: mujoco.MjData, arm_qpos: np.ndarray) -> None:
        """将 IK 解出的关节目标写入 MuJoCo 控制输入。"""
        data.ctrl[self.arm_actuator_ids] = np.asarray(arm_qpos, dtype=np.float64)
        data.ctrl[self.gripper_actuator_id] = self.task_cfg.gripper_ctrl

    def solve(
        self,
        full_qpos: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        iterations: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """迭代求解使 `panda_hand` 接近目标位姿的关节角。

        返回最优关节角，以及对应的位置 / 姿态误差。
        """
        self.scratch.qpos[:] = full_qpos
        self.scratch.qvel[:] = 0
        mujoco.mj_forward(self.model, self.scratch)

        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = _quat_normalize(np.asarray(target_quat, dtype=np.float64))
        best_arm_qpos = self.scratch.qpos[self.arm_qpos_adr].copy()
        best_pos_error = float("inf")
        best_rot_error = float("inf")

        # 使用 body Jacobian 对 `panda_hand` 进行位置和姿态控制。
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        damping_identity = np.eye(6, dtype=np.float64)

        for _ in range(max(1, iterations)):
            current_pos = self.scratch.xpos[self.hand_body_id].copy()
            current_quat = _quat_normalize(self.scratch.xquat[self.hand_body_id].copy())
            pos_error = target_pos - current_pos
            quat_error = _quat_multiply(target_quat, _quat_conjugate(current_quat))
            rot_error = _quat_to_rotvec(quat_error)

            pos_error_norm = float(np.linalg.norm(pos_error))
            rot_error_norm = float(np.linalg.norm(rot_error))
            if pos_error_norm + rot_error_norm < best_pos_error + best_rot_error:
                best_arm_qpos = self.scratch.qpos[self.arm_qpos_adr].copy()
                best_pos_error = pos_error_norm
                best_rot_error = rot_error_norm
            if pos_error_norm < 1e-4 and rot_error_norm < 1e-3:
                break

            mujoco.mj_jacBody(self.model, self.scratch, jac_pos, jac_rot, self.hand_body_id)
            reduced_jacobian = np.vstack(
                [
                    jac_pos[:, self.arm_dof_adr],
                    self.task_cfg.ik_orientation_weight * jac_rot[:, self.arm_dof_adr],
                ]
            )
            task_error = np.concatenate([pos_error, self.task_cfg.ik_orientation_weight * rot_error], dtype=np.float64)
            system = reduced_jacobian @ reduced_jacobian.T + (self.task_cfg.ik_damping**2) * damping_identity
            delta_q = reduced_jacobian.T @ np.linalg.solve(system, task_error)

            # NumPy 高级索引返回副本，因此需要显式写回 scratch.qpos。
            arm_qpos = self.scratch.qpos[self.arm_qpos_adr].copy()
            arm_qpos += self.task_cfg.ik_step_size * delta_q
            arm_qpos[:] = np.clip(arm_qpos, self.arm_joint_limits[:, 0], self.arm_joint_limits[:, 1])
            self.scratch.qpos[self.arm_qpos_adr] = arm_qpos
            mujoco.mj_forward(self.model, self.scratch)

        return best_arm_qpos, {"ik_pos_error": best_pos_error, "ik_rot_error": best_rot_error}


class MujocoPushTEnv:
    """MuJoCo 版 Push-T 环境。

    接口风格尽量贴近 Gym：
    - `reset()` 返回 `(observation, info)`
    - `step(action)` 返回 `(observation, reward, terminated, truncated, info)`
    """

    def __init__(
        self,
        model_path: str | Path,
        frame_skip: int | None = None,
        max_steps: int | None = None,
        seed: int = 42,
        task_cfg: PushTTaskConfig | None = None,
    ):
        self.task_cfg = task_cfg or PushTTaskConfig()

        # 优先使用仓库内相对路径；若用户传入绝对路径且位于当前仓库内，
        # 则尽量转换为相对路径，提升项目可移植性，并绕开部分平台下
        # MuJoCo 对 Unicode 绝对路径的加载问题。
        raw_model_path = Path(model_path)
        load_model_path = raw_model_path
        if raw_model_path.anchor:
            try:
                load_model_path = Path(os.path.relpath(raw_model_path, start="."))
            except ValueError:
                load_model_path = raw_model_path
        self.model_path = load_model_path
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip or self.task_cfg.frame_skip
        self.max_steps = max_steps or self.task_cfg.max_steps
        self.rng = np.random.default_rng(seed)
        self.controller = PandaHandIKController(self.model, self.task_cfg)
        self.hand_body_id = self.controller.hand_body_id
        self.block_body_id = _lookup_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "t_block")
        self.target_body_id = _lookup_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "t_block_target")
        self.block_joint_id = _lookup_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "t_block_freejoint")
        self.block_qpos_adr = int(self.model.jnt_qposadr[self.block_joint_id])
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float64)

    @property
    def control_dt(self) -> float:
        """控制层实际步长 = MuJoCo timestep × frame_skip。"""
        return float(self.model.opt.timestep * self.frame_skip)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境。

        行为包括：
        - 恢复 Panda 初始关节；
        - 保持夹爪闭合；
        - 随机化 T 块位置和朝向；
        - 随机初始化 `panda_hand` 的平面位置。
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        if self.model.na:
            self.data.act[:] = 0

        self.data.qpos[self.controller.arm_qpos_adr] = np.asarray(self.task_cfg.init_arm_qpos, dtype=np.float64)
        self.data.qpos[self.controller.finger_qpos_adr] = self.task_cfg.finger_closed_qpos

        # T 块随机化范围与 Isaac Lab 配置保持一致。
        block_pos = np.array(self.task_cfg.block_default_pos, dtype=np.float64)
        block_pos[0] += self.rng.uniform(*self.task_cfg.block_reset_x_range)
        block_pos[1] += self.rng.uniform(*self.task_cfg.block_reset_y_range)
        block_yaw = self.rng.uniform(*self.task_cfg.block_reset_yaw_range)
        self.data.qpos[self.block_qpos_adr : self.block_qpos_adr + 3] = block_pos
        self.data.qpos[self.block_qpos_adr + 3 : self.block_qpos_adr + 7] = _yaw_to_quat(block_yaw)

        # 将手部随机放到工作空间内的合法位置。
        reset_action = np.array(
            [
                self.rng.uniform(*self.task_cfg.action_x_range),
                self.rng.uniform(*self.task_cfg.action_y_range),
            ],
            dtype=np.float64,
        )
        target_pos = np.array([reset_action[0], reset_action[1], self.task_cfg.hand_target_z], dtype=np.float64)
        arm_qpos, ik_info = self.controller.solve(
            self.data.qpos.copy(),
            target_pos=target_pos,
            target_quat=self.controller.fixed_hand_quat,
            iterations=self.task_cfg.ik_reset_iterations,
        )

        self.data.qpos[self.controller.arm_qpos_adr] = arm_qpos
        self.data.qvel[:] = 0
        self.controller.set_ctrl(self.data, arm_qpos)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_action = reset_action.copy()
        observation = self.get_observation()
        success, metrics = self.evaluate_success()
        info: dict[str, Any] = {"success": success, "reset_action": reset_action.astype(np.float32)}
        info.update(metrics)
        info.update(ik_info)
        return observation, info

    def get_observation(self) -> np.ndarray:
        """提取 8 维低维观测。"""
        block_pos = self.data.xpos[self.block_body_id]
        block_quat = _quat_normalize(self.data.xquat[self.block_body_id])
        hand_pos = self.data.xpos[self.hand_body_id]
        return np.concatenate([block_pos[:2], block_quat, hand_pos[:2]], dtype=np.float64).astype(np.float32)

    def evaluate_success(self) -> tuple[bool, dict[str, float]]:
        """根据位置和姿态阈值判断当前局是否成功。"""
        block_pos = self.data.xpos[self.block_body_id]
        target_pos = self.data.xpos[self.target_body_id]
        block_quat = _quat_normalize(self.data.xquat[self.block_body_id])
        target_quat = _quat_normalize(self.data.xquat[self.target_body_id])
        position_distance = float(np.linalg.norm(block_pos - target_pos))
        rotation_distance = _quat_distance(block_quat, target_quat)
        success = (
            position_distance < self.task_cfg.success_position_threshold
            and rotation_distance < self.task_cfg.success_rotation_threshold
        )
        return success, {"position_distance": position_distance, "rotation_distance": rotation_distance}

    def step(self, action_xy: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步环境推进。

        输入动作是训练语义下的物理坐标 `[ee_x, ee_y]`，
        内部会自动固定 `panda_hand.z = 0.13` 并保持姿态不变。
        """
        action = np.asarray(action_xy, dtype=np.float64).reshape(-1)
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}.")

        # 对动作做工作空间裁剪，避免 IK 收到无效目标。
        clamped_action = action.copy()
        clamped_action[0] = np.clip(clamped_action[0], *self.task_cfg.action_x_range)
        clamped_action[1] = np.clip(clamped_action[1], *self.task_cfg.action_y_range)
        target_pos = np.array([clamped_action[0], clamped_action[1], self.task_cfg.hand_target_z], dtype=np.float64)

        arm_qpos, ik_info = self.controller.solve(
            self.data.qpos.copy(),
            target_pos=target_pos,
            target_quat=self.controller.fixed_hand_quat,
            iterations=self.task_cfg.ik_control_iterations,
        )
        self.controller.set_ctrl(self.data, arm_qpos)

        # 每个控制步推进若干个仿真子步。
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.last_action = clamped_action
        observation = self.get_observation()
        success, metrics = self.evaluate_success()
        terminated = success
        truncated = self.step_count >= self.max_steps
        info: dict[str, Any] = {
            "success": success,
            "step": self.step_count,
            "action": clamped_action.astype(np.float32),
            "time": float(self.data.time),
        }
        info.update(metrics)
        info.update(ik_info)
        return observation, float(success), terminated, truncated, info

    def close(self) -> None:
        """占位接口，便于与上层脚本保持统一调用风格。"""
        return None
