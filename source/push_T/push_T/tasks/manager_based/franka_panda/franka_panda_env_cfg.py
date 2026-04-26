# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda Push-T 任务配置."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG

from . import mdp
from .mdp.actions import FixedDownIKAction

# 项目资产目录
_ASSETS_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))
ASSETS_DIR = os.path.join(_ASSETS_BASE, "assets")


##
# 场景定义
##


@configclass
class FrankaPandaSceneCfg(InteractiveSceneCfg):
    """Franka Panda Push-T 场景配置."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.1894,
                "panda_joint3": -0.1107,
                "panda_joint4": -2.5,
                "panda_joint5": 0.0,
                "panda_joint6": 2.3775,
                "panda_joint7": 0.6952,
            },
        ),
    )


    t_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tblock",
        spawn=UsdFileCfg(
            usd_path=f"{ASSETS_DIR}/block_T.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.3, 0.0, 0.0],
            rot=[1, 0, 0, 0],
        ),
    )

    t_block_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TblockTarget",
        spawn=UsdFileCfg(
            usd_path=f"{ASSETS_DIR}/target.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.45, 0.0, 0.0],
            rot=[0.707, 0, 0, 0.707],
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class FrankaPandaEePosSceneTeleopCfg(InteractiveSceneCfg):
    """Franka Panda Push-T 遥操作场景配置 - 高位姿PD控制."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.1894,
                "panda_joint3": -0.1107,
                "panda_joint4": -2.5,
                "panda_joint5": 0.0,
                "panda_joint6": 2.3775,
                "panda_joint7": 0.6952,
            },
        ),
    )


    t_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tblock",
        spawn=UsdFileCfg(
            usd_path=f"{ASSETS_DIR}/block_T.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.3, 0.0, 0.0],
            rot=[1, 0, 0, 0],
        ),
    )

    t_block_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TblockTarget",
        spawn=UsdFileCfg(
            usd_path=f"{ASSETS_DIR}/target.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.45, 0.0, 0.0],
            rot=[0.707, 0, 0, 0.707],
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP 设置
##


@configclass
class EePosActionsCfg:
    """MDP 动作规格 - 末端位置控制 (2D平面位置，固定姿态与高度).

    使用自定义 FixedDownIKAction 拦截器：
    - 网络只输出 2 维 (x, y)
    - 内部使用绝对位置控制，姿态固定朝下，z 固定为常量
    - use_relative_mode=False 直接使用传入的绝对位置和固定朝下姿态
    """

    arm_action = DifferentialInverseKinematicsActionCfg(
        class_type=FixedDownIKAction,  # 使用自定义拦截器
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,  # 绝对位置控制
            ik_method="dls",
            ik_params={"lambda_val": 0.2},  # 增大阻尼，减少抖动
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0],
        ),
    )


@configclass
class EePosObservationsCfg:
    """MDP 观察规格 - 末端位置控制版本.

    观测空间 (8维):
    - T型块平面位置 (2)
    - T型块旋转四元数 (4)
    - 末端执行器平面位置 (2)

    动作空间 (2维):
    - 末端执行器平面目标 [x, y]
    """

    @configclass
    class PolicyCfg(ObsGroup):
        t_block_position_xy = ObsTerm(
            func=mdp.root_pos_xy_w,
            params={"asset_cfg": SceneEntityCfg("t_block")},
        )
        t_block_orientation = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("t_block")},
        )
        end_effector_xy = ObsTerm(
            func=mdp.end_effector_xy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EePosRewardsCfg:
    """最小奖励配置.

    当前主线是遥操作 + Diffusion 训练，奖励不再用于策略优化，
    因此仅保留成功奖励用于日志观测和任务完成反馈。
    """

    success_bonus = RewTerm(
        func=mdp.is_success,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("t_block"),
            "target_cfg": SceneEntityCfg("t_block_target"),
            # 与 termination.success 阈值保持一致，避免“成功终止但无成功奖励”
            "position_threshold": 0.02,
            "rotation_threshold": 0.07,
        },
    )


@configclass
class EventCfg:
    """事件配置."""

    reset_robot = EventTerm(
        func=mdp.reset_robot_ee_xy_random_fixed_z,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                             "panda_joint5", "panda_joint6", "panda_joint7"],
            ),
            "x_range": (0.15, 0.7),
            "y_range": (-0.4, 0.4),
            "z_fixed": 0.13,
            "action_term_name": "arm_action",
            "ik_iterations": 2,
        },
    )

    randomize_t_block = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("t_block"),
            "pose_range": {
                "x": (0.05, 0.2),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-1.57, 1.57),
            },
            "velocity_range": {},
        },
    )


@configclass
class TerminationsCfg:
    """MDP 终止条件."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.is_success,
        params={
            "object_cfg": SceneEntityCfg("t_block"),
            "target_cfg": SceneEntityCfg("t_block_target"),
            "position_threshold": 0.02,
            "rotation_threshold": 0.07,
        },
    )


##
# 环境配置
##


@configclass
class FrankaPandaEePosEnvCfg(ManagerBasedRLEnvCfg):
    """Franka Panda Push-T RL训练环境配置 - 末端位置控制.

    动作空间: 2D末端执行器平面位置 (x, y)，z 固定
    观测空间: 8维 (t_block_xy(2) + t_block_quat(4) + ee_xy(2))
    使用高位PD控制，与遥操作一致。
    """
    scene: FrankaPandaEePosSceneTeleopCfg = FrankaPandaEePosSceneTeleopCfg(num_envs=4096, env_spacing=4.0)
    observations: EePosObservationsCfg = EePosObservationsCfg()
    actions: EePosActionsCfg = EePosActionsCfg()
    events: EventCfg = EventCfg()
    rewards: EePosRewardsCfg = EePosRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.eye = (1.5, 0.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.3)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class FrankaPandaEePosTeleopEnvCfg(ManagerBasedRLEnvCfg):
    """Franka Panda Push-T 遥操作环境配置 - 末端位置控制."""
    scene: FrankaPandaEePosSceneTeleopCfg = FrankaPandaEePosSceneTeleopCfg(num_envs=1, env_spacing=4.0)
    observations: EePosObservationsCfg = EePosObservationsCfg()
    actions: EePosActionsCfg = EePosActionsCfg()
    events: EventCfg = EventCfg()
    rewards: EePosRewardsCfg = EePosRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 60.0
        self.viewer.eye = (1.5, 0.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.3)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
