# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda Push-T 任务配置."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp

# 项目资产目录
_ASSETS_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))
ASSETS_DIR = os.path.join(_ASSETS_BASE, "assets")

##
# 场景定义
##


@configclass
class FrankaPandaSceneCfg(InteractiveSceneCfg):
    """Franka Panda Push-T 场景配置."""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Franka Panda 机器人
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.1],
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

    # 推动任务用的桌子
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.5, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.3, 0.0, 0.05],
            rot=[1, 0, 0, 0],
        ),
    )

    # T形方块 
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
            pos=[0.35, 0.0, 0.1],
            rot=[1, 0, 0, 0],
        ),
    )

    # T形方块目标位置标记
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
            pos=[0.55, 0.0, 0.1],
            rot=[0.707, 0, 0, 0.707],
        ),
    )

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP 设置
##


@configclass
class ActionsCfg:
    """MDP 动作规格."""

    # 使用关节位置控制
    # scale=3.0，允许更大的动作幅度
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                     "panda_joint5", "panda_joint6", "panda_joint7"],
        scale=3.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """MDP 观察规格."""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观察组."""

        # 机器人关节状态
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # T型块状态
        t_block_position = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("t_block")},
        )
        t_block_orientation = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("t_block")},
        )

        # 末端执行器
        end_effector_pos = ObsTerm(
            func=mdp.end_effector_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )

        # 目标位置和姿态 (固定值)
        # 目标位置: (0.6, 0.0, 0.1), 目标旋转: 绕Z轴-90°

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """事件配置."""

    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                             "panda_joint5", "panda_joint6", "panda_joint7"],
            ),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # 随机化T型块位置
    randomize_t_block = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("t_block"),
            "pose_range": {
                "x": (0.35, 0.45),
                "y": (-0.1, 0.1),
                "z": (0.1, 0.1),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-1.57, 1.57),
            },
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """MDP 奖励项."""


    # 阶段1：让机械臂末端靠近T型块
    reaching_object = RewTerm(
        func=mdp.object_fingers_distance,
        weight=-2.0,  
        params={
            "object_cfg": SceneEntityCfg("t_block"),
            "fingers_cfg": SceneEntityCfg("robot", body_names=["panda_leftfinger", "panda_rightfinger"]),
        },
    )

    # 阶段2：将T型块推向目标位置
    object_to_target_pos = RewTerm(
        func=mdp.object_target_distance,
        weight=-20.0,
        params={
            "object_cfg": SceneEntityCfg("t_block"),
            "target_cfg": SceneEntityCfg("t_block_target"),
        },
    )

    # 阶段3：对齐T型块的角度 (大幅降低权重，先专注位置)
    object_to_target_rot = RewTerm(
        func=mdp.object_target_rotation_diff,
        weight=-1.0,  # 从-8.0降到-1.0，课程学习
        params={
            "object_cfg": SceneEntityCfg("t_block"),
            "target_cfg": SceneEntityCfg("t_block_target"),
        },
    )

    # 成功奖励
    success_bonus = RewTerm(
        func=mdp.is_success,
        weight=10000.0,  # 从5000提升到10000，让成功梯度如核弹
        params={
            "object_cfg": SceneEntityCfg("t_block"),
            "target_cfg": SceneEntityCfg("t_block_target"),
            "position_threshold": 0.12,
            "rotation_threshold": 0.50,
        },
    )

    # === 动作平滑惩罚 ===
    # 极低惩罚，允许推物体时有动作爆发力
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,  # 从-0.05降到-0.01
    )

    # 惩罚关节速度 (防止瞬移) - 降低惩罚允许更大动作
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,  # 从-0.01降到-0.001
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """MDP 终止条件."""

    # 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # T型块掉落桌面 (允许低一点)
    object_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("t_block"),
            "minimum_height": 0.03,
        },
    )

    # 成功终止：T型块到达目标位置且角度对齐 (放宽阈值)
    success = DoneTerm(
        func=mdp.is_success,
        params={
            "object_cfg": SceneEntityCfg("t_block"),
            "target_cfg": SceneEntityCfg("t_block_target"),
            "position_threshold": 0.12,  # 从0.08放宽到0.12
            "rotation_threshold": 0.50,  # 从0.35放宽到0.50 (rad)
        },
    )


##
# 环境配置
##


@configclass
class FrankaPandaEnvCfg(ManagerBasedRLEnvCfg):
    # 场景设置
    scene: FrankaPandaSceneCfg = FrankaPandaSceneCfg(num_envs=4096, env_spacing=4.0)
    # 基础设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP 设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 后处理初始化
    def __post_init__(self) -> None:
        """后处理初始化."""
        # 通用设置
        self.decimation = 2
        self.episode_length_s = 20.0  # 从15秒延长到20秒，给更多推行时间
        # 显示器设置
        self.viewer.eye = (1.5, 0.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.3)
        # 仿真设置
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation