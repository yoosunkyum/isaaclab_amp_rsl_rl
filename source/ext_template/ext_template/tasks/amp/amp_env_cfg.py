from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ext_template.tasks.amp.mdp as mdp



##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    
    # robots
    robot: ArticulationCfg = MISSING
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""

#     base_velocity = mdp.UniformVelocityCommandCfg(
#         asset_name="robot",
#         resampling_time_range=(10.0, 10.0),
#         rel_standing_envs=0.02,
#         rel_heading_envs=1.0,
#         heading_command=True,
#         heading_control_stiffness=0.5,
#         debug_vis=True,
#         ranges=mdp.UniformVelocityCommandCfg.Ranges(
#             lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
#         ),
#     )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_positions = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_velocities = ObsTerm(func=mdp.joint_vel , noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_velocities = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_velocities = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.05, n_max=0.05))
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})   
        # base_quat = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class AmpCfg(ObsGroup):
        """Observations for amp(discriminator) group."""
        # observation terms (order preserved)
        joint_positions = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_velocities = ObsTerm(func=mdp.joint_vel , noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_velocities = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_velocities = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # root_position = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_pos_z = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # base_quat = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    # # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (-0.5, 0.5),
    #             "roll": (-0.5, 0.5),
    #             "pitch": (-0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #         },
    #     },
    # )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )
    
    
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"z" : (0.0, 0.1), 
                               "roll" : (-3.14/12*0, 3.14/12*0), 
                               "pitch" : (-3.14/12*0, 3.14/12*0), 
                               "yaw" : (-3.14*0, 3.14*0)}, 
                "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_from_motion,
        mode="reset",
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.02, 0.02),
    #         "velocity_range": (-0.01, 0.01),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # -- task
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )
    # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    # alive = RewTerm(func=mdp.is_alive, weight=0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    died = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height" : 0.5})
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class AmpEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 1/(30 * 10)
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15