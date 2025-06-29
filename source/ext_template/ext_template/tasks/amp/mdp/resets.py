from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from amp_rsl_rl.utils import MotionLoader
from pathlib import Path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# motion = MotionLoader(device="cuda",
#                     dataset_path_root=Path("C:/Research/isaaclab_amp_rsl_rl/motions/h1"),
#                     dataset_names=["h1_walk"],
#                     dataset_weights=[1.0],
#                     simulation_dt = 1 /60,
#                     slow_down_factor=1,
#                     expected_joint_names=['left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_joint', 'right_ankle_joint', 'left_elbow_joint', 'right_elbow_joint']
#                     )

motion = MotionLoader(device="cuda",
                    dataset_path_root=Path("C:/Research/isaaclab_amp_rsl_rl/motions/g1"),
                    dataset_names=["g1_walk"],
                    dataset_weights=[1.0],
                    simulation_dt = 1 /60,
                    slow_down_factor=1,
                    expected_joint_names=['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
                    )

def reset_joints_from_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos,joint_vel,base_pos,base_quat,base_lin_vel,base_ang_vel = motion.sample(env_ids.shape[0])
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:,0:3] = env.scene.env_origins[env_ids]
    root_state[:,2] = base_pos[:,0,2] + 0.1
    root_state[:,3:7] = base_quat[:,0,:]
    root_state[:,7:10] = base_lin_vel[:,0,:]
    root_state[:,10:13] = base_ang_vel[:,0,:]

    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids=env_ids)
    asset.write_root_link_pose_to_sim(root_state[:,0:7],  env_ids = env_ids)
    asset.write_root_com_velocity_to_sim(root_state[:,7:13], env_ids= env_ids)
    
