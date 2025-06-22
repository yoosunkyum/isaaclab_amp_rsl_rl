from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from amp_rsl_rl.utils import MotionLoader
from pathlib import Path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

motion4 = MotionLoader(device="cuda",
                    dataset_path_root=Path("C:/Research/isaaclab_amp_rsl_rl/motions/h1"),
                    dataset_names=["h1_walk"],
                    dataset_weights=[1.0],
                    simulation_dt = 1 /60,
                    slow_down_factor=1,
                    expected_joint_names=['left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_joint', 'right_ankle_joint', 'left_elbow_joint', 'right_elbow_joint']
                    )

def reset_joints_from_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    pos,vel,_,_,_,_ = motion4.sample(1)
    asset.write_joint_state_to_sim(pos, torch.zeros_like(vel), None, env_ids=env_ids)
    # root_state = torch.zeros(7,device="cuda")
    # root_state[2] = 1.5
    # root_state[3] = 1.0
    # asset.write_root_link_pose_to_sim(root_state, None, env_ids = env_ids)
    
