from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_joints_from_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,

):
    command: MotionCommand = env.command_manager.get_term("motion")
    command._resample_command(env_ids)
    # joint_pos, joint_vel, base_pos, base_quat, base_lin_vel, base_ang_vel = command.motion.sample(env_ids.shape[0])
    # root_state = asset.data.default_root_state[env_ids].clone()
    # root_state[:,0:3] = env.scene.env_origins[env_ids]
    # root_state[:,2] = base_pos[:,0,2] + 0.02
    # root_state[:,3:7] = base_quat[:,0,:]
    # root_state[:,7:10] = base_lin_vel[:,0,:]
    # root_state[:,10:13] = base_ang_vel[:,0,:]

    # asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids=env_ids)
    # asset.write_root_link_pose_to_sim(root_state[:,0:7],  env_ids = env_ids)
    # asset.write_root_com_velocity_to_sim(root_state[:,7:13], env_ids= env_ids)
    
