from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, axis_angle_from_quat, quat_error_magnitude, quat_inv, quat_mul

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def anchor_ori_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    mat = matrix_from_quat(command.anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def anchor_lin_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    return command.anchor_lin_vel_w.view(env.num_envs, -1)


def anchor_ang_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    return command.anchor_ang_vel_w.view(env.num_envs, -1)


def body_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    num_bodies = len(command.robot.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.body_pos_w,
        command.body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def body_ori_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    num_bodies = len(command.robot.body_names)
    _, ori_b = subtract_frame_transforms(
        command.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.body_pos_w,
        command.body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def body_lin_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    return command.body_lin_vel_w.reshape(env.num_envs, -1)


def body_ang_vel_w(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    return command.body_ang_vel_w.reshape(env.num_envs, -1)




def anchor_pos_ref_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    pos, _ = subtract_frame_transforms(
        command.anchor_pos_w,
        command.anchor_quat_w,
        command.anchor_pos_ref_w,
        command.anchor_quat_ref_w,
    )

    return pos.view(env.num_envs, -1)


def anchor_ori_ref_b(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")

    _, ori = subtract_frame_transforms(
        command.anchor_pos_w,
        command.anchor_quat_w,
        command.anchor_pos_ref_w,
        command.anchor_quat_ref_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

#tracking error observations 
#1)joint error
def err_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.joint_pos_ref - command.joint_pos).view(env.num_envs, -1)

def err_joint_vel(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.joint_vel_ref - command.joint_vel).view(env.num_envs, -1)

#2)anchor error
def err_anchor_pos(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.anchor_pos_ref_w - command.anchor_pos_w).view(env.num_envs, -1)

def err_anchor_rot(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return axis_angle_from_quat(quat_mul(command.anchor_quat_ref_w, quat_inv(command.anchor_quat_w))).view(env.num_envs, -1)
    # return quat_error_magnitude(command.anchor_quat_ref_w, command.anchor_quat_w) #TODO : verify which is better

def err_anchor_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.anchor_lin_vel_ref_w - command.anchor_lin_vel_w).view(env.num_envs, -1)

def err_anchor_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.anchor_ang_vel_ref_w - command.anchor_ang_vel_w).view(env.num_envs, -1)

#3)body error
def err_body_pos(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.body_pos_relative_w - command.body_pos_w).view(env.num_envs, -1)

def err_body_rot(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return axis_angle_from_quat(quat_mul(command.body_quat_relative_w, quat_inv(command.body_quat_w))).view(env.num_envs, -1)
    # return quat_error_magnitude(command.body_quat_relative_w, command.body_quat_w) #TODO : verify which is better

def err_body_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.body_lin_vel_ref_w - command.body_lin_vel_w).view(env.num_envs, -1)

def err_body_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term("motion")
    return (command.body_ang_vel_ref_w - command.body_ang_vel_w).view(env.num_envs, -1)