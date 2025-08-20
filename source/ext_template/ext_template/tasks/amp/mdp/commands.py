from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING
from amp_rsl_rl.utils import MotionLoader
from pathlib import Path
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        super().__init__(cfg, env)
        self.motion = MotionLoader(dataset_path_root=cfg.dataset_path_root,
                                   dataset_names=cfg.dataset_names,
                                   dataset_weights=cfg.dataset_weights,                              
                                   simulation_dt=env.step_dt,
                                   slow_down_factor=cfg.slow_down_factor,
                                   expected_joint_names=self.robot.joint_names,
                                   device=self.device)
    
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.motion.get_body_index([self.cfg.anchor_body])[0]
        
        self.motion_joint_indexes = self.motion.get_dof_index(self.robot.joint_names)
        self.motion_body_indexes = self.motion.get_body_index(self.robot.body_names)
        # print("motion_joint_indexes : ", self.motion_joint_indexes)
        # print("motion_body_indexes : ", self.motion_body_indexes)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(self.robot.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(self.robot.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos_ref, self.joint_vel], dim=1)   

    @property
    def joint_pos_ref(self) -> torch.Tensor:
        return self.motion.dof_positions[self.time_steps][:, self.motion_joint_indexes]  #TODO: check joint order
    
    @property
    def joint_vel_ref(self) -> torch.Tensor:
        return self.motion.dof_velocities[self.time_steps][:, self.motion_joint_indexes]  #TODO: check joint order
    
    @property
    def body_pos_ref_w(self) -> torch.Tensor:
       
        return self.motion.body_positions[self.time_steps][:, self.motion_body_indexes,:] + self._env.scene.env_origins[:, None, :]  #TODO: check body order

    @property
    def body_quat_ref_w(self) -> torch.Tensor:
        return self.motion.body_rotations[self.time_steps][:, self.motion_body_indexes,:]  #TODO: check body order
    
    @property
    def body_lin_vel_ref_w(self) -> torch.Tensor:
        return self.motion.body_linear_velocities[self.time_steps][:, self.motion_body_indexes,:]   #TODO: check body order
    
    @property
    def body_ang_vel_ref_w(self) -> torch.Tensor:
        return self.motion.body_angular_velocities[self.time_steps][:, self.motion_body_indexes,:]   #TODO: check body order
    
    @property
    def anchor_pos_ref_w(self) -> torch.Tensor:
        return self.motion.body_positions[self.time_steps][:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_ref_w(self) -> torch.Tensor:
        return self.motion.body_rotations[self.time_steps][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_ref_w(self) -> torch.Tensor:
        return self.motion.body_linear_velocities[self.time_steps][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_ref_w(self) -> torch.Tensor:
        return self.motion.body_angular_velocities[self.time_steps][:, self.motion_anchor_body_index]
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos
    
    @property
    def joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel
    
    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w
    
    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w
    
    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w
    
    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):

        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_ref_w - self.anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_ref_w, self.anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_ref_w - self.anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_ref_w - self.anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.body_quat_w).mean(
            dim=-1
        )
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_ref_w - self.body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_ref_w - self.body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos_ref - self.joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel_ref - self.joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = (phase * (self.motion.num_frames - 1)).long()

        root_pos = self.body_pos_ref_w[:, 0].clone()
        root_ori = self.body_quat_ref_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_ref_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_ref_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos_ref.clone()
        joint_vel = self.joint_vel_ref.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.num_frames)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_ref_w[:, None, :].repeat(1, len(self.robot.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_ref_w[:, None, :].repeat(1, len(self.robot.body_names), 1)
        robot_anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.robot.body_names), 1)
        robot_anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.robot.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_ref_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_ref_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.robot.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.robot.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.robot.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        for i in range(len(self.robot.body_names)):
            self.current_body_visualizers[i].visualize(self.body_pos_w[:, i], self.body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])
            # self.goal_body_visualizers[i].visualize(self.body_pos_ref_w[:, i], self.body_quat_ref_w[:, i])
@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    # motion_file: str = MISSING
    dataset_path_root: Path = MISSING
    dataset_names: list[str] = MISSING
    dataset_weights: list[float] = MISSING
    slow_down_factor: int = 1
    anchor_body: str = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
