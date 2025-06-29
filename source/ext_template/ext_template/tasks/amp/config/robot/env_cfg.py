from isaaclab.utils import configclass

from ext_template.tasks.amp.amp_env_cfg import *
from isaaclab_assets.robots.unitree import H1_CUSTOM_CFG, G1_29DOF_CFG

        
@configclass
class H1AmpEnvCfg(AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_CUSTOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_positions = mdp.JointPositionActionCfg(asset_name="robot", 
                                                joint_names=["left_hip_yaw_joint", 
                                                             "left_hip_roll_joint",
                                                             "left_hip_pitch_joint", 
                                                             "left_knee_joint", 
                                                             "left_ankle_joint",
                                                             "right_hip_yaw_joint", 
                                                             "right_hip_roll_joint",
                                                             "right_hip_pitch_joint", 
                                                             "right_knee_joint", 
                                                             "right_ankle_joint",
                                                             "torso_joint", 
                                                             "left_shoulder_pitch_joint",
                                                             "left_shoulder_roll_joint", 
                                                             "left_shoulder_yaw_joint",
                                                             "left_elbow_joint", 
                                                             "right_shoulder_pitch_joint",
                                                             "right_shoulder_roll_joint", 
                                                             "right_shoulder_yaw_joint",
                                                             "right_elbow_joint"],
                                                 scale=1.0, 
                                                 use_default_offset=True)
        
@configclass
class G1AmpEnvCfg(AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_positions = mdp.JointPositionActionCfg(asset_name="robot", 
                                                joint_names=['left_hip_pitch_joint',
                                                             'left_hip_roll_joint',
                                                             'left_hip_yaw_joint',
                                                             'left_knee_joint',
                                                             'left_ankle_pitch_joint',
                                                             'left_ankle_roll_joint',
                                                             'right_hip_pitch_joint',
                                                             'right_hip_roll_joint',
                                                             'right_hip_yaw_joint',
                                                             'right_knee_joint',
                                                             'right_ankle_pitch_joint',
                                                             'right_ankle_roll_joint',
                                                             'waist_yaw_joint',
                                                             'waist_roll_joint',
                                                             'waist_pitch_joint',
                                                             'left_shoulder_pitch_joint',
                                                             'left_shoulder_roll_joint',
                                                             'left_shoulder_yaw_joint',
                                                             'left_elbow_joint',
                                                             'left_wrist_roll_joint',
                                                             'left_wrist_pitch_joint',
                                                             'left_wrist_yaw_joint',
                                                             'right_shoulder_pitch_joint',
                                                             'right_shoulder_roll_joint',
                                                             'right_shoulder_yaw_joint',
                                                             'right_elbow_joint',
                                                             'right_wrist_roll_joint',
                                                             'right_wrist_pitch_joint',
                                                             'right_wrist_yaw_joint'],
                                                 scale=1.0, 
                                                 use_default_offset=True)
