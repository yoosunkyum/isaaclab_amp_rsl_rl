from isaaclab.utils import configclass

from ext_template.tasks.amp.amp_env_cfg import *
from isaaclab_assets.robots.unitree import H1_CUSTOM_CFG

        
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
                                                 scale=0.2, 
                                                 use_default_offset=True)
