from amp_rsl_rl.utils import AMPLoader, MotionLoader
import numpy as np
from pathlib import Path

if __name__  == "__main__":
    # motion = np.load("motions/h1/h1_walk.npy", allow_pickle=True).item()
    # motion2 = np.load("motions/h1/h1_walk.npz", "rb")

    # print("motion joint positions : ", motion["joint_positions"][0:2])
    # print("motion joint positions : ", motion2["dof_positions"][0:2, :])


    # motion3 = AMPLoader(device="cuda",
    #                    dataset_path_root=Path("C:/Research/isaaclab_amp_rsl_rl/motions/h1"),
    #                    dataset_names=["h1_walk"],
    #                    dataset_weights=[1.0],
    #                    simulation_dt = 1 /60,
    #                    slow_down_factor=1,
    #                    expected_joint_names=None
    #                    )

    # motion4 = MotionLoader(device="cuda",
    #                     dataset_path_root=Path("C:/Research/isaaclab_amp_rsl_rl/motions/h1"),
    #                     dataset_names=["h1_walk"],
    #                     dataset_weights=[1.0],
    #                     simulation_dt = 1 /60,
    #                     slow_down_factor=1,
    #                     expected_joint_names=None
    #                     )
    
    # print("motion 3 states : ", motion3.all_obs[0:3])

    # times = np.array([0, 1/60, 2/60])

    # print("motion 4 states : ", motion4.selected_generator(times))
    motion = np.load("motions/g1/g1_walk.npz","rb")
    print("dof names : ",motion["dof_names"])
    print("body names : ",motion["body_names"])

    motion2 = MotionLoader(device="cuda",
                        dataset_path_root=Path("C:/Research/isaaclab_amp_rsl_rl/motions/g1"),
                        dataset_names=["g1_walk"],
                        dataset_weights=[1.0],
                        simulation_dt = 1 /60,
                        slow_down_factor=1,
                        expected_joint_names=['left_hip_pitch_joint', 
                                          'right_hip_pitch_joint', 
                                          'waist_yaw_joint', 
                                          'left_hip_roll_joint', 
                                          'right_hip_roll_joint', 
                                          'waist_roll_joint', 
                                          'left_hip_yaw_joint', 
                                          'right_hip_yaw_joint', 
                                          'waist_pitch_joint', 
                                          'left_knee_joint', 
                                          'right_knee_joint', 
                                          'left_shoulder_pitch_joint', 
                                          'right_shoulder_pitch_joint', 
                                          'left_ankle_pitch_joint', 
                                          'right_ankle_pitch_joint', 
                                          'left_shoulder_roll_joint', 
                                          'right_shoulder_roll_joint', 
                                          'left_ankle_roll_joint', 
                                          'right_ankle_roll_joint', 
                                          'left_shoulder_yaw_joint', 
                                          'right_shoulder_yaw_joint', 
                                          'left_elbow_joint', 
                                          'right_elbow_joint', 
                                          'left_wrist_roll_joint', 
                                          'right_wrist_roll_joint', 
                                          'left_wrist_pitch_joint', 
                                          'right_wrist_pitch_joint', 
                                          'left_wrist_yaw_joint', 
                                          'right_wrist_yaw_joint']
                        )
    print(f"outer joint dof indices : ",motion2.get_dof_index(motion['dof_names']))
    print(f"inner joint dof indices : ",motion2.motion_dof_indexes)
