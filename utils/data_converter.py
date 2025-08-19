import joblib
import torch
import pinocchio as pin
import numpy as np

# from phc.utils.torch_humanoid_batch import Humanoid_Batch

def main():
    """load retargetted pickle data"""
    DATA_NAME = 'g1_walk'
    motion_data = joblib.load(DATA_NAME+'.pkl')
    data = motion_data[list(motion_data.keys())[0]]

    episode_length = len(data['root_trans_offset'])
    print('episode length : ', episode_length)
    print('root_trans_offset length : ', data['root_trans_offset'].shape)
    print('pose_aa length : ', data['pose_aa'].shape)
    print('dof : ', data['dof'].shape)
    print('root_rot : ', data['root_rot'].shape)
    print('smpl_joints : ', data['smpl_joints'].shape)
    print('fps : ', data['fps'])
   
    """load robot model & solve forward kinematics"""
    # model_filename = "../../../../phc/data/assets/robot/unitree_g1_29dof/g1_29dof_rev_1_0.xml"
    # mjcf_model = pin.buildModelFromMJCF(model_filename, pin.JointModelFreeFlyer()) #floating base
    model_filename = "../../../../phc/data/assets/robot/unitree_g1_29dof/g1_29dof_rev_1_0.urdf"
    mjcf_model = pin.buildModelFromUrdf(model_filename, pin.JointModelFreeFlyer()) #floating base
    mjcf_data = mjcf_model.createData()    

    print("model name : ", mjcf_model.name)
    print("joint size : ", mjcf_model.nq)
    print("number of bodies : ", mjcf_model.nbodies)
    print("number of frames : ", mjcf_model.nframes)
    print("number of joints : ", mjcf_model.njoints)

    """joint position"""
    q = np.zeros((data['root_trans_offset'].shape[0], 
                data['root_trans_offset'].shape[1] + data['root_rot'].shape[1] + data['dof'].shape[1]))

    # print('q shape : ', q.shape)
    q[:,:3] = data['root_trans_offset']
    q[:,3:7] = data['root_rot']
    q[:,7:] = data['dof']

    """data conversion"""
    conv_data={}
    conv_data["fps"] = data['fps']
    conv_data["dof_names"] = [mjcf_model.names[k] for k in range(mjcf_model.njoints)]
    print(conv_data["dof_names"])
    conv_data["body_names"] = [frame.name for frame in mjcf_model.frames if frame.type == pin.FrameType.BODY]
    conv_data["dof_positions"] = data['dof']

    dt = 1 / data['fps']

    """dof velocity calculation"""
    v_temp = []
    for i in range(episode_length-1):
        v_temp.append((conv_data["dof_positions"][i+1]-conv_data["dof_positions"][i]) / dt)
    v_temp.append(v_temp[-1])
    conv_data["dof_velocities"] = np.vstack(v_temp)

    """body position & rotations calculation"""
    conv_data["body_positions"] = []
    conv_data["body_rotations"] = []
    for i in range(episode_length):
        pin.forwardKinematics(mjcf_model, mjcf_data, q[i,:])
        pin.updateFramePlacements(mjcf_model, mjcf_data)
        b_pos =[]
        b_rot =[]

        #compute all body xyzquat
        for j in range(len(conv_data["body_names"])):
            frame_id = mjcf_model.getFrameId(conv_data["body_names"][j])
            p_temp = pin.SE3ToXYZQUAT(mjcf_data.oMf[frame_id])
            b_pos.append(p_temp[:3])
            b_rot.append(p_temp[3:][[3,0,1,2]]) #NOTE : change order from xyzw to wxyz
        b_pos = np.vstack(b_pos)
        b_rot = np.vstack(b_rot)

        conv_data["body_positions"].append(b_pos)
        conv_data["body_rotations"].append(b_rot)

    conv_data["body_positions"] = np.stack(conv_data["body_positions"], axis=0)
    conv_data["body_rotations"] = np.stack(conv_data["body_rotations"], axis=0)
    # print('body_positions : ', conv_data["body_positions"].shape)
    # print('body_rotations : ', conv_data["body_rotations"].shape)

    """body linear velocities calculation"""
    b_vel_temp = []
    for i in range(episode_length - 1):
        b_vel_temp.append((conv_data["body_positions"][i+1]-conv_data["body_positions"][i]) / dt)
    b_vel_temp.append(b_vel_temp[-1])
    conv_data["body_linear_velocities"] = np.stack(b_vel_temp, axis=0)
    # print("body_linear_velocities : ", conv_data["body_linear_velocities"][1])

    """body angular velocities calculation"""
    q_vel_temp = []
    for i in range(episode_length - 1):
        q_vel_temp.append((conv_data["body_rotations"][i+1]-conv_data["body_rotations"][i]) / dt)
    q_vel_temp.append(q_vel_temp[-1])
    q_vel_temp = np.stack(q_vel_temp, axis=0)

    conv_data["body_angular_velocities"] = []
    for i in range(episode_length):
        q_vel_temp2=[]
        for j in range(len(conv_data["body_names"])):
            q_vel_temp2.append(quaternion_to_angular_velocity(conv_data["body_rotations"][i][j],q_vel_temp[i][j]))
        q_vel_temp2 = np.vstack(q_vel_temp2)
        conv_data["body_angular_velocities"].append(q_vel_temp2)
    conv_data["body_angular_velocities"] = np.stack(conv_data["body_angular_velocities"],axis=0)

    """save converted data into an npz file"""
    np.savez(DATA_NAME + "_converted", 
             fps = conv_data["fps"],
             dof_names = conv_data["dof_names"][2:],
             body_names = conv_data["body_names"],
             dof_positions = conv_data["dof_positions"],
             dof_velocities = conv_data["dof_velocities"],
             body_positions = conv_data["body_positions"],
             body_rotations = conv_data["body_rotations"],
             body_linear_velocities = conv_data["body_linear_velocities"],
             body_angular_velocities = conv_data["body_angular_velocities"])

    print("dof_names : ",conv_data["dof_names"])
    print("body_names : ",conv_data["body_names"])
    # print("body angular velocities : ", conv_data["body_angular_velocities"][0])

    npy = {"joints_list" : [row for row in conv_data["dof_names"][2:]],
           "joint_positions" : [row for row in conv_data["dof_positions"]],
           "root_position" : [row for row in conv_data["body_positions"][:,0,:]],
           "root_quaternion" : [row[[1,2,3,0]] for row in conv_data["body_rotations"][:,0,:]],#xyzw
           "fps" : conv_data["fps"] } 
    
    np.save(DATA_NAME + "_converted",npy)

"""functions for angular velocity calculation from quaternion"""
def quaternion_conjugate(q):
    """쿼터니언 켤레 계산"""
    q0, q1, q2, q3 = q
    return np.array([q0, -q1, -q2, -q3])

def quaternion_multiply(q1, q2):
    """두 쿼터니언의 곱 (q1 ⊗ q2)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_to_angular_velocity(q, q_dot):
    """
    단위 쿼터니언 q와 그 시간 미분 q_dot로부터
    각속도 벡터 omega를 계산
    """
    q_conj = quaternion_conjugate(q)
    omega_quat = quaternion_multiply(q_dot, q_conj)
    omega = 2.0 * omega_quat[1:]  # 벡터 파트만 사용
    return omega

if __name__ == "__main__":
    main()