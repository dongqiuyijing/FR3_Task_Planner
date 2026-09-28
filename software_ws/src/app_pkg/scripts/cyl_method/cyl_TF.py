import tf
import math
import numpy as np

def tf_angle2angle_Z(pose:list,move_Z): #欧拉角不发生变化，以当前姿态沿着工件的z轴移动
    """输入,输出都是一个6维变量的欧拉角参数,角度制的输入输出"""
    # 先将欧拉角转换成变换矩阵
    euler_angle = [pose[3]*math.pi/180,pose[4]*math.pi/180,pose[5]*math.pi/180]
    R_1 = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angle))[:3, :3]
    T_aim = np.eye(4)
    T_aim[:3,:3] = R_1
    T_aim[:3,3] = np.array([pose[0],pose[1],pose[2]+move_Z])
    quaternion_move = tf.transformations.quaternion_from_matrix(T_aim)
    pos_move = T_aim[:3,3]
    angle = tf.transformations.euler_from_quaternion(quaternion_move)
    angle = np.array(angle)*180/math.pi
    list_data = [pos_move[0],pos_move[1],pos_move[2],angle[0],angle[1],angle[2]]
    return list_data
def tf_angle2T(now_pose): # 欧拉角转化成矩阵
    """输入是六维欧拉角，输出是变化矩阵"""
    # 先将欧拉角转换成变换矩阵
    euler_angle = [now_pose[3]*math.pi/180,now_pose[4]*math.pi/180,now_pose[5]*math.pi/180]
    R_1 = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angle))[:3, :3]
    T_aim = np.eye(4)
    T_aim[:3,:3] = R_1
    T_aim[:3,3] = np.array([now_pose[0],now_pose[1],now_pose[2]])
    # print("当前高度矩阵",T_aim)
    # 获取到当前位态 T_now
    return T_aim
def tf_angle2T_tool(now_pose,T_tool): # 欧拉角转化成矩阵
    """输入是六维欧拉角，输出是变化矩阵"""
    # 先将欧拉角转换成变换矩阵
    euler_angle = [now_pose[3]*math.pi/180,now_pose[4]*math.pi/180,now_pose[5]*math.pi/180]
    R_1 = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angle))[:3, :3]
    T_aim = np.eye(4)
    T_aim[:3,:3] = R_1
    T_aim[:3,3] = np.array([now_pose[0],now_pose[1],now_pose[2]])
    T_move = np.dot(T_aim, T_tool)
    quaternion_move = tf.transformations.quaternion_from_matrix(T_move)
    pos_move = T_move[:3, 3]
    angle = tf.transformations.euler_from_quaternion(quaternion_move)
    angle = np.array(angle) * 180/math.pi
    
    # 构建位姿数据
    pose_data = [
        pos_move[0], pos_move[1], pos_move[2],
        angle[0], angle[1], angle[2]
    ]
    return pose_data
def tf_T2angle(T_aim):# 旋转矩阵转成欧拉角
    quaternion_move = tf.transformations.quaternion_from_matrix(T_aim)
    pos_move = T_aim[:3,3]
    angle = tf.transformations.euler_from_quaternion(quaternion_move)
    angle = np.array(angle)*180/math.pi
    list_data = [pos_move[0],pos_move[1],pos_move[2],angle[0],angle[1],angle[2]]
    return list_data
def tf_pose2T(pose): #pose转成矩阵
    pos_aim = [pose.pose.position.x,pose.pose.position.y,pose.pose.position.z]
    quaternion = [pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w]
    R_aim = tf.transformations.quaternion_matrix(quaternion)[:3,:3]
    T_aim = np.eye(4)
    T_aim[:3,:3]=R_aim
    T_aim[:3,3]=np.array(pos_aim)
    return T_aim











