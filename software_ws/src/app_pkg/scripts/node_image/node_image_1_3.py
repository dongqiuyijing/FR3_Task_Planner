#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import actionlib
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
import time
import sys
import rospkg
import os
import signal  # 新增 signal 模块
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0,scripts_path)
from FR_SDK import fr_Robot
from std_msgs.msg import Float32MultiArray
import nav_msgs.msg
from gui import button
TaskController = button.TaskController
import numpy as np
import tf
from std_msgs.msg import String
import math
import actionlib
import random
from app_pkg.msg import MessageAction, MessageGoal ,MessageResult# type: ignore

class SDK_control():
    def __init__(self):
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)  # 注册 Ctrl+C 处理函数
        self.pub_run = rospy.Publisher('chat2', String, queue_size=10)
        self.robot = fr_Robot()
        rospy.init_node('action_server', anonymous=True)
        self.robot.ServoMoveStart()
        self.shutdown_flag = False  # 标志位，用于安全退出循环
        self.T_tool = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,120],
                            [0,0,0,1]])
    def signal_handler(self, sig, frame):
        """处理 Ctrl+C 信号"""
        rospy.loginfo("检测到 Ctrl+C，正在安全关闭...")
        self.shutdown_flag = True  # 设置标志位，通知循环退出
        self.cleanup()  # 执行清理操作
        sys.exit(0)  # 退出程序

    def cleanup(self):
        """安全关闭机器人连接"""
        rospy.loginfo("执行清理操作...")
        try:
            self.robot.ServoMoveEnd()
        except Exception as e:
            rospy.logerr(f"清理时发生错误: {e}")
        finally:
            rospy.loginfo("清理完成")

    def change_180(self,go_pos):
        if go_pos[3] > 180 :
            go_pos[3]-=360
        elif go_pos[3] < -180:
            go_pos[3]+=360
        if go_pos[5] > 180 :
            go_pos[5]-=360
        elif go_pos[5] < -180:
            go_pos[5]+=360
    def change_error(self,aim_pose,desc_pos,error_pose):
        if aim_pose[3]*desc_pos[3]<0:
            if aim_pose[3]<0 :
                error_pose[3]+=360
            else:
                error_pose[3]-=360 
        if aim_pose[5]*desc_pos[5]<0: 
            if aim_pose[5]<0 :
                error_pose[5]+=360
            else:
                error_pose[5]-=360 
    def tf_angle2T_tool(self,now_pose,T_tool): # 欧拉角转化成矩阵
        """输入是六维欧拉角，输出是变化矩阵"""
        # 先将欧拉角转换成变换矩阵
        euler_angle = [now_pose[3]*math.pi/180,now_pose[4]*math.pi/180,now_pose[5]*math.pi/180]
        R_1 = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angle))[:3, :3]
        T_aim = np.eye(4)
        T_aim[:3,:3] = R_1
        T_aim[:3,3] = np.array([now_pose[0],now_pose[1],now_pose[2]])
        T_move = np.dot(T_aim, np.linalg.inv(T_tool))
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
    def T_tool2tf_angle(self,now_pose,T_tool): # 欧拉角转化成矩阵
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
    def path_form(self):
        """生成平滑路径"""
        pos1 = np.array([-147 ,-530, 250.0, 180.0, 0.0, 180], dtype=np.float64)
        pose = self.tf_angle2T_tool(pos1.copy(),self.T_tool)
        error = self.robot.MoveCart(pose, 0, 0, vel=20)
            
        time.sleep(1)
        pos1 = np.array([-147 ,-530, 138.0, 180.0, 0.0, 180], dtype=np.float64)
        pose = self.tf_angle2T_tool(pos1.copy(),self.T_tool)
        error = self.robot.MoveCart(pose, 0, 0, vel=20)
        start_time = time.time()
        pos2 = np.array([-222, -451.0, 130.0, 178.0, -3.8, 178], dtype=np.float64)
        random_list =[0,0,0,0,0,0]
        num_points = 75
        for t in np.linspace(0, 1, num_points):
            if self.shutdown_flag:  # 检查是否收到退出信号
                break  # 提前退出循环
            aim_pose = pos1 + t * (pos2 - pos1)
            random_list[0] = random.uniform(-1, 1)
            random_list[1] = random.uniform(-1, 1)
            random_list[2] = random.uniform(-1, 1)
            random_list[3] = random.uniform(-5, 5)
            random_list[4] = random.uniform(-5, 5)
            random_list[5] = random.uniform(-5, 5)
            aim_pose[0] += random_list[0]
            aim_pose[1] += random_list[1]
            aim_pose[2] += random_list[2]
            aim_pose[3] += random_list[3]
            aim_pose[4] += random_list[4]
            aim_pose[5] += random_list[5]
            aim_pose = self.tf_angle2T_tool(aim_pose,self.T_tool)
            error, desc_pos = self.robot.GetActualTCPPose()
            error_pose = np.array(aim_pose) - np.array(desc_pos)
            self.change_error(aim_pose, desc_pos, error_pose)
            magnitude = np.linalg.norm(error_pose)
            steps = max(1, int(magnitude / 0.28))  # 确保至少1步
            for i in range(steps):
                if self.shutdown_flag:  # 再次检查
                    break
                step_ratio = (i + 1) / steps
                intermediate_pos = desc_pos + step_ratio * error_pose
                self.change_180(intermediate_pos)
                self.robot.ServoCart(0, intermediate_pos)
                time.sleep(0.015)  # 控制速度
        num_points = 30
        for j in range(num_points+1):
            aim_pose = pos2.copy()
            random_0 = ((num_points-j)/num_points) * random.uniform(-1, 1)
            random_1 = ((num_points-j)/num_points) * random.uniform(-1, 1)
            random_2 = ((num_points-j)/num_points) * random.uniform(-1, 1)
            random_3 = ((num_points-j)/num_points) * random.uniform(-5, 5)
            random_4 = ((num_points-j)/num_points) * random.uniform(-5, 5)
            random_5 = ((num_points-j)/num_points) * random.uniform(-5, 5)
            aim_pose[0] += random_0
            aim_pose[1] += random_1
            aim_pose[2] += random_2
            aim_pose[3] += random_3
            aim_pose[4] += random_4
            aim_pose[5] += random_5
            aim_pose = self.tf_angle2T_tool(aim_pose,self.T_tool)
            error, desc_pos = self.robot.GetActualTCPPose()
            error_pose = np.array(aim_pose) - np.array(desc_pos)
            self.change_error(aim_pose, desc_pos, error_pose)
            magnitude = np.linalg.norm(error_pose)
            steps = max(1, int(magnitude / 0.1))  # 确保至少1步
            for i in range(steps):
                if self.shutdown_flag:  # 再次检查
                    break
                step_ratio = (i + 1) / steps
                intermediate_pos = desc_pos + step_ratio * error_pose
                self.change_180(intermediate_pos)
                error = self.robot.ServoCart(0, intermediate_pos)
                time.sleep(0.015)  # 控制速度
         # 第二次获取时间
        end_time = time.time()
        # 计算时间间隔
        time_interval = end_time - start_time
        print(f"总耗时: {time_interval:.2f}秒")

def main():
    # try:
    sdk = SDK_control()
    sdk.path_form()
    # except rospy.ROSInterruptException:
    #     rospy.loginfo("节点被中断")
    # except Exception as e:
    #     rospy.logerr(f"主函数错误: {str(e)}")
    # finally:
    #     rospy.loginfo("节点关闭")

if __name__ == "__main__":
    main()