#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
from cv_bridge import CvBridge
import os
import sys
import math
import tf
import numpy as np
import actionlib
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
import time

class ImagePoseSaver:
    def __init__(self):
        rospy.init_node('image_pose_saver_node', anonymous=True)
        
        # 工具坐标系变换矩阵
        self.T_tool = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 120],
            [0, 0, 0, 1]
        ])
        # 图像计数器
        self.image_count = 0
        self.image_old = np.zeros((420,420))
        self.current_pose = None  # 存储当前接收到的机械臂位姿
        # 创建CV桥接
        self.bridge = CvBridge()
         # 自动创建数据保存目录
        self.base_dir = "/home/admind/software_ws/src/app_pkg/config"
        self.server = actionlib.SimpleActionServer('data_save',  MessageAction, lambda goal: self._execute_callback(goal,self.server) ,auto_start=False)
        self.server.start()
        self.data_save_run = False
        self.angle_cornor = 0
        self.angle_cornor_num = 0
    def _execute_callback(self,goal,server):
        feedback = MessageFeedback()
        result = MessageResult() 
        self.data_save_run = not self.data_save_run
        if self.data_save_run:
            self.image_count = 0
            self.save_dir_image, self.save_dir_txt = self.create_new_data_dir()
            self.image_sub = rospy.Subscriber("/camera/cropped_image", Image, self.image_callback, queue_size=1)
            # 订阅机械臂位姿话题
            self.pose_sub = rospy.Subscriber("arm_pose", Float32MultiArray, self.pose_callback, queue_size=1)
            feedback.status = "start data save!!!"
            result.reply = "start successful"
        else:
            self.image_sub.unregister()
            self.pose_sub.unregister()
            feedback.status = "stop data save "
            result.reply = "stop successful"
        server.publish_feedback(feedback)
        time.sleep(0.1)
        server.set_succeeded(result)  # 明确
    def create_new_data_dir(self):
        """自动创建递增序号的数据目录"""
        index = 1
        while True:
            image_dir = os.path.join(self.base_dir, f"data{index}", "image")
            data_dir = os.path.join(self.base_dir, f"data{index}", "data")
            
            # 检查目录是否已存在
            if not os.path.exists(image_dir) and not os.path.exists(data_dir):
                os.makedirs(image_dir, exist_ok=True)
                os.makedirs(data_dir, exist_ok=True)
                rospy.loginfo(f"Created new data directory: data{index}")
                return image_dir, data_dir
            
            index += 1
            # 防止无限循环，设置最大尝试次数
            if index > 1000:
                rospy.logerr("Failed to create new data directory after 1000 attempts")
                raise Exception("Too many existing data directories")
    
    def pose_callback(self, msg):
        """接收机械臂位姿信息的回调函数"""
        self.current_pose = msg.data
        rospy.logdebug(f"Received new robot pose: {self.current_pose}")
    
    def image_callback(self, msg):
        try:
             # 使用从话题接收的机械臂位姿
            desc_pos = self.current_pose
            # print("转角变换度数",abs(self.angle_cornor - desc_pos[5]))
            if abs(self.angle_cornor - desc_pos[5])>0.2:
                # print("转角图片不要")
                self.angle_cornor = desc_pos[5]
                return
                # self.angle_cornor_num +=1
                # if self.angle_cornor_num < 5:
                #     return
                # else :
                #     self.angle_cornor_num = 0
            self.angle_cornor = desc_pos[5]
            # 检查是否已接收到机械臂位姿
            if self.current_pose is None:
                rospy.logwarn("Waiting for robot pose data...")
                return
                
            # 转换图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 检查是否重复图像
            if np.array_equal(cv_image, self.image_old):
                print("检测到重复图片")
                return
            self.image_old = cv_image
            
            # 保存图像
            save_path_image = os.path.join(self.save_dir_image, f"image_{self.image_count}.jpg")
            cv2.imwrite(save_path_image, cv_image)
            rospy.loginfo(f"Image {self.image_count} saved successfully!")
            
            # 计算工具末端位姿
            euler_angle = [desc_pos[3]*math.pi/180, desc_pos[4]*math.pi/180, desc_pos[5]*math.pi/180]
            quaternion = tf.transformations.quaternion_from_euler(*euler_angle)
            R = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array([desc_pos[0], desc_pos[1], desc_pos[2]])
            
            # 计算安装工具后的运行位姿
            T_move = np.dot(T, self.T_tool)
            quaternion_move = tf.transformations.quaternion_from_matrix(T_move)
            pos_move = T_move[:3, 3]
            angle = tf.transformations.euler_from_quaternion(quaternion_move)
            angle = np.array(angle) * 180/math.pi
            
            # 构建位姿数据
            pose_data = [
                pos_move[0], pos_move[1], pos_move[2],
                angle[0], angle[1], angle[2]
            ]
            
            # 保存位姿数据到文件
            file_name = "all_data.txt"
            save_path_txt = os.path.join(self.save_dir_txt, file_name)
            mode = 'w' if self.image_count == 0 else 'a'
            with open(save_path_txt, mode) as f:
                f.write(' '.join(map(str, pose_data)) + "\n")
            
            rospy.loginfo(f"Pose data for Image {self.image_count} saved successfully!")
            self.image_count += 1
            
        except Exception as e:
            rospy.logerr(f"Error in image callback: {str(e)}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ImagePoseSaver()
        node.run()
    except rospy.ROSInterruptException:
        pass