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
import shutil
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
        self.image_old = None
        self.current_pose = None  # 存储当前接收到的机械臂位姿
        # 创建CV桥接
        self.bridge = CvBridge()
         # 自动创建数据保存目录
        self.base_dir = "/home/admind/software_ws/src/app_pkg/config"
        self.save_dir_image, self.save_dir_txt  = self.create_new_data_dir()
        self.image_sub = rospy.Subscriber("/camera/cropped_image", Image, self.image_callback, queue_size=1)
        # self.server = actionlib.SimpleActionServer('data_save',  MessageAction, lambda goal: self._execute_callback(goal,self.server) ,auto_start=False)
        # self.server.start()
        self.data_save_run = False
    def create_new_data_dir(self):
        index = 1
        while True:
            image_dir = os.path.join(self.base_dir, f"image{index}", "image")
            data_dir = os.path.join(self.base_dir, f"image{index}", "data")
            
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
    
    def image_callback(self, msg):
        try:
            # 检查是否已接收到机械臂位姿
            # if self.current_pose is None:
            #     rospy.logwarn("Waiting for robot pose data...")
            #     return
                
            # 转换图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 检查是否重复图像
            if np.array_equal(cv_image, self.image_old):
                return
            self.image_old = cv_image
            
            # 保存图像
            save_path_image = os.path.join(self.save_dir_image, f"image_{self.image_count}.jpg")
            cv2.imwrite(save_path_image, cv_image)
            rospy.loginfo(f"Image {self.image_count} saved successfully!")
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