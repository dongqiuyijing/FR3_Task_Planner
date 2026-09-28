#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import re
import numpy as np

class ImageFilePublisher:
    def __init__(self, image_folder):
        # 初始化 ROS 节点
        rospy.init_node('image_file_publisher', anonymous=True)
        
        # 创建图像发布者
        self.image_pub = rospy.Publisher('/camera/cropped_image', Image, queue_size=10)
        
        # 创建 CvBridge 对象
        self.bridge = CvBridge()
        
        # 图像文件夹路径
        self.image_folder = image_folder
        
        # 获取文件列表并自然排序
        self.image_files = self.natural_sort(os.listdir(image_folder))
        
        # 设置发布频率
        self.rate = rospy.Rate(10)  # 每秒发布一次

    def natural_sort(self, file_list):
        """自然排序函数"""
        # 使用正则表达式提取数字
        def alphanum_key(key):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', key)]
        
        return sorted(file_list, key=alphanum_key)

    def publish_images(self):
        count = 0
        for image_file in self.image_files:
            if count <= 1:
                count += 1
                continue
            # 构建完整的图像路径
            image_path = os.path.join(self.image_folder, image_file)
            
            # 读取图像
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                continue
            
            # 将图像转换为 ROS 图像消息
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            # print(cv_image.mean())
            # 添加时间戳
            ros_image.header.stamp = rospy.Time.now()
            
            # 发布图像
            self.image_pub.publish(ros_image)
            print("发送一张")
            # # 显示图像并监听键盘事件
            cv2.imshow('Image', cv_image)
            key = cv2.waitKey(10)  # 等待 100 毫秒
            if key == ord('q'):  # 如果按下 'q' 键
                cv2.destroyAllWindows()
                rospy.signal_shutdown("User pressed 'q' to quit.")
                break

            # 等待下一次发布
            self.rate.sleep()

if __name__ == '__main__':
    image_folder = '/home/admind/software_ws/src/app_pkg/config/data2/image'  # 替换为实际的图像文件夹路径
    try:
        image_publisher = ImageFilePublisher(image_folder)
        image_publisher.publish_images()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()