#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
class USBImageCropper:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('usb_camera_cropper', anonymous=True)

        # 创建图像发布者
        self.image_pub = rospy.Publisher('/camera/cropped_image', Image, queue_size=10)

        # 创建 CvBridge 对象
        self.bridge = CvBridge()

        # 打开 USB 相机
        for i in range(10):
            self.cap = cv2.VideoCapture(i)  # 0 表示第一个相机
            if self.cap.isOpened():
                break
            else :
                rospy.logerr("无法打开 USB 相机")
                exit()

           # 设置图像分辨率
        width = 1280  # 想要设置的宽度
        height = 720  # 想要设置的高度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"相机分辨率：{width}x{height}")
        # 矩形区域的坐标（x, y, width, height）
        self.crop_x = 25
        self.crop_y = 125
        self.crop_width = 420
        self.crop_height = 420
        # self.crop_x = 80
        # self.crop_y = 250
        # self.crop_width = 300
        # self.crop_height = 200
        self.image_old = None
        # 循环捕获和发布图像
        self.publish_images()
    def publish_images(self):

        while not rospy.is_shutdown():
            # 捕获图像
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("无法读取图像")
                continue

            # 截取矩形区域
            cropped_image = frame[self.crop_y:self.crop_y + self.crop_height,
                                  self.crop_x:self.crop_x + self.crop_width]

            # 将 OpenCV 图像转换为 ROS 图像消息
            try:
                ros_image = self.bridge.cv2_to_imgmsg(cropped_image, encoding="bgr8")
                ros_image.header.stamp = rospy.Time.now()
                if np.array_equal(ros_image, self.image_old):
                    print("0")
                    continue
                self.image_old = ros_image
                self.image_pub.publish(ros_image)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge 错误: {e}")

        # 释放相机
        self.cap.release()

if __name__ == '__main__':
    try:
        USBImageCropper()
    except rospy.ROSInterruptException:
        pass