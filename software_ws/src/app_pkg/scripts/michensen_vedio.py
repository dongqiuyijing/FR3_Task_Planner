#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class USBCameraPublisher:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('usb_camera_publisher', anonymous=True)

        # 创建图像发布者
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

        # 创建 CvBridge 对象
        self.bridge = CvBridge()

        # 打开 USB 相机
        self.cap = None
        for i in range(10):  # 尝试多个设备号
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                rospy.loginfo(f"成功打开 USB 相机设备 /dev/video{i}")
                break
        
        if not self.cap or not self.cap.isOpened():
            rospy.logerr("无法打开任何 USB 相机设备")
            exit(1)

        # 设置图像分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 获取实际分辨率
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rospy.loginfo(f"相机分辨率设置为: {width}x{height}")

        # 创建OpenCV窗口
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Preview", 640, 480)

        # 开始发布图像
        self.publish_images()

    def publish_images(self):
        rate = rospy.Rate(30)  # 30Hz
        while not rospy.is_shutdown():
            # 捕获图像
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("从相机读取帧失败")
                continue

            # 显示图像
            cv2.imshow("Camera Preview", frame)
            
            # 检测按键输入 (ESC退出)
            key = cv2.waitKey(1)
            if key == 27:  # ESC键
                rospy.signal_shutdown("用户按下ESC键退出")
                break

            # 将 OpenCV 图像转换为 ROS 图像消息
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                ros_image.header.stamp = rospy.Time.now()
                self.image_pub.publish(ros_image)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge 错误: {e}")

            rate.sleep()

        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        rospy.loginfo("相机资源已释放")

if __name__ == '__main__':
    try:
        USBCameraPublisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点已关闭")