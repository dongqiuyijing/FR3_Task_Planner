#!/usr/bin/env python
import rospy
import actionlib
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import math
import tf
from sensor_msgs.msg import Image
import rospkg
import sys
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# 动态构建scripts目录路径
scripts_path = os.path.join(pkg_path, 'fairino-python-sdk-master/linux/fairino')
sys.path.insert(0,scripts_path)


class ActionServer:
    def __init__(self):
        rospy.init_node('image_save_node', anonymous=True)
        self.image_count = 0
        self.image_old = np.zeros((420,420))
        self.T_image = True
        self.T_tool = np.array([[0,1,0,0],
                            [-1,0,0,0],
                            [0,0,1,182],
                            [0,0,0,1]])
        self.server = actionlib.SimpleActionServer(
            'message_1', 
            MessageAction, 
            execute_cb=self.execute_cb,
            auto_start=False)
        self.server.start()

    def image_callback(self,msg):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            if np.array_equal(cv_image,image_old):
                # print("图片重复")
                return
            image_old = cv_image
        except Exception as e:
            print(e)
            return
        # 指定保存图像的目录，可按需修改
        save_dir_image = "/home/admind/control_ws/src/realsense_control_pkg/config/data13/image"
        if not os.path.exists(save_dir_image):
            os.makedirs(save_dir_image)
        # 生成图像文件名，按照顺序编号，如image_0.jpg、image_1.jpg等
        save_path_image = os.path.join(save_dir_image, f"image_{self.image_count}.jpg")
        cv2.imwrite(save_path_image, cv_image)
        print(f"Image {self.image_count} saved successfully!")

        # 模拟的图像相关列表数据（这里你要替换为真实的列表数据获取逻辑）
        error,desc_pos = self.robot.GetActualTCPPose()
        list_data = desc_pos
        euler_angle = [desc_pos[3]*math.pi/180,desc_pos[4]*math.pi/180,desc_pos[5]*math.pi/180]
        R = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angle))[:3, :3]
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = np.array([desc_pos[0],desc_pos[1],desc_pos[2]])
        #求解安装了工具的运行路径
        T_move = np.dot(T,self.T_tool)
        quaternion_move = tf.transformations.quaternion_from_matrix(T_move)
        pos_move = T_move[:3,3]
        angle = tf.transformations.euler_from_quaternion(quaternion_move)
        angle = np.array(angle)*180/math.pi
        list_data=[pos_move[0],pos_move[1],pos_move[2],angle[0],angle[1],angle[2]]
        # 指定保存列表数据的TXT文件目录，这里统一使用同一个文件，可按需修改
        save_dir_txt = "/home/admind/control_ws/src/realsense_control_pkg/config/data13/data"
        if not os.path.exists(save_dir_txt):
            os.makedirs(save_dir_txt)
        file_name = "all_data.txt"
        save_path_txt = os.path.join(save_dir_txt, file_name)
        mode = 'w' if self.image_count == 0 else 'a'
        with open(save_path_txt, mode) as f:
            for element in list_data:
                f.write(str(element) + " ")
            f.write("\n")
        print(f"List data for Image {self.image_count} saved to text file successfully!")
        self.image_count += 1
    def execute_cb(self, goal):
        # 处理目标请求
        feedback = MessageFeedback()
        result = MessageResult()
        # 替换为实际发布图像的ROS话题名称
        if self.T_image:
            self.sub = rospy.Subscriber("/camera/cropped_image", Image, self.image_callback)
        else:
            # 需要关闭时调用
            self.sub.unregister()
        self.T_image = not self.T_image
        feedback.status = "proccing"
        result.reply = "success"   
        
        self.server.publish_feedback(feedback)
        self.server.set_succeeded(result)  # 明确设置终止状态

if __name__ == '__main__':
    rospy.init_node('action_server')
    server = ActionServer()
    rospy.spin()