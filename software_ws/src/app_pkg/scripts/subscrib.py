#!/home/admind/software_ws/src/app_pkg/vens/venv2/bin/python
import os
import sys

# 测试代码
import numpy
print(f"=== VENV1 环境 ===")
print(f"2Python解释器: {sys.executable}")
print(f"NumPy版本: {numpy.__version__}")
print(f"NumPy路径: {numpy.__file__}")

import rospy
from std_msgs.msg import String

def callback(msg):
    # 收到消息时的回调函数
    rospy.loginfo(f"Received: {msg.data}")

def subscriber():
    # 初始化节点
    rospy.init_node('simple_subscriber', anonymous=True)
    
    # 创建订阅者，订阅 topic "chatter"，回调函数 callback
    rospy.Subscriber('chatter', String, callback)
    
    rospy.spin()  # 保持节点运行，直到关闭

if __name__ == '__main__':
    subscriber()