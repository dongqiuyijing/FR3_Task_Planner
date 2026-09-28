#!/home/admind/software_ws/src/app_pkg/vens/venv1/bin/python
import os
import sys
import numpy
import rospy
from std_msgs.msg import String
print(f"NumPy路径: {numpy.__file__}")
print(f"rospy路径: {rospy.__file__}")
def publisher():
    # 初始化节点，名称唯一
    rospy.init_node('simple_publisher', anonymous=True)
    
    # 创建发布者，发布到 topic "chatter"，消息类型 String，队列大小 10
    pub = rospy.Publisher('chatter', String, queue_size=10)
    
    rate = rospy.Rate(1)  # 1Hz 发布频率
    
    while not rospy.is_shutdown():
        message = f"Hello ROS at {rospy.get_time()}"
        rospy.loginfo(f"Publishing: {message}")  # 打印日志
        pub.publish(message)  # 发布消息
        rate.sleep()  # 按指定频率休眠

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass