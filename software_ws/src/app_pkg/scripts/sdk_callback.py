import sys #provides access to Python interpreter functionalities and system-specific parameters
import os
import rospy
import rospkg
# 获取当前ROS包的路径
# rospack = rospkg.RosPack()
# pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# # 动态构建scripts目录路径
# scripts_path = os.path.join(pkg_path, 'fairino-python-sdk-master/linux/fairino')
# sys.path.insert(0,scripts_path)
sys.path.insert(0,"/home/admind/control_ws/src/fairino-python-sdk-master/linux/fairino")
import tf
import numpy as np
import Robot  # type: ignore
from std_msgs.msg import Float32MultiArray
import math
robot = Robot.RPC('192.168.58.2')
class SDK_callback_control():
    def __init__(self):
        # rospy.init_node('sdk_callback', anonymous=True)
        self.pub = rospy.Publisher('float_list_topic', Float32MultiArray, queue_size=10)
        self.T_tool = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,260],
                                [0,0,0,1]])
        self.image_count = 0
    def save_data(self):
      while not rospy.is_shutdown():
        list_data = [0,0,0,0,0,0]
        error,desc_pos = self.robot.GetActualTCPPose()
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
        list_data[0]=pos_move[0]
        list_data[1]=pos_move[1]
        list_data[2]=pos_move[2]
        list_data[3]=angle[0]
        list_data[4]=angle[1]
        list_data[5]=angle[2]
        # 指定保存列表数据的TXT文件目录，这里统一使用同一个文件，可按需修改
        save_dir_txt = "/home/admind/software_ws/src/app_pkg/config/data/"
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
      
def main():
  sdk_control = SDK_callback_control()
  sdk_control.save_data()
  rospy.spin()

if __name__ == '__main__':
  main()