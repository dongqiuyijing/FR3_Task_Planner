import time
import os
import rospkg
import sys
# 获取当前ROS包的路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# 动态构建scripts目录路径
scripts_path = os.path.join(pkg_path, 'fairino-python-sdk-master/linux/fairino')
sys.path.insert(0,scripts_path)
import Robot  # type: ignore
import time
import rospy
from std_msgs.msg import Float32MultiArray

class fr_Robot():
    def __init__(self):
        self.robot = Robot.RPC('192.168.58.2')
    def ServoCart(self,mode, desc_pos, pos_gain = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] , acc = 0.0, vel = 0.0, cmdT = 0.008, filterT = 0.0, gain = 0.0):
        """笛卡尔空间伺服控制
            mode:[0]-绝对运动(基坐标系),[1]-增量运动(基坐标系),[2]-增量运动(工具坐标系);
            desc_pos:目标笛卡尔位置/目标笛卡尔位置增量；
        """
        error = self.robot.ServoCart(mode, desc_pos, pos_gain , acc, vel, cmdT, filterT, gain)
        return error
    def ServoMoveStart(self):
        """启动伺服控制"""
        error = self.robot.ServoMoveStart()
        return error
    def ServoMoveEnd(self):
        """关闭伺服控制"""
        error = self.robot.ServoMoveEnd()
        return error
    def GetActualTCPPose(self):
        """获取当前机械臂笛卡尔位姿态"""
        error,data = self.robot.GetActualTCPPose()
        return error,data
    def MoveCart(self,pose,tool,wob,vel = 30):
        error = self.robot.MoveCart(pose,tool,wob,vel)
        return error

    