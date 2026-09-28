#!/usr/bin/env python
import rospy
import open3d as o3d
import numpy as np
import sys
sys.path.insert(0, "/home/admind/control_ws/src/fairino-python-sdk-master/linux/fairino")
import Robot  # type: ignore
import time
import math
# 与机器人控制器建立连接，连接成功返回一个机器人对象
class control_sdk :
    def __init__(self):
        self.robot = Robot.RPC('192.168.58.2')
        self.tool = 1 #工具坐标系编号
        self.user = 1 #工件坐标系编号
        self.P_size = 4
        self.D_size = 0.1
        self.aim_pos=[-220,-120,400,-180,0,93]
    def RunCart(self,desc_pos=[-300,-100,350,-180,0,93]):
        self.robot.MoveCart(desc_pos, self.tool, self.user)

    def RandomCart(self):
        self.robot.MoveCart(self.aim_pos, self.tool, self.user)
    def get_pos(self):
        error,desc_pos = self.robot.GetActualTCPPose()
        desc_pos[0] = desc_pos[0]*0.001
        desc_pos[1] = desc_pos[1]*0.001
        desc_pos[2] = desc_pos[2]*0.001
        desc_pos[3] = desc_pos[3]/180*math.pi
        desc_pos[4] = desc_pos[4]/180*math.pi
        desc_pos[5] = desc_pos[5]/180*math.pi
        rounded_list = [round(num, 5) for num in desc_pos]
        print(rounded_list)
def main():
    robot = control_sdk()
    while True:
        robot.get_pos()
        time.sleep(1)
    # for i in range(10):
    #     robot.RandomCart()
    #     time.sleep(0.5)
    #设置循环频率
if __name__ == '__main__':
    main()