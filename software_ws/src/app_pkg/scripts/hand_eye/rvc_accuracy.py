# 手外标定矩阵精度检测 末端靶标法


#!/usr/bin/env python
import rospy
import open3d as o3d
import numpy as np
import os
import sys
sys.path.insert(0, "/home/admind/control_ws/src/fairino-python-sdk-master/linux/fairino")
import Robot  # type: ignore
import time
import math


# 获取当前工作目录的绝对路径
path = os.path.abspath(".")
# 将自定义模块路径添加到 Python 的模块搜索路径中
sys.path.insert(0, path + "/src/realsense_control_pkg/scripts")

import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import yaml
import PyRVC as RVC # type: ignore
# 加载相机内参
file_path = "/home/admind/control_ws/src/realsense_control_pkg/config/rvc标定文件.yaml"
with open(file_path, "r") as file: 
    parameter = yaml.load(file.read(), Loader=yaml.Loader)
    mtx = parameter['camera_matrix']
    dist = parameter['dist_coeff']
    mtx = np.array(mtx)
    dist = np.array(dist)

# ArUco 字典和参数
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
parameters = aruco.DetectorParameters()
# 与机器人控制器建立连接，连接成功返回一个机器人对象
class accuracy_test :
    def __init__(self):
        self.robot = Robot.RPC('192.168.58.2')
        self.tool = 1 #工具坐标系编号
        self.user = 1 #工件坐标系编号
        self.P_size = 4
        self.D_size = 0.1
        self.X_delta_sum = []
        self.Y_delta_sum = []
        self.Z_delta_sum = []
        self.T_base2camera = np.array([[0.06878885  ,0.99704467 ,-0.03420567,-0.3970135],
                                        [0.9925145  ,-0.06492766 , 0.10343772,-0.14850681],
                                        [0.10091113 ,-0.04106499 ,-0.99404759,0.97279382],
                                        [0,0,0,1]])

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
        print("机械臂位姿:",rounded_list)
        return rounded_list
    def posToMatrix(self,pose):
        """参数一：位置，单位毫米
            参数二：欧拉角，角度"""
        yaw = pose[5]
        pitch = pose[4]
        roll = pose[3]
        Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                        [np.sin(yaw),np.cos(yaw) ,0],
                        [0           ,0            ,1]])
        Ry = np.array([[np.cos(pitch) ,0,np.sin(pitch)],
                        [0             ,1,0            ],
                        [-np.sin(pitch),0,np.cos(pitch)]])
        Rx = np.array([[1,0          ,0           ],
                        [0,np.cos(roll),-np.sin(roll)],
                        [0,np.sin(roll),np.cos(roll) ]])
        R = np.dot(np.dot(Rz , Ry ) , Rx)
        A = np.eye(4)
        A[:3,:3] = R
        A[:3,3] = np.array([pose[0],pose[1],pose[2]])
        return A
    def App(self):
        # 初始化 RVC X 系统
        RVC.SystemInit()

        # 查找设备
        ret, devices = RVC.SystemListDevices(RVC.SystemListDeviceTypeEnum.All)
        if len(devices) == 0:
            print("未找到任何设备！")
            RVC.SystemShutdown()
            return -1
        device = devices[0]

        # 获取设备信息
        ret, info = device.GetDeviceInfo()
        print(info.name + "-" + info.sn)

        # 创建并打开相机
        x1 = RVC.X1.Create(device, RVC.CameraID_Left)
        ret = x1.Open()
        if ret and x1.IsValid():
            print("RVC X 相机有效！")
        else:
            print("RVC X 相机无效！")
            RVC.X1.Destroy(x1)
            RVC.SystemShutdown()
            return -1

        # 打印曝光时间范围
        _, exp_range_min, exp_range_max = x1.GetExposureTimeRange()
        print("曝光时间范围：[{}, {}]".format(exp_range_min, exp_range_max))

        # 连续采集图像
        while True:
            # 捕获图像
            ret = x1.Capture()
            if ret:
                print("RVC X 相机捕获成功！")
            else:
                print("RVC X 相机捕获失败！")
                print(RVC.GetLastErrorMessage())
                break

            # 获取图像数据
            img = x1.GetImage()
            img_np = np.array(img, copy=False)  # 将图像转换为 NumPy 数组
            print(img_np.shape)
            # 转换为灰度图像
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

            # 检测 ArUco 码
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                # 0.1 是 ArUco 码的边长
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist)

                for i in range(rvec.shape[0]):
                    # 绘制 ArUco 码的坐标系
                    cv2.drawFrameAxes(img_np, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.1)
                    aruco.drawDetectedMarkers(img_np, corners)

                    # 将 rvec 转换为旋转矩阵
                    rotation = R.from_rotvec(rvec[i, 0, :])  # 将 rvec 的形状转换为 (3,)
                    euler_angles = rotation.as_euler('xyz', degrees=False)

                    # 显示 ID、rvec、tvec、欧拉角 (弧度制)
                    cv2.putText(img_np, "Id:" + str(ids[i]), (0, 40 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(img_np, "Euler: " + str(euler_angles), (0, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    # 打印位姿信息
                    result = [tvec[i, :, :][0][0], tvec[i, :, :][0][1], tvec[i, :, :][0][2], euler_angles[0], euler_angles[1], euler_angles[2]]
                    rounded_list = [round(num, 5) for num in result]
                    # print("相机获取标定板位姿:",rounded_list)
                    T_camera = self.posToMatrix(rounded_list)
                    # print("相机变换矩阵",T_camera)
                    # print("机械臂变换矩阵",T_end)
                    hand_pose = self.get_pos()
                    T_end = self.posToMatrix(hand_pose)
                    # print("机械臂tcp在base坐标系下的位姿:",hand_pose)
                    T_result = np.linalg.inv(self.T_base2camera)@T_camera
                    # print("aruco在base坐标系下的位姿:",T_result)
                    self.X_delta_sum.append(T_result[0,3]-T_end[0,3])
                    self.Y_delta_sum.append(T_result[1,3]-T_end[1,3])
                    self.Z_delta_sum.append(T_result[2,3]-T_end[2,3])
                    x_variance = np.var(self.X_delta_sum)
                    y_variance = np.var(self.Y_delta_sum)
                    z_variance = np.var(self.Z_delta_sum)
                    x_average = np.mean(self.X_delta_sum)
                    y_average = np.mean(self.Y_delta_sum)
                    z_average = np.mean(self.Z_delta_sum)
                    print("结果方差:",x_variance,"    ",y_variance,"     ",z_variance)
                    print("结果平均值:",x_average,"    ",y_average,"     ",z_average)
            else:
                cv2.putText(img_np, "No Ids", (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 显示图像
            cv2.imshow("RVC X Camera with ArUco Detection", img_np)
            key = cv2.waitKey(1)  # 等待用户按下任意键

            if key == ord('q'):  # 如果按下 'q' 键，退出循环
                print("用户退出采集。")
                break

        # 释放资源
        x1.Close()
        RVC.X1.Destroy(x1)
        RVC.SystemShutdown()

        return 0
    
def main():
    robot = accuracy_test()
    robot.App()
    #设置循环频率
if __name__ == '__main__':
    main()