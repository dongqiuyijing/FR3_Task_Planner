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
import PyRVC as RVC
# 加载相机内参
file_path = "/home/admind/control_ws/src/realsense_control_pkg/config/标定文件.yaml"
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
        self.X_sum = []
        self.Y_sum = []
        self.Z_sum = []
        self.T_camera2end = np.array([
            [ 0.6873346061 ,-0.7263209625, -0.0053850495 , 0.0388776228],
            [ 0.7263383021,  0.6873331579 , 0.0024085187,-0.0845730666],
            [ 0.0019519655 ,-0.005566826  , 0.9999826 ,-0.1226924352],
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
        # print("机械臂位姿:",rounded_list)
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
        # 打开笔记本摄像头,笔记本内置摄像头为0，USB外接摄像头为1或2，深度图或RGB

        cap = cv2.VideoCapture(4)  # 或者使用正确的设备索引
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        cap.set(3,1280)
        cap.set(4,720)
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
        # num = 0
        while True:
            start = time.time()
            ret, frame = cap.read()
            # operations on the frame come here

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)    #修改成自己需求格式的Aruco二维码图片
            parameters = aruco.DetectorParameters()

            # lists of ids and the corners beloning to each id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                                aruco_dict,
                                                                parameters=parameters)
            if ids is not None:
                # 0.1是arucco码的边长
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error

                for i in range(rvec.shape[0]):
                    cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.1)
                    aruco.drawDetectedMarkers(frame, corners)

                    # 将 rvec 转换为旋转矩阵
                    rotation = R.from_rotvec(rvec[i, 0, :])  # 将 rvec 的形状转换为 (3,)
                    euler_angles = rotation.as_euler('xyz', degrees=False)

                    # 显示ID，rvec, tvec, 欧拉角 (弧度制)
                    cv2.putText(frame, "Id:" + str(ids[i]), (0, 40 + i * 20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "Euler: " + str(euler_angles), (0, 60 + i * 20), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    result = [tvec[i, :, :][0][0],tvec[i, :, :][0][1],tvec[i, :, :][0][2],euler_angles[0],euler_angles[1],euler_angles[2]]
                    rounded_list = [round(num, 5) for num in result]
                    print(rounded_list)
                    time.sleep(0.3)
                                        # print("相机获取标定板位姿:",rounded_list)
                    hand_pose = self.get_pos()
                    T_camera = self.posToMatrix(rounded_list)
                    T_end = self.posToMatrix(hand_pose)
                    # print("相机变换矩阵",T_camera)
                    # print("机械臂变换矩阵",T_end)
                    T_result = np.linalg.inv(T_end)@np.linalg.inv(self.T_camera2end)@T_camera
                    print("结果：",T_result)
                    self.X_sum.append(T_result[0,3])
                    self.Y_sum.append(T_result[1,3])
                    self.Z_sum.append(T_result[2,3])
                    x_average = sum(self.X_sum)/len(self.X_sum)
                    y_average = sum(self.Y_sum)/len(self.Y_sum)
                    z_average = sum(self.Z_sum)/len(self.Z_sum)
                    x_variance = sum((self.X_sum-x_average)*2)/len(self.X_sum)
                    y_variance = sum((self.Y_sum-y_average)*2)/len(self.Y_sum)
                    z_variance = sum((self.Z_sum-z_average)*2)/len(self.Z_sum)
                    print("结果方差:",x_variance,"    ",y_variance,"     ",z_variance)
            else:
                cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                end = time.time()
    # 计算帧率并显示
            cv2.putText(frame, "rate: " + str(1 / (end - start)), (10, 120), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("frame", frame)

            key = cv2.waitKey(1)

            if key == 27:  # 按esc键退出
                print('esc break...')
                cap.release()
                cv2.destroyAllWindows()
                break
            if key == ord(' '):  # 按空格键保存
                #        num = num + 1
                #        filename = "frames_%s.jpg" % num  # 保存一张图像
                filename = str(time.time())[:10] + ".jpg"
                cv2.imwrite(filename, frame)
    
def main():
    robot = accuracy_test()
    robot.App()
    #设置循环频率
if __name__ == '__main__':
    main()