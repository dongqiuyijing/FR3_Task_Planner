import sys
sys.path.insert(0,"/home/admind/software_ws/devel/lib/python3/dist-packages")
import os
import rospkg
import rospy
import numpy as np
import actionlib
import cv2
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
import cv2.aruco as aruco
# 检查路径
print(cv2.__version__)
print(f"[DEBUG] OpenCV path: {cv2.__file__}")
from scipy.spatial.transform import Rotation as R
import yaml
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0,scripts_path)
import PyRVC as RVC # type: ignore
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
from gui import button
TaskController = button.TaskController
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from std_msgs.msg import Float32MultiArray
# 动态构建scripts目录路径
file_path = os.path.join(pkg_path, 'config/camera_rvc/rvc_IntrinsicParameters.yaml')

class RVC_get_aruco:
    def __init__(self):
        rospy.init_node('rvc_aruco_get',anonymous=True)
        task_controller  = TaskController(task_function = self.get_image)
        server = actionlib.SimpleActionServer('aruco_get', MessageAction, lambda goal: self._execute_callback(goal, server, task_controller),auto_start=False)
        #开启动作执行器
        server.start()
        self.pose_pub = rospy.Publisher('image_pose', Float32MultiArray, queue_size=1)
        self.image_pub = rospy.Publisher('/camera/cropped_image', Image, queue_size=10)
         # 创建 CvBridge 对象
        self.bridge = CvBridge()
        self.camera_init_first = False
        self.hand_eye_text = ["Please mount the calibration board to the end effector of the robotic arm",
                              "ensuring it's fully visible to the camera. Then activate the joystick control",
                              "Please follow these steps carefully",
                              "1: Control the robotic arm to move the calibration board until:",
                              "It's fully within the camera's field of view",
                              "The system successfully detects the board's coordinate axes",
                              "2: When both conditions are met:",
                              "Stop the arm movement immediately",
                              "Press the 'Record Data' button",
                              "3: Repeat this process to collect:",
                              "Minimum 20 valid data sets",
                              "From different positions/orientations",
                              "Ensuring good marker visibility in each",
                              "4: Final step:",
                              "After reaching 20+ recordings",
                              "Initiate the final calibration calculation"]
        # 加载相机内参
        with open(file_path, "r") as file: 
            parameter = yaml.load(file.read(), Loader=yaml.Loader)
            mtx = parameter['camera_matrix']
            dist = parameter['dist_coeff']
            self.mtx = np.array(mtx)
            self.dist = np.array(dist)
        # 初始化相机
        while not self.camera_init_first:
            self.camera_init_first = self.camera_init()
            time.sleep(3)
        # ArUco 字典和参数
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        self.parameters = aruco.DetectorParameters()
        
        
    def camera_init(self):
        # 初始化 RVC X 系统
        RVC.SystemInit()

        # 查找设备
        ret, devices = RVC.SystemListDevices(RVC.SystemListDeviceTypeEnum.All)
        if len(devices) == 0:
            print("未找到任何设备！")
            RVC.SystemShutdown()
            return False
        device = devices[0]

        # 获取设备信息
        ret, info = device.GetDeviceInfo()
        print(info.name + "-" + info.sn)

        # 创建并打开相机
        self.x1 = RVC.X1.Create(device, RVC.CameraID_Left)
        ret = self.x1.Open()
        if ret and self.x1.IsValid():
            print("RVC X 相机有效！")
        else:
            print("RVC X 相机无效！")
            RVC.X1.Destroy(self.x1)
            RVC.SystemShutdown()
            return -1
        # 打印曝光时间范围
        _, exp_range_min, exp_range_max = self.x1.GetExposureTimeRange()
        print("曝光时间范围：[{}, {}]".format(exp_range_min, exp_range_max))
        return True
    def get_image(self):
            # 连续采集图像
        # while True:
        # 捕获图像
        ret = self.x1.Capture()
        if ret:
            pass
            # print("RVC X 相机捕获成功！")
        else:
            print("RVC X 相机捕获失败！")
            print(RVC.GetLastErrorMessage())
            return

        # 获取图像数据
        img = self.x1.GetImage()
        img_np = np.array(img, copy=False)  # 将图像转换为 NumPy 数组

        # 转换为灰度图像
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # 检测 ArUco 码
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            # 0.1 是 ArUco 码的边长
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, self.mtx, self.dist)

            for i in range(rvec.shape[0]):
                # 绘制 ArUco 码的坐标系
                cv2.drawFrameAxes(img_np, self.mtx, self.dist, rvec[i, :, :], tvec[i, :, :], 0.1)
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
                msg = Float32MultiArray(data=rounded_list)
                self.pose_pub.publish(msg)
                # print(rounded_list)
        else:
            cv2.putText(img_np, "No Ids", (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 显示图像
        ros_image = self.bridge.cv2_to_imgmsg(img_np, "bgr8")
        # 添加时间戳
        ros_image.header.stamp = rospy.Time.now()
        # 发布图像
        self.image_pub.publish(ros_image)
    def _execute_callback(self,goal,server, controller):# 行为回调函数
        """带参数的回调函数"""
        feedback = MessageFeedback()
        result = MessageResult() 
        try:
            if goal.message == "Start":
                rospy.loginfo("收到启动请求")
                if self.camera_init_first:
                    for text in self.hand_eye_text:
                        feedback.status = text
                        server.publish_feedback(feedback)
                        time.sleep(0.1)
                    feedback.status = "start success"
                    result.reply = "start success"
                    controller.start_task()
                else:
                    feedback.status = "fail ..."
                    result.reply = "fail!!! unable open the camera ; please connect camera!!!!!"
            elif goal.message == "Stop":
                rospy.loginfo("收到停止请求")
                controller.stop_task()
                feedback.status = "stop"
                result.reply = "stop success"   
            server.publish_feedback(feedback)
            time.sleep(0.5)
            server.set_succeeded(result)  # 明确设置终止状态
        except Exception as e:
            rospy.logerr("处理错误: %s", str(e))
            result.reply = f"错误: {str(e)}"

if __name__ == "__main__":
    rvc = RVC_get_aruco()
    rospy.spin()