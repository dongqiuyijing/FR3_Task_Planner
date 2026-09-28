import PyRVC as RVC  # type: ignore # 导入 RVC 相机控制库
import os
import cv2           # 导入 OpenCV 库，用于图像处理
import numpy as np
import rospkg
import rospy
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import actionlib
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
from std_msgs.msg import Header
import open3d as o3d # 导入 Open3D 库，用于点云可视化
from sensor_msgs.msg import PointCloud2, PointField
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# 动态构建scripts目录路径
file_path = os.path.join(pkg_path, 'scripts')

class RVC_camera :
    def __init__(self):
        rospy.init_node('RVC_camera', anonymous=True)
        self.RVC_init()
        self.image_pub = rospy.Publisher('/RVC_camera/image', Image, queue_size=10)
        self.cloud_points_pub = rospy.Publisher('/RVC_camera/pointcloud', PointCloud2, queue_size=10)
        self.bridge = CvBridge()  # OpenCV 到 ROS 图像的转换工具
        self.width = 0  #相机图片的宽度和长度
        self.height = 0
        server = actionlib.SimpleActionServer('photo_get',  MessageAction, lambda goal: self._execute_callback(goal,server) ,auto_start=False)
        #开启动作执行器
        server.start()
    def _execute_callback(self,goal,server):
        feedback = MessageFeedback()
        result = MessageResult() 
        self.get_image()
        self.get_pointCloud()
        feedback.status = "get_image..."
        result.reply = "successful"
        server.publish_feedback(feedback)
        time.sleep(0.1)
        server.set_succeeded(result)  # 明确
    def RVC_init(self):
        RVC.SystemInit()
        # 选择要扫描的 RVC X 相机类型（USB、GigE 或全部）
        opt = RVC.SystemListDeviceTypeEnum.All
        # 扫描所有 RVC X 相机设备
        ret, devices = RVC.SystemListDevices(opt)
        print("扫描到的 RVC X 相机设备数量：%d" % len(devices))
        # 检查是否有 RVC X 相机连接
        if len(devices) == 0:
            print("未找到任何 RVC X 相机！")
            RVC.SystemShutdown()  # 关闭系统
            return 1
        print("设备数量：%d" % len(devices))
        # 创建一个 RVC X 相机实例，并选择使用左侧相机
        camera_rvc = RVC.X1.Create(devices[0], RVC.CameraID_Left)
        # 检查 RVC X 相机是否有效
        if camera_rvc.IsValid() == True:
            print("RVC X 相机有效！")
        else:
            print("RVC X 相机无效！")
            RVC.X1.Destroy(camera_rvc)  # 销毁相机实例
            RVC.SystemShutdown()  # 关闭系统
            return 1
        # 打开 RVC X 相机
        ret1 = camera_rvc.Open()
        # 检查相机是否成功打开
        if ret1 and camera_rvc.IsOpen() == True:
            print("RVC X 相机已打开！")
        else:
            print("RVC X 相机未打开！")
            RVC.X1.Destroy(camera_rvc)  # 销毁相机实例
            RVC.SystemShutdown()  # 关闭系统
            return 1
        # 获取相机的曝光时间范围并打印
        _, exp_range_min, exp_range_max = camera_rvc.GetExposureTimeRange()
        print("曝光时间范围：[{}, {}]".format(exp_range_min, exp_range_max))
        CONFIG_FILE = os.path.join(pkg_path, 'config/camera_rvc/rvc_parameter.json')
        if not self.load_config_and_capture(camera_rvc, CONFIG_FILE):
            print("配置加载失败")
            return 1
        self.camera_rvc = camera_rvc
    def load_config_and_capture(self,x, config_path):
            """方法四：从配置文件加载参数并捕获数据"""
            if not os.path.exists(config_path):
                print(f"错误：配置文件 {config_path} 不存在！")
                return False
            if not x.LoadSettingFromFile(config_path):
                print(f"参数加载失败: {RVC.GetLastErrorMessage()}")
                return False
            print("成功加载配置文件参数！")
            # 执行捕获
            return x.Capture()
    def get_image(self): #获取相机图形
        """返回相机图形，图片的宽度，图片的高度"""
        ret2 = self.camera_rvc.Capture()
        if ret2 == True:
            print("RVC X 相机捕获成功！")
            # 获取图像数据
            img = self.camera_rvc.GetImage()
            self.width = img.GetSize().cols  # 获取图像宽度
            self.height = img.GetSize().rows  # 获取图像高度
            print("图像宽度=%d, 图像高度=%d" % (self.width,self.height))
            # 将图像转换为 NumPy 数组
            img_display = np.array(img, copy=False) # 保存下来的图像
            if img.GetType() == RVC.ImageTypeEnum.Mono8:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像
            ros_image = self.bridge.cv2_to_imgmsg(img_display, encoding="bgr8")
            self.image_pub.publish(ros_image)
            return img_display
    def get_pointCloud(self):
        """返回点云"""
        # 获取点云数据
        pm = np.array(self.camera_rvc.GetPointMap(), copy=False).reshape(-1, 3)
        pm = pm.reshape((self.height, self.width, 3))  # 将点云数据重塑为 3D 数组
        # 将点云数据转换为 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pm.reshape(-1, 3))  # 设置点云数据
        self._publish_pointcloud(pm)
        return pm
    def _publish_pointcloud(self, points_np):
        """
        将 NumPy 点云数组发布为 ROS PointCloud2 消息
        参数:
            points_np: np.ndarray, 形状为 [H, W, 3] 的点云数据
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_depth_optical_frame"  # 根据实际坐标系修改

        # 定义字段（x, y, z）
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        # 转换为字节流
        points_flat = points_np.reshape(-1, 3).astype(np.float32)
        data = points_flat.tobytes()
        # 构建消息
        cloud_msg = PointCloud2(
            header=header,
            height=points_np.shape[0],
            width=points_np.shape[1],
            is_dense=False,
            fields=fields,
            point_step=12,  # 3 * float32
            row_step=12 * points_np.shape[1],
            data=data
        )
        self.cloud_points_pub.publish(cloud_msg)
    def IntrinsicParameters(self):

        # Initialize RVC X system.
        RVC.SystemInit()

        # Choose RVC X Camera type (USB, GigE or All)
        opt = RVC.SystemListDeviceTypeEnum.All

        # Scan all RVC X Camera devices.
        ret, devices = RVC.SystemListDevices(opt)
        print("RVC X Camera devices number:%d" % len(devices))

        #  Find whether any RVC X Camera is connected or not.
        if len(devices) == 0:
            print("Can not find any RVC X Camera!")
            RVC.SystemShutdown()
            return 1
        print("devices size = %d" % len(devices))

        # Create a RVC X Camera and choose use left side camera.
        x = RVC.X1.Create(devices[0], RVC.CameraID_Left)

        # Test RVC X Camera is valid or not.
        if x.IsValid() == True:
            print("RVC X Camera is valid!")
        else:
            print("RVC X Camera is not valid!")
            RVC.X1.Destroy(x)
            RVC.SystemShutdown()
            return 1
        #PrintCaptureMode(devices[0])

        x.Open()

        if x.IsOpen() == False:
            print("RVC X Camera is not opened!")
            RVC.X1.Destroy(x)
            RVC.SystemShutdown()
            exit(1)


        ret, T_extrinsic = x.GetExtrinsicMatrix()
        if ret == True:
            print("\nExtrinsicMatrix:")
            print(T_extrinsic)

        ret, T_intrinsic_matrix, T_distortion = x.GetIntrinsicParameters()
        if ret == True:
            print("\nIntrinsicParameters:")
            print(T_intrinsic_matrix)
            print("\ndistortion:")
            print(T_distortion)

        # Close RVC X Camera.
        x.Close()

        # Destroy RVC X Camera.
        RVC.X1.Destroy(x)

        # Shutdown RVC X System.
        RVC.SystemShutdown()

        return 0
def main():
    rvc = RVC_camera()
    rospy.spin()
    # while not rospy.is_shutdown():
    #     rvc.get_image()
    #     rvc.get_pointCloud()
    #     time.sleep(2)
if __name__ == "__main__":
    main()
