import sys
import os

# 获取当前工作目录的绝对路径
path = os.path.abspath(".")
# 将自定义模块路径添加到 Python 的模块搜索路径中
sys.path.insert(0, path + "/src/realsense_control_pkg/scripts")

# Copyright (c) RVBUST, Inc - All rights reserved.
import PyRVC as RVC  # 导入 RVC 相机控制库
import numpy as np   # 导入 NumPy 库，用于数值计算
import cv2           # 导入 OpenCV 库，用于图像处理
import open3d as o3d # 导入 Open3D 库，用于点云可视化
import nav_msgs.msg

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from scipy.spatial import KDTree
import sensor_msgs.point_cloud2 as pc2
import math
import geometry_msgs.msg
import nav_msgs.msg
import tf
import tf.transformations
import rospkg
import time
import os
# 获取当前ROS包的路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# 动态构建scripts目录路径
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0,scripts_path)
from cyl_method import cyl_TF
# 全局变量，用于存储鼠标点击的点
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sklearn.metrics import r2_score
class ImageReceiver:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('image_receiver', anonymous=True)
        self.pub = rospy.Publisher('path_cyl',nav_msgs.msg.Path,queue_size=10)
        # 导入手眼标定结果
        path_hand_eye = os.path.join(pkg_path, 'config/camera_rvc/hand_eye.txt')
        self.T_camera = self._read_matrix_from_txt(path_hand_eye)
        self.bridge = CvBridge()  # OpenCV 到 ROS 图像的转换工具
        self.sub_RVC_image = rospy.Subscriber('/RVC_camera/image', Image, self.image_callback)
        self.sub_pointcloud = rospy.Subscriber('/RVC_camera/pointcloud', PointCloud2,self.pointcloud_callback)
        #两个全局图像
        self.img_display = None
        self.points = []
        #点的间距
        self.points_num = 4 #储存点的数量
        self.th_list = [[],[]]
        self.pointcloud = None
        self.img_display = None
        self.target_height = 1080
        self.target_width = 1440
        self.pose_z = None
        self.point_last = None
        self.cornor_pose =[]
        self.last_pose = []
    def image_callback(self,msg):
        self.points_num = rospy.get_param("points_num")
        width = msg.width
        height = msg.height
        rospy.loginfo(f"图像尺寸: 宽度={width}, 高度={height}")
        #不在接收图片
        # self.sub_RVC_image.unregister()
        # 转换为OpenCV格式
        self.img_display = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("Image", self.img_display)
        cv2.setMouseCallback("Image", self.mouse_callback, self.img_display)  # 设置鼠标回调函数
        print(f"请在图像上点击 {self.points_num} 个点以选择直线...")
        while len(self.points) < self.points_num:  # 等待用户点击 n 个点
            cv2.waitKey(1)
        cv2.destroyAllWindows()  # 关闭图像窗口
    def pointcloud_callback(self, msg):
        """点云数据回调函数"""
        # self.sub_pointcloud.unregister()
        try:
            points = np.array(list(point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=False
            )))
            
            # 2. 重塑为二维结构 (height, width, 3)
            structured_pc = points.reshape(self.target_height, self.target_width, 3)
            
            # 3. 验证尺寸
            if structured_pc.shape != (self.target_height, self.target_width, 3):
                rospy.logwarn(f"尺寸不匹配！期望({self.target_height}, {self.target_width}, 3)，实际{structured_pc.shape}")
                return
            self.pointcloud = {
                'data': structured_pc,
                'info': {
                    'width': msg.width,
                    'height': msg.height,
                    'timestamp': msg.header.stamp.to_sec()
                }
            }
            
        except Exception as e:
            rospy.logerr(f"处理点云时出错: {str(e)}")
    def mouse_callback(self,event, x, y, flags, param):
        """鼠标回调函数，用于记录点击的点"""
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            if len(self.points) < self.points_num:  # 限制选点数量为 n
                self.points.append((x, y))
                print(f"点击点坐标：({x}, {y})")
                # 在图像上绘制点击的点
                cv2.circle(self.img_display, (x, y), 5, (0, 0, 255), -1)  # 绘制红色实心圆点
                if len(self.points) > 1:  # 如果已经点击了至少两个点
                    # 绘制连接所有点的直线
                    for i in range(len(self.points) - 1):
                        cv2.line(self.img_display, self.points[i], self.points[i + 1], (0, 255, 0), 2)  # 绘制绿色直线
                cv2.imshow("Image", self.img_display)  # 更新显示图像
    def _read_matrix_from_txt(self,file_path):
        """
        参数:
            file_path (str): TXT 文件路径

        返回:
            np.ndarray: 矩阵数据（例如 4x4 矩阵）
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 按行分割数据，并转换为浮点数
        matrix = []
        for line in lines:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)

        return np.array(matrix)
    def rotate_vector(self,vector, angle_degrees):
        # 将角度转换为弧度
        angle_radians = np.radians(angle_degrees)
        
        # 旋转矩阵
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])
        
        # 进行矩阵乘法
        rotated_vector = np.dot(rotation_matrix, np.array(vector))
        return rotated_vector
    def vector_angle(self, b):
        a = np.array([0,0,1])
        # 计算点积
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # 计算模长
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        
        # 检查零向量
        if mag_a == 0 or mag_b == 0:
            raise ValueError("向量不能为零向量")
        
        # 处理浮点误差并计算余弦值
        cos_theta = dot_product / (mag_a * mag_b)
        cos_theta = max(min(cos_theta, 1.0), -1.0)  # 确保在[-1, 1]范围内
        
        # 计算弧度与角度
        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)
        print("法向量夹角：",theta_deg)
        return theta_deg
    def angle_between_vectors(self,v1, v2):
        # 计算点积
        dot_product = np.dot(v1, v2)
        # 计算模长
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        # 计算夹角（弧度）
        angle_rad = np.arccos(dot_product / (v1_norm * v2_norm))

        # 转换为度
        angle_deg = np.degrees(angle_rad)

        # 使用 atan2 判断象限
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product < 0:
            angle_deg = 360 - angle_deg

        return angle_deg
    def path_form(self,normals,points):
        path_msgs = nav_msgs.msg.Path()
        path_msgs.header.frame_id = "base_link"
        path_msgs.header.stamp = rospy.Time.now()
        # print(len(points))
        # print(self.th_list)
        #用来获取角度索引
        j = 0
        sum = self.th_list[0][j]
        print("结构",sum)
        for i,(point, normal) in enumerate(zip(points,normals)): 
            if i < sum-1:
                dir_ant = self.th_list[1][j]
            else :
                if j < len(self.th_list[0])-1:
                    j+=1
                    sum += self.th_list[0][j]
            # print(j)
            pose_goal = geometry_msgs.msg.PoseStamped()
            pose_goal.header.frame_id = "base_link"
            pose_goal.header.stamp = rospy.Time.now()
            pose_goal.pose = self.pose_form(point,normal,dir_ant)
            path_msgs.poses.append(pose_goal)#将位姿信息按照顺序放入
        self.th_list = [[],[]]
        self.pub.publish(path_msgs)
    def pose_form(self,pos,dir,dir_ant):#需要将相机的坐标先转换到机械臂末端在转化到基座坐标系
        #定义一个位姿信息作为返回参数
        pose_goal = geometry_msgs.msg.Pose()
        dir = np.asarray(dir)#改成矩阵
		#定义一个位姿数据，并赋值位置信息
        norm = np.linalg.norm(dir) #求模
        if norm == 0 :
            print("方向向量出错")
            return 0
        dir = dir/norm #单位化

        # dir_ant = np.array([x,y,0]) #固定角度运行

        if np.all(np.cross(dir,dir_ant)==0):
            print("向量平行，重新选取")
            return
        all_direction = rospy.get_param("all_direction")
        # if all_direction:
        x_axis = np.cross(dir,dir_ant) #选取X
        x_axis = x_axis /np.linalg.norm(x_axis) #求模单位化
        y_axis = np.cross(dir,x_axis)
        # else:
        #     x_axis = np.cross(dir,dir_ant) #选取X
        #     x_axis = x_axis /np.linalg.norm(x_axis) #求模单位化
        #     # y_axis = dir_ant /np.linalg.norm(dir_ant) #求模单位化
        #     dir = np.cross(x_axis,y_axis)
       
        #写旋转矩阵，Z轴分向量就是输入的方向向量
        R = np.array([[x_axis[0],y_axis[0],dir[0]],[x_axis[1],y_axis[1],dir[1]],[x_axis[2],y_axis[2],dir[2]]])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = np.array(pos)
        #转化到机械臂末端
        T_world = self.T_camera @ T
        if np.allclose(pos, self.cornor_pose[0]) :
            self.cornor_pose.pop(0)

        pose = T_world[:3,3]
        quaternion = tf.transformations.quaternion_from_matrix(T_world)
        euler_angles = tf.transformations.euler_from_quaternion(quaternion)
        euler_angles_list = list(euler_angles)
        if all_direction:
            pass
        else :
            euler_angles_list[0] = math.pi
            euler_angles_list[1] = 0
        euler_angles = tuple(euler_angles_list)
        # print("角度偏移",euler_angles)
        T_world = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angles))
        
        quaternion = tf.transformations.quaternion_from_matrix(T_world)
        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        z_height = rospy.get_param("z_height")
        pose_goal.position.z = pose[2]+0.012 + z_height
        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]
        return pose_goal
    def sliding_window_filter(self,normals, window_size=5):#平滑法向量
        filtered_normals = np.copy(normals)
        half_window = window_size // 2
        for i in range(len(normals)):
            # 计算窗口的范围
            start = max(0, i - half_window)
            end = min(len(normals), i + half_window + 1)
            # 计算窗口内法向量的平均值
            filtered_normals[i] = np.mean(normals[start:end], axis=0)
        return filtered_normals
    def polynomial_fit(self,y_data, max_degree=4):
        """
        将离散数据拟合成多项式函数，并返回拟合后的数据
        
        参数:
            y_data: list/array - 一维y值数据
            max_degree: int - 最大尝试的多项式阶数(默认为4)
        
        返回:
            predicted_y: list - 拟合后的y值
        """
        y = np.array(y_data)
        x = np.arange(len(y))
        
        # 存储最佳结果
        best_prediction = None
        best_degree = 0
        min_error = np.inf  # 最小残差平方和
        
        for degree in range(1, max_degree + 1):
            try:
                # 多项式拟合
                coeffs = np.polyfit(x, y, degree)
                poly = np.poly1d(coeffs)
                y_pred = poly(x)
                
                # 计算残差平方和
                error = np.sum((y - y_pred) ** 2)
                
                # 选择误差最小的阶数
                if error < min_error:
                    min_error = error
                    best_degree = degree
                    best_prediction = y_pred
            except:
                continue
        
        print(f"最佳拟合阶数: {best_degree}, 残差平方和: {min_error:.2f}")
        return best_prediction.tolist()
    def find_parallel_points(self,p1, p2, distance=2):
        """
        输入两个点，返回右侧和左侧偏移后的两个点（整数坐标）
        
        参数:
            p1, p2: 元组/列表，表示两个二维点，如 (x1,y1), (x2,y2)
            distance: 偏移距离（默认2）
        
        返回:
            (right_points, left_points) - 每个都是 (point1, point2) 的元组
        
        异常:
            当输入点重合时会抛出ValueError
        """
        # print(p1,p2)
        # 转换为numpy数组
        A = np.array(p1)
        B = np.array(p2)
        # 计算方向向量和垂直向量
        direction = B - A
        perp = np.array([-direction[1], direction[0]])  # 逆时针旋转90度
        
        # 检查点是否重合
        if np.linalg.norm(direction) < 1e-6:
            raise ValueError("输入点重合，无法确定方向")
        
        # 归一化垂直向量并缩放到目标距离
        perp_unit = perp / np.linalg.norm(perp)
        scaled_perp = perp_unit * distance
        
        # 计算偏移点（四舍五入到整数）
        right_A = np.round(A + scaled_perp).astype(int)
        right_B = np.round(B + scaled_perp).astype(int)
        left_A = np.round(A - scaled_perp).astype(int)
        left_B = np.round(B - scaled_perp).astype(int)
        
        # 返回结果元组
        return [right_A, right_B], [left_A, left_B]
    def Path_Discretization(self,pose1,pose2,pointcloud,num_points):
        points_list = []
        normal_list = []
        sum_points = num_points
        # self.th_list[0].append(num_points)
        for j in range(num_points + 1):
            x = int(pose1[0] + (pose2[0] - pose1[0]) * j / num_points)
            y = int(pose1[1] + (pose2[1] - pose1[1]) * j / num_points)
            if 0 <= x < self.target_width and 0 <= y < self.target_height:
                point = pointcloud[y, x]  # 获取点云中的点
                normal = -self.calculate_normal_pose_one(pointcloud,y,x)
                if (not np.isnan(point).any()) and (not np.isnan(normal).any()):  # 忽略无效点和法向量
                    # print("遇到了无效点",point)
                    # if self.pose_z is None :
                    #     self.pose_z = point[2]
                    # else:
                    #     # if abs(self.pose_z - point[2])>0.002:
                    #     #     point[2] = self.pose_z
                    #         # print("高度不符合的z值", self.pose_z , point[2])
                    #     # else :
                    #     #     self.pose_z = point[2]
                    points_list.append(point)
                    normal_list.append(normal)
                else:
                    sum_points -= 1
                    pass
            if j ==0 or j == num_points:
                self.cornor_pose.append(point)
            # print("数据",j,num_points)
        self.th_list[0].append(sum_points)
        return points_list , normal_list
    def normal_calculate(self,curve1,curve2,aim_curve):
        normals = []
        first_point = aim_curve[0].copy()  # 使用copy避免视图问题
        pose1 = curve1[1]
        pose2 = curve2[1]
        dir1 = pose1 - first_point
        dir2 = pose2 - first_point
        first_normal = np.cross(dir1, dir2)  # 计算平面法向量
        first_normal /= np.linalg.norm(first_normal)  # 归一化平面法向量
        normals.append(first_normal)
        # 移除第一个点（创建新数组）
        aim_curve = aim_curve[1:]
        curve1 = curve1[:len(curve1)-1]
        curve2 = curve2[:len(curve2)-1]
        # print("长度",len(curve1),len(curve2),len(aim_curve))
        for pose1 ,pose2,pose_aim in zip(curve1,curve2,aim_curve):
            dir1 = pose1 - pose_aim
            dir2 = pose2 - pose_aim
            pos_normal = np.cross(dir1, dir2)  # 计算平面法向量
            pos_normal /= np.linalg.norm(pos_normal)  # 归一化平面法向量
            normals.append(pos_normal)
            # print("法向量",pos_normal)
        return normals
    def sliding_window_filter(self,normals: np.ndarray, window_size=5):#平滑法向量
        """平滑法向量函数
        输入:法向量矩阵
            平滑参数，越大越平均
        输出:平滑后的法向量
        """
        filtered_normals = np.copy(normals)
        half_window = window_size // 2
        for i in range(len(normals)):
            # 计算窗口的范围
            start = max(0, i - half_window)
            end = min(len(normals), i + half_window + 1)
            # 计算窗口内法向量的平均值
            filtered_normals[i] = np.mean(normals[start:end], axis=0)
        return filtered_normals
    def calculate_normal_pose_one(self,points_of_cloud,x,y):
        #x轴偏离点
        pos1 = points_of_cloud[x+1,y]
        #y轴偏离点
        pos2 = points_of_cloud[x,y+1]
        #目标点
        pos_aim = points_of_cloud[x,y]
        dir1 = pos1 - pos_aim
        dir2 = pos2 - pos_aim
        pos_normal = np.cross(dir1, dir2)  # 计算平面法向量
        pos_normal /= np.linalg.norm(pos_normal)  # 归一化平面法向量
        return pos_normal
    def run(self):
        while not rospy.is_shutdown():
            if  (not self.pointcloud is None) and (not self.img_display is None) and (not len(self.points) != self.points_num):
                break
            # print("未获得图片点云数据！")
            time.sleep(2)
        # 提取多条直线上的点云数据和法向量
        line_points = []
        line_normals = []
        pointcloud = self.pointcloud['data']
        for i in range(len(self.points) - 1):  # 遍历每两个相邻点
            start_point = [self.points[i][0], self.points[i][1]] # 起点
            end_point = [self.points[i + 1][0],self.points[i + 1][1]]  # 终点
            right , left = self.find_parallel_points(start_point,end_point,2)
            #获取方向
            line_direction = np.array(end_point) - np.array(start_point)
            # #更换工件的默认初始角度
            line_direction[:2] = self.rotate_vector(line_direction[:2],45)

            # print("初始方向",line_direction)
            th = self.angle_between_vectors(line_direction[:2], np.array([-50 ,-48]))
            print("当前角度",th)
            if th < 90 or th > 270 :
                line_direction = -line_direction
            self.th_list[1].append(line_direction)

            num_points = max(abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1]))  # 直线上的点数

            # self.th_list[0].append(num_points)
            # 将图片中的两个点离散能多个点云的点
            points_list_1,normals_list_1  = self.Path_Discretization(start_point,end_point,pointcloud,num_points)
            normals_list_1 = self.sliding_window_filter(normals_list_1, window_size=31)
            points_list_1 = np.array(points_list_1)
            fit_line = rospy.get_param("fit_line")
            if fit_line:
                    points_list_1[:,2] = self.polynomial_fit(points_list_1[:,2],20) # 多次函数拟合
            line_points.extend(points_list_1)
            line_normals.extend(normals_list_1)
        # 将直线上的点云数据和法向量转换为 Open3D 点云对象
        line_points = np.array(line_points)
        line_normals = np.array(line_normals)
        # 对投影后的法向量进行滑动平均滤波
        line_normals = self.sliding_window_filter(line_normals, window_size=31)
        # 利用话题发布路径
        self.path_form(line_normals,line_points)
        line_pcd = o3d.geometry.PointCloud()
        line_pcd.points = o3d.utility.Vector3dVector(line_points)  # 设置点云数据
        line_pcd.normals = o3d.utility.Vector3dVector(line_normals)  # 设置法向量数据
        # 显示直线上的点云和法向量
        o3d.visualization.draw_geometries([line_pcd], window_name="Line Points in Point Cloud with Normals",
                                            point_show_normal=True)  # 显示点云和法向量
        # 重置
        self.pointcloud = None
        self.img_display = None
        self.points = []


if __name__ == '__main__':
    try:
        receiver = ImageReceiver()
        while not rospy.is_shutdown():
            receiver.run()
        # receiver.App()
    except rospy.ROSInterruptException:
        pass