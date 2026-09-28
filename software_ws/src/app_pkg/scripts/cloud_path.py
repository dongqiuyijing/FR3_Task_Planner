# Copyright (c) RVBUST, Inc - All rights reserved.
import PyRVC as RVC
import os
import numpy as np
import cv2
import sys
from scipy.spatial import KDTree
import open3d as o3d
import geometry_msgs.msg
import nav_msgs.msg
import tf
import math
import rospy
class RVCameraCloudPoint:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('RVCameraCloudPoint', anonymous=True)
        RVC.SystemInit()
        self.opt = RVC.SystemListDeviceTypeEnum.All
        self.ret, self.devices = RVC.SystemListDevices(self.opt)
        print("RVC X Camera devices number:%d" % len(self.devices))
        if len(self.devices) == 0:
            print("Can not find any RVC X Camera!")
            RVC.SystemShutdown()
            raise Exception("No RVC X Camera found")
        print("devices size = %d" % len(self.devices))
        self.x = RVC.X1.Create(self.devices[0], RVC.CameraID_Left)
        if not self.x.IsValid():
            print("RVC X Camera is not valid!")
            RVC.X1.Destroy(self.x)
            RVC.SystemShutdown()
            raise Exception("RVC X Camera is not valid")
        print("RVC X Camera is valid!")
        self.ret1 = self.x.Open()
        if not (self.ret1 and self.x.IsOpen()):
            print("RVC X Camera is not opened!")
            RVC.X1.Destroy(self.x)
            RVC.SystemShutdown()
            raise Exception("RVC X Camera is not opened")
        print("RVC X Camera is opened!")
        _, self.exp_range_min, self.exp_range_max = self.x.GetExposureTimeRange()
        print("ExposureTime Range:[{}, {}]".format(self.exp_range_min, self.exp_range_max))
        self.pub = rospy.Publisher('path_cyl',nav_msgs.msg.Path,queue_size=10)
        self.th_list = [[],[]]
        self.T_camera = np.array( [[ 0.00974247 ,-0.99797327 ,-0.06288425 ,-0.36906707],
                                [-0.99809087 ,-0.00586945 ,-0.06148311 ,-0.02513205],
                                [ 0.0609894   ,0.0633632  ,-0.99612519  ,0.83221417],
                                [ 0.          ,0.          ,0.          ,1.        ]])

    def calculate_normal(self,path_point_xy,kdtree_xy,points_of_cloud):
        #x轴偏离点
        dist, idx = kdtree_xy.query([path_point_xy[0]+0.003,path_point_xy[1]])
        pos1 = points_of_cloud[idx]
        #y轴偏离点
        dist, idx = kdtree_xy.query([path_point_xy[0],path_point_xy[1]+0.003])
        pos2 = points_of_cloud[idx]
        #目标点
        dist, idx = kdtree_xy.query([path_point_xy[0],path_point_xy[1]])
        pos_aim = points_of_cloud[idx]
        dir1 = pos1 - pos_aim
        dir2 = pos2 - pos_aim
        pos_normal = np.cross(dir1, dir2)  # 计算平面法向量
        pos_normal /= np.linalg.norm(pos_normal)  # 归一化平面法向量
        return pos_normal
    def capture_and_show_point_cloud(self):
        # Capture a point map and a image.
        ret2 = self.x.Capture()
        if ret2:
            print("RVC X Camera capture successed!")
            # Get point map data (m).
            pm = self.x.GetPointMap()
            pm_np = np.array(pm, copy=False).reshape(-1, 3)
            # 去除无效点（NaN）
            valid_indices = ~np.isnan(pm_np).any(axis=1)
            pm_np = pm_np[valid_indices]
            # 创建 Open3D 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pm_np)
            #获取点云的所有点
            points = np.asarray(pcd.points)
            # 显示点云并使用 open3d 的可视化工具手动选择点
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Point Cloud Visualization", width=1200, height=900, left=50, top=50)
            
            vis.add_geometry(pcd)
            
            vis.run()  # 用户可以在此选择点，按住 Shift + 左键选择点，按 Q 退出
            vis.destroy_window()
            # 获取用户选择的点索引
            picked_indexes = vis.get_picked_points()
            self.cloud_path_form(picked_indexes,points)
        else:
            print("RVC X Camera capture failed!")
            print(RVC.GetLastErrorMessage())
        # Close RVC X Camera.
        self.x.Close()
        # Destroy RVC X Camera.
        RVC.X1.Destroy(self.x)
        # Shutdown RVC X System.
        RVC.SystemShutdown()
        cv2.destroyAllWindows()
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
        #用来获取角度索引
        j = 0
        sum = self.th_list[1][0]
        print("长度",len(self.th_list[0]))
        for i,(point, normal) in enumerate(zip(points,normals)): 
            if i < sum-1:
                dir_ant = self.th_list[0][j]
            else :
                j+=1
                if j < len(self.th_list[1]):
                    sum += self.th_list[1][j]
            pose_goal = geometry_msgs.msg.PoseStamped()
            pose_goal.header.frame_id = "base_link"
            pose_goal.header.stamp = rospy.Time.now()
            pose_goal.pose = self.pose_form(point,normal,dir_ant)
            path_msgs.poses.append(pose_goal)#将位姿信息按照顺序放入
        self.th_list = [[],[]]
        self.pub.publish(path_msgs)
    def pose_form(self,pos,normal,dir_ant):#需要将相机的坐标先转换到机械臂末端在转化到基座坐标系
        #定义一个位姿信息作为返回参数
        pose_goal = geometry_msgs.msg.Pose()
        normal = np.asarray(normal)#改成矩阵
		#定义一个位姿数据，并赋值位置信息
        norm = np.linalg.norm(normal) #求模
        if np.isnan(norm):
            normal = np.array([0,0,1])
            print("方向向量出错")
            # return 0
        normal = normal/norm #单位化
        if np.all(np.cross(normal,dir_ant)==0):
            print("向量平行，重新选取")
            return
        x_axis = np.cross(normal,dir_ant) #选取X
        x_axis = x_axis /np.linalg.norm(x_axis) #求模单位化
        y_axis = np.cross(normal,x_axis)
        #写旋转矩阵，Z轴分向量就是输入的方向向量
        R = np.array([[x_axis[0],y_axis[0],normal[0]],[x_axis[1],y_axis[1],normal[1]],[x_axis[2],y_axis[2],normal[2]]])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = np.array(pos)
        #转化到机械臂末端
        T_world = self.T_camera @ T
        pose = T_world[:3,3]
        quaternion = tf.transformations.quaternion_from_matrix(T_world)
        euler_angles = tf.transformations.euler_from_quaternion(quaternion)
        euler_angles_list = list(euler_angles)
        euler_angles_list[0] = math.pi
        euler_angles_list[1] = 0
        euler_angles = tuple(euler_angles_list)
        T_world = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angles))
        quaternion = tf.transformations.quaternion_from_matrix(T_world)
        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        pose_goal.position.z = pose[2]+0.033  #小探头：0.005
        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]
        return pose_goal
    def cloud_path_form(self,picked_indexes,points_of_cloud):
        # 投影点云的点到 XY 平面,方便后面选取
        points_xy = points_of_cloud[:, :2]  # 忽略 Z 坐标，只保留 X 和 Y
        # 使用 KDTree 查找投影到 XY 平面上最近的点
        kdtree_xy = KDTree(points_xy)
        # 用于存储路径上最近的点和投影后的法向量
        path_points_with_normals = []
        last_pose = None
        for i in range(len(picked_indexes)-1):
            now_pos = points_of_cloud[picked_indexes[i]]
            next_pos = points_of_cloud[picked_indexes[i+1]]
            # 打印选中的两点坐标
            print(f"选中的点坐标: {now_pos}")
            print(f"选中的点坐标: {next_pos}")
            #求解方向
            line_direction = np.array(next_pos) - np.array(now_pos)
            print("初始方向",line_direction)
            th = self.angle_between_vectors(line_direction[:2], np.array([-0.09423062 , -0.00015984]))
            print(th)
            #更换工件的默认初始角度
            line_direction[:2] = self.rotate_vector(line_direction[:2],0)
            if th < 90 or th > 270:
                line_direction = -line_direction
            #将路径方向存入一个列表里保存
            self.th_list[0].append(line_direction)
            #计算这条路径上有几个点，每个点的间隔为一毫米
            num = np.linalg.norm(line_direction)//0.001
            #将点数量的数量值也保存到同一个列表的另外一行
            self.th_list[1].append(int(num))
            # 生成插值路径
            path_points = np.linspace(now_pos, next_pos, int(num))
            # 获取点云上理论值的路径
            path_points_xy = path_points[:, :2]  # 忽略 Z 坐标，只保留 X 和 Y
            # 使用 KDTree 查找投影到 XY 平面上最近的点
            for path_point_xy in path_points_xy:
                # 查找最近的点（在 XY 平面上）
                dist, idx = kdtree_xy.query(path_point_xy)
                closest_point = points_of_cloud[idx]
                #过滤长距离的点
                if last_pose is None:
                    last_pose = closest_point
                # print("z轴偏差",math.fabs(last_pose[2] - closest_point[2]))
                closest_normal = self.calculate_normal(path_point_xy , kdtree_xy , points_of_cloud)
                if math.fabs(last_pose[2] - closest_point[2]) > 0.005:
                    print("Z轴偏差过大")
                    closest_point[2] = last_pose[2]
                #重新赋值
                last_pose = closest_point
                nan_mask = np.isnan(closest_normal)
                if np.any(nan_mask):
                    try:
                        closest_normal = self.last_normal
                    except NameError:
                        print("点选得太偏")
                        return
                else :
                    self.last_normal =  closest_normal
                path_points_with_normals.append(np.hstack((closest_point, closest_normal)))
        # 将点和投影后的法向量转换为 NumPy 数组
        path_points_with_normals = np.array(path_points_with_normals)
        # 对投影后的法向量进行滑动平均滤波
        smoothed_normals = self.sliding_window_filter(path_points_with_normals[:, 3:], window_size=31)
        # 创建路径的点云对象，使用平滑后的法向量
        path_point_cloud = o3d.geometry.PointCloud()
        path_point_cloud.points = o3d.utility.Vector3dVector(path_points_with_normals[:, :3])
        path_point_cloud.normals = o3d.utility.Vector3dVector(smoothed_normals)
        self.path_form(smoothed_normals,path_points_with_normals[:, :3])
        # 可视化生成的路径及其法向量
        o3d.visualization.draw_geometries([path_point_cloud],
                                        window_name="Interpolated Path with Smoothed Normals",
                                        width=800, height=600,
                                        point_show_normal=True)
if __name__ == "__main__":
    try:
        camera_handler = RVCameraCloudPoint()
        camera_handler.capture_and_show_point_cloud()
    except Exception as e:
        print(f"An error occurred: {e}")
