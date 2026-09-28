import os
import rospkg
import sys
# 获取当前ROS包的路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# 动态构建scripts目录路径
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0,scripts_path)
# print(sys.path)
import PyRVC as RVC  # 导入 RVC 相机控制库
import numpy as np   # 导入 NumPy 库，用于数值计算
import cv2           # 导入 OpenCV 库，用于图像处理
import open3d as o3d # 导入 Open3D 库，用于点云可视化
import nav_msgs.msg
import rospy
from sensor_msgs.msg import Image
import math
import geometry_msgs.msg
import nav_msgs.msg
import tf
from std_msgs.msg import Int32
import tf.transformations
import time
# 全局变量，用于存储鼠标点击的点
from cyl_method import RVC_camera
from cyl_method import calculate
import actionlib
from std_msgs.msg import String
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult # type: ignore
class ImageReceiver:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('image_receiver', anonymous=True)
        self.image_pub = rospy.Publisher('camera/image_processed', Image, queue_size=10)
        self.path_pub = rospy.Publisher('path_cyl',nav_msgs.msg.Path,queue_size=10)
        self.pub = rospy.Publisher('integer_topic', Int32, queue_size=10)
        self.T_camera = np.array([[ 0.00974247 ,-0.99797327 ,-0.06288425 ,-0.36906707],
                                [-0.99809087 ,-0.00586945 ,-0.06148311 ,-0.02513205],
                                [ 0.0609894   ,0.0633632  ,-0.99612519  ,0.83221417],
                                [ 0.          ,0.          ,0.          ,1.        ]])
        time.sleep(2)
        self.rvc_camera = RVC_camera.RVC_camera()
        self.rvc_camera.RVC_init()
        #两个全局图像
        self.img_display = None
        #点的间距
        self.th_list = [[],[]]
        self.points = [] # 所有选中的点
        self.points_alone = [] # 单独的一条线
        self.path_all = []
        self.server = actionlib.SimpleActionServer('message_1_action', MessageAction, self.execute_callback, False)
        self.server.start()
        self.feedback = MessageFeedback() # 反馈回复信息
        self.result = MessageResult() # 结果回复信息
        self.feedback.status = "Processing"
        
        self.result.reply = "sucaacess"
        rospy.spin()
    def execute_callback(self,goal):
        self.server.publish_feedback(self.feedback)
        self.App()
        self.server.set_succeeded(self.result)
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
    def mouse_callback(self,event, x, y, flags, param):
        """鼠标回调函数，用于记录点击的点"""
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            self.points_alone.append((x, y))
            cv2.circle(self.img_display, (x, y), 5, (0, 0, 255), -1)  # 绘制红色实心圆点
            if len(self.points_alone) > 1:  # 如果已经点击了至少两个点
                # 绘制连接所有点的直线
                for i in range(len(self.points_alone) - 1):
                    cv2.line(self.img_display, self.points_alone[i], self.points_alone[i + 1], (0, 255, 0), 2)  # 绘制绿色直线
            cv2.imshow("Image", self.img_display)  # 更新显示图像
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append(self.points_alone)#将第一轮选好的点保存到新的列表里
            self.points_alone = []
            # 绘制连接所有点的直线
            for poses in self.points:
                for i in range(len(poses) - 1):
                    cv2.line(self.img_display, poses[i], poses[i + 1], (255, 255, 0), 2)  # 绘制绿色直线
            cv2.imshow("Image", self.img_display)  # 更新显示图像
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
            pose_goal.pose = self.pose_form(point,normal,128,dir_ant,j)
            path_msgs.poses.append(pose_goal)#将位姿信息按照顺序放入
        self.th_list = [[],[]]
        self.path_all.append(path_msgs)
        # self.pub.publish(path_msgs)
    def pose_form(self,pos,dir,th,dir_ant,j):#需要将相机的坐标先转换到机械臂末端在转化到基座坐标系
        #定义一个位姿信息作为返回参数
        pose_goal = geometry_msgs.msg.Pose()
        dir = np.asarray(dir)#改成矩阵
		#定义一个位姿数据，并赋值位置信息
        norm = np.linalg.norm(dir) #求模
        if norm == 0 :
            print("方向向量出错")
            return 0
        dir = dir/norm #单位化
        #沿着Z轴方向投影画圆
        x = np.cos(th*math.pi/180)
        y = np.sin(th*math.pi/180)

        # dir_ant = np.array([x,y,0]) #固定角度运行

        if np.all(np.cross(dir,dir_ant)==0):
            print("向量平行，重新选取")
            return
        x_axis = np.cross(dir,dir_ant) #选取X
        x_axis = x_axis /np.linalg.norm(x_axis) #求模单位化
        y_axis = np.cross(dir,x_axis)
        #写旋转矩阵，Z轴分向量就是输入的方向向量
        R = np.array([[x_axis[0],y_axis[0],dir[0]],[x_axis[1],y_axis[1],dir[1]],[x_axis[2],y_axis[2],dir[2]]])
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
        euler_angles = tf.transformations.euler_from_quaternion(quaternion)
        print(euler_angles_list[2]*180/math.pi)
        pose_goal.position.x = pose[0]
        pose_goal.position.y = pose[1]
        pose_goal.position.z = pose[2]+0.015
        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]
        return pose_goal
    def calculate_normal(self,points_of_cloud,x,y):
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
    def App(self):
        self.img_display = self.rvc_camera.get_image()
        self.width = self.rvc_camera.width
        self.height = self.rvc_camera.height
        # 显示图像并等待用户点击 n 个点
        cv2.imshow("Image", self.img_display)
        cv2.setMouseCallback("Image", self.mouse_callback, self.img_display)  # 设置鼠标回调函数
        pm = self.rvc_camera.get_pointCloud()
        while True:
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break
        cv2.destroyAllWindows()  # 关闭图像窗口
        self.deal_poses(pm,0)
    def deal_poses(self,pm,th_arm):
        # 提取多条直线上的点云数据和法向量
        all_line_points = None
        all_line_normals = None
        for poses in self.points:
            line_points = []
            line_normals = []
            for i in range(len(poses) - 1):  # 遍历每两个相邻点
                start_point = poses[i]  # 起点
                end_point = poses[i + 1]  # 终点
                # print(start_point)
                #获取方向
                line_direction = np.array(end_point) - np.array(start_point)
                #更换工件的默认初始角度
                line_direction[:2] = self.rotate_vector(line_direction[:2],th_arm)
                print("初始方向",line_direction)
                th = self.angle_between_vectors(line_direction[:2], np.array([281 , 84]))
                print(th)
                if  th > 90 and th < 270 :
                    line_direction = -line_direction
                self.th_list[1].append(line_direction)
                num_points = max(abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1]))  # 直线上的点数
                self.th_list[0].append(num_points)
                for j in range(num_points + 1):
                    x = int(start_point[0] + (end_point[0] - start_point[0]) * j / num_points)
                    y = int(start_point[1] + (end_point[1] - start_point[1]) * j / num_points)
                    if 0 <= x < self.width and 0 <= y < self.height:
                        point = pm[y, x]  # 获取点云中的点
                        normal = -self.calculate_normal(pm,y,x)
                        if not np.isnan(point).any() and not np.isnan(normal).any():  # 忽略无效点和法向量
                            line_points.append(point)
                            line_normals.append(normal)
            # 对投影后的法向量进行滑动平均滤波
            line_normals = calculate.sliding_window_filter(line_normals, window_size=31)
            if all_line_normals is None:
                all_line_points = line_points
                all_line_normals = line_normals
            else:
                all_line_points = np.vstack((all_line_points, line_points))
                all_line_normals = np.vstack((all_line_normals, line_normals))
            #将点云路径转化成path_msgs的格式
            self.path_form(np.array(line_normals),np.array(line_points))
        # 将直线上的点云数据和法向量转换为 Open3D 点云对象
        all_line_points = np.array(all_line_points)
        all_line_normals = np.array(all_line_normals)
        line_pcd = o3d.geometry.PointCloud()
        line_pcd.points = o3d.utility.Vector3dVector(all_line_points)  # 设置点云数据
        line_pcd.normals = o3d.utility.Vector3dVector(all_line_normals)  # 设置法向量数据
        # 显示直线上的点云和法向量
        o3d.visualization.draw_geometries([line_pcd], window_name="Line Points in Point Cloud with Normals",
                                            point_show_normal=True)  # 显示点云和法向量
        for path in self.path_all:
            self.path_pub.publish(path)
            time.sleep(1)
            self.pub.publish(1)
        self.path_all=[]
        self.points = [] # 所有选中的点
        self.points_alone = [] # 单独的一条线

def main():
    try:
        receiver = ImageReceiver()
    except rospy.ROSInterruptException:
        pass
if __name__ == '__main__':
    main()
    