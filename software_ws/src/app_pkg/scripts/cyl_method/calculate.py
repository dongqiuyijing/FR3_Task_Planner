import numpy as np
import tf
import geometry_msgs.msg
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
def pose_form(T_camera,pos,normal,dir_ant,dirct:bool):#点和法向量转换成路径点
    """路径点生成函数
    输入：一个手眼标定矩阵，
        一个位置，
        一个法向量，
        一个路径走向，
        一个是否零方位控制路径 
    形成一个点的信息最后返回这个带有四元数的点"""
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
    T_world = T_camera @ T
    pose = T_world[:3,3]
    quaternion = tf.transformations.quaternion_from_matrix(T_world)
    euler_angles = tf.transformations.euler_from_quaternion(quaternion)
    euler_angles_list = list(euler_angles)
    if dirct :
        euler_angles_list[0] = math.pi
        euler_angles_list[1] = 0
    euler_angles = tuple(euler_angles_list)
    T_world = tf.transformations.quaternion_matrix(tf.transformations.quaternion_from_euler(*euler_angles))
    quaternion = tf.transformations.quaternion_from_matrix(T_world)
    pose_goal.position.x = pose[0]
    pose_goal.position.y = pose[1]
    pose_goal.position.z = pose[2]+0.005  #小探头：0.005
    pose_goal.orientation.x = quaternion[0]
    pose_goal.orientation.y = quaternion[1]
    pose_goal.orientation.z = quaternion[2]
    pose_goal.orientation.w = quaternion[3]
    return pose_goal
def sliding_window_filter(normals: np.ndarray, window_size=5):#平滑法向量
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
def poly_fit(x:np.ndarray,y:np.ndarray,degree = 5): # 利用多次函数拟合路径
        """输入是:x和y的一维路径矩阵
                degree是拟合次数,可以使用默认值"""
        coefficients = np.polyfit(x, y, degree)
        poly = np.poly1d(coefficients)
        # # 生成用于绘制拟合曲线的 x 值
        x_fit = np.linspace(min(x), max(x), y.shape[0])
        y_fit = poly(x_fit)
        return y_fit
def find_best_degree(x:np.ndarray,y:np.ndarray,max_degree=20): # 寻找最佳的拟合次数
    """输入是:x和y的一维路径矩阵
    degree寻找的最大拟合次数,可以使用默认值"""
    best_degree = 0
    best_score = -np.inf
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    for degree in range(1, max_degree + 1):
        # 进行多项式拟合
        coefficients = np.polyfit(x_train, y_train, degree)
        poly = np.poly1d(coefficients)
        y_pred = poly(x_test)
        # 计算 R^2 分数
        score = r2_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_degree = degree
    return best_degree
def normals_form(points_list:np.ndarray,side_list:np.ndarray): # 利用旁边的路径计算法向量
    """ 输入是两条路径矩阵
        输出是一个路径的法向量"""
    # 转化成列表
    points_list = list(points_list)
    side_list = list(side_list)
    # 定义法向量列表
    normals_list = []
    normal = np.array([0,0,-1])
    for i in range(len(points_list)):
        if i == 0:
            normals_list.append(normal)
            continue
        pos1 = points_list[i-1]
        #y轴偏离点
        pos2 = side_list[i]
        #目标点
        pos_aim = points_list[i]
        dir1 = pos1 - pos_aim
        dir2 = pos2 - pos_aim
        normal = np.cross(dir1, dir2)  # 计算平面法向量
        normal /= np.linalg.norm(normal)  # 归一化平面法向量
        normals_list.append(-normal)
    return normals_list
def rotate_vector(vector, angle_degrees):# 将一个二维向量沿着z轴旋转一个度数
    """将一个二维向量沿着z轴旋转一个度数
        输入:一个向量和角度制的角度"""
    angle_radians = np.radians(angle_degrees)
    
    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    
    # 进行矩阵乘法
    rotated_vector = np.dot(rotation_matrix, np.array(vector))
    return rotated_vector
def angle_between_vectors(v1, v2): # 求解两个向量的夹角
    """输入 :两个向量，求解两个向量的夹角
    第二个参数应该是相机的初始角度"""
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
def force_close(error,tolerance_f = 2,tolerance_t = 0.05):# 力的检测
    for index in range(len(error)):
        if index < 3:
            if abs(error[index]) < tolerance_f:
                error[index] = 0
        else :
            if abs(error[index]) < tolerance_t:
                error[index] = 0
    return error
def change_180(go_pos):
    """输入:六轴欧拉角,避开机械臂的死角"""
    if go_pos[3] > 180 :
        go_pos[3]-=360
    elif go_pos[3] < -180:
        go_pos[3]+=360
    if go_pos[5] > 180 :
        go_pos[5]-=360
    elif go_pos[5] < -180:
        go_pos[5]+=360
def change_error(aim_pose,desc_pos,error_pose):
    if aim_pose[3]*desc_pos[3]<0:
        if aim_pose[3]<0 :
            error_pose[3]+=360
        else:
            error_pose[3]-=360 
    if aim_pose[5]*desc_pos[5]<0:
        if aim_pose[5]<0 :
            error_pose[5]+=360
        else:
            error_pose[5]-=360 






