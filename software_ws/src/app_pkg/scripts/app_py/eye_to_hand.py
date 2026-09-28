import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String, Float32MultiArray
import rospy
import actionlib
import math
import os
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
import rospkg
class eye_to_hand:
    def __init__(self):
        rospy.init_node("calculate_hand_eye", anonymous=True)
        rospy.Subscriber('arm_pose', Float32MultiArray, self._arm_pose_callback)
        rospy.Subscriber('image_pose', Float32MultiArray, self._image_pose_callback)
        self.server  = actionlib.SimpleActionServer('record_data',  MessageAction,self._execute_callback,auto_start=False)
        self.server.start()
        self.hand_pose = [0,0,0,0,0,0]
        self.camera_pose = [0,0,0,0,0,0]

        self.hand_poses = []
        self.camera_poses = []
    def _arm_pose_callback(self, data):
        self.hand_pose[0] = data.data[0]*0.001
        self.hand_pose[1] = data.data[1]*0.001
        self.hand_pose[2] = data.data[2]*0.001
        self.hand_pose[3] = data.data[3]*math.pi/180
        self.hand_pose[4] = data.data[4]*math.pi/180
        self.hand_pose[5] = data.data[5]*math.pi/180
    def _image_pose_callback(self,data):
        self.camera_pose = list(data.data)
    def _execute_callback(self,goal):
        """带参数的回调函数"""
        feedback = MessageFeedback()
        result = MessageResult() 
        try:
            if goal.message == "record":
                if self.camera_pose == None:
                    feedback.status = "record data ..."
                    result.reply = "fail !"
                else :
                    #记录数据
                    print("arm:",self.hand_pose)
                    print("camera:",self.camera_pose)
                    self.hand_poses.append(list(self.hand_pose)) # 这里使用list是为了保存数据的副本，而不是引用
                    self.camera_poses.append(list(self.camera_pose))
                    print(self.hand_poses)
                    print(self.camera_poses)
                    feedback.status = "record data ..."
                    result.reply = "success!!!"
            elif goal.message == "calculate":
                feedback.status = "To calculate final data"
                result.reply = "success"   
                self.calculate()
            else :
                feedback.status = "fail"
                result.reply = "fail"
            self.server.publish_feedback(feedback)
            self.server.set_succeeded(result)  # 明确设置终止状态
        except Exception as e:
            rospy.logerr("处理错误: %s", str(e))
            result.reply = f"错误: {str(e)}" 

    # ==================== 数据预处理 ====================
    def pose_to_rot_trans(self,pose):
        """ 将6自由度位姿转换为旋转矩阵和平移向量 """
        x, y, z, rx, ry, rz = pose
        rot = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        trans = np.array([x, y, z])
        return rot, trans
    def calculate(self):
        print(self.hand_poses)
        print(self.camera_poses)
        # 转换为OpenCV需要的格式
        R_gripper2base, t_gripper2base = [], []
        R_target2cam, t_target2cam = [], []

        for h_pose, c_pose in zip(self.hand_poses, self.camera_poses):
            # 机械臂末端到基座（注意：眼在手外需取逆）
            R_gi, t_gi = self.pose_to_rot_trans(h_pose)
            R_gripper2base.append(R_gi.T)  # 逆旋转
            t_gripper2base.append(-R_gi.T @ t_gi)  # 逆平移
            
            # 标定板到相机（直接使用检测结果）
            R_tc, t_tc = self.pose_to_rot_trans(c_pose)
            R_target2cam.append(R_tc)
            t_target2cam.append(t_tc)

        # ==================== 执行标定 ====================
        # 选择算法（推荐TSAI或PARK）
        X = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK
        )
        T = np.eye(4)
        T[:3,:3] = X[0]
        T[:3,3] = X[1].flatten()
        print(T)
        # 保存到 txt 文件（默认格式）
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
        # 动态构建scripts目录路径
        data_path = os.path.join(pkg_path, 'config/camera_rvc/hand_eye.txt')
        np.savetxt(data_path, T)
def main():
    eye_hand = eye_to_hand()
    # eye_hand.calculate()
    rospy.spin()
if __name__ == '__main__':
    main()
