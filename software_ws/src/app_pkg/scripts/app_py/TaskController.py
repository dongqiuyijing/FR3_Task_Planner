#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import actionlib
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
import time
import sys
import rospkg
import os
# 获取当前ROS包的路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# 动态构建scripts目录路径
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0,scripts_path)
from FR_SDK import fr_Robot
from std_msgs.msg import Float32MultiArray
import nav_msgs.msg
from gui import button
TaskController = button.TaskController
import numpy as np
import tf
from std_msgs.msg import String
import math
class SDK_control():
    def __init__(self):
        self.robot = fr_Robot()
        # 初始化ROS节点
        rospy.init_node('action_server',anonymous=True)
        self.action_long_press_init()
        rospy.Subscriber('joy_control', Float32MultiArray, self._joy_control_callback)
        self.subscriber_path = rospy.Subscriber('path_cyl',nav_msgs.msg.Path,self.pathCallback) #定义订阅者
        self.timer = rospy.Timer(rospy.Duration(0.05), self.publish_pose_callback) # 定时器中断
        self.pose_pub = rospy.Publisher('arm_pose', Float32MultiArray, queue_size=1) # 位置发布器
        self.chat_pub = rospy.Publisher("chat", String, queue_size=1) # 聊天发布
        self.aim_pos = [-293,-5,539,180,0,168]
        # self.aim_pos = [-308.58984251580534, -345.5362317045451, 283.9512985371837, 176.42760604489106, 0.5750185720704686, 59.10353269910553]

        self.T_tool = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0.12],
                              [0,0,0,1]])
        self.now_path = None
        self.path_run = False
        self.path_break = False
    def change_180(self,go_pos):
        if go_pos[3] > 180 :
            go_pos[3]-=360
        elif go_pos[3] < -180:
            go_pos[3]+=360
        if go_pos[5] > 180 :
            go_pos[5]-=360
        elif go_pos[5] < -180:
            go_pos[5]+=360
    def change_error(self,aim_pose,desc_pos,error_pose):
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
    def starting(self):
        pos = [-293,-278,435,180,0,168]
        error = self.robot.MoveCart(pos, 0,0,vel = 20)
        time.sleep(1)
        for i in range(8):
            pos = [-293,-278,350,180,0,168]
            error = self.robot.MoveCart(pos, 0,0,vel = 100)
            pos = [-293,-278,435,180,0,168]
            error = self.robot.MoveCart(pos, 0,0,vel = 100)
        time.sleep(1)
        self.robot.MoveCart(self.aim_pos, 0,0,vel=20)
    def publish_pose_callback(self, event):
        """定时器回调函数"""
        try:
            # 获取机器人位姿
            error, data = self.robot.GetActualTCPPose()
            if error:
                rospy.logwarn(f"获取位姿失败，错误码: {error}")
            
            else:
                # 发布位姿信息
                msg = Float32MultiArray(data=data)
                self.pose_pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"位姿发布异常: {str(e)}")
            self.error_count += 1
    def pathCallback(self,msg):
        # input("路径是否可以执行")
        msg2 = String()
        msg2.data = "已经获取到了路径"
        self.chat_pub.publish(msg2)
        self.now_path = msg.poses
    def _path_run(self):
        pos = [-293,-278,435,180,0,168]
        error = self.robot.MoveCart(pos, 0,0,vel = 20)
        inhere_height = rospy.get_param("inhere_height")
        print("运动状态：",error,pos)
        if self.now_path is None:
            msg = String()
            msg.data = "路径未获取或已经舍弃"
            self.chat_pub.publish(msg)
            print(msg.data)
            self.path_run = False
            return
        aim_pose = [0,0,0,0,0,0]
        for i,pose in enumerate(self.now_path):
            if self.path_break:
                msg = String()
                msg.data = "中断当前路径运行"
                self.chat_pub.publish(msg)
                # 状态清除
                self.path_break = False
                self.path_run = False
                return
            pos_aim = [pose.pose.position.x,pose.pose.position.y,pose.pose.position.z]
            quaternion = [pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w]
            R_aim = tf.transformations.quaternion_matrix(quaternion)[:3,:3]
            T_aim = np.eye(4)
            T_aim[:3,:3]=R_aim
            T_aim[:3,3]=np.array(pos_aim)
            #求解安装了工具的运行路径
            if inhere_height:
                T_move = T_aim
            else :
                T_move = np.dot(T_aim,np.linalg.inv(self.T_tool))
            quaternion_move = tf.transformations.quaternion_from_matrix(T_move)
            pos_move = T_move[:3,3]
            angle = tf.transformations.euler_from_quaternion(quaternion_move)
            angle = np.array(angle)*180/math.pi
            aim_pose[0]=pos_move[0]*1000
            aim_pose[1]=pos_move[1]*1000
            
            inhere_height_data = rospy.get_param("inhere_height_data")
            if inhere_height:
                z_height = rospy.get_param("z_height")
                aim_pose[2] = inhere_height_data + z_height*1000
            else :
                aim_pose[2]=pos_move[2]*1000  #254
                if aim_pose[2] < 219 :  #限位
                    aim_pose[2] = 219
            aim_pose[3]=angle[0]
            aim_pose[4]=angle[1]
            aim_pose[5]=angle[2]
            if i == 0:
                # self.robot.ServoMoveEnd()
                error = self.robot.MoveCart(aim_pose, 0,0,vel = 10)
                # print("运动状态：",error,aim_pose)
                # self.robot.ServoMoveStart()
                continue
            error,desc_pos = self.robot.GetActualTCPPose()
            print("当前末端角度",aim_pose[5],desc_pos[5])
            error_pose = np.array(aim_pose)-np.array(desc_pos)
            self.change_error(aim_pose,desc_pos,error_pose)
            magnitude = np.linalg.norm(error_pose) 
            # print(error_pose)
            num = magnitude//0.3
            error_pose = error_pose/num
            go_pos = np.array(desc_pos)
            for i in range(int(num)):
                go_pos += error_pose
                self.change_180(go_pos)
                self.robot.ServoCart(0,go_pos)
                time.sleep(0.01)#调整速度
        self.robot.ServoMoveEnd()
        self.path_run = False
    def _two_pose_run(self):
        pos1 = np.array([-295,-382,250, 180, 0, 0])
        pos2 = np.array([-146,-518,250 , 180, 0, 0])
        error = self.robot.MoveCart(pos1, 0,0,vel = 20)
        # 离散化成100个点
        num_points = 100
        for t in np.linspace(0, 1, num_points):
            aim_pose = pos1 + t * (pos2 - pos1)  # 线性插值
            error,desc_pos = self.robot.GetActualTCPPose()
            # print("当前末端角度",aim_pose[5],desc_pos[5])
            error_pose = np.array(aim_pose)-np.array(desc_pos)
            self.change_error(aim_pose,desc_pos,error_pose)
            magnitude = np.linalg.norm(error_pose) 
            num = magnitude//0.1
            error_pose = error_pose/num
            go_pos = np.array(desc_pos)
            for i in range(int(num)):
                go_pos += error_pose
                self.change_180(go_pos)
                self.robot.ServoCart(0,go_pos)
                time.sleep(0.015)#调整速度
        self.robot.ServoMoveEnd()
        self.path_run = False
    #   self.starting()
    def servo_tool(self,jop,T):# 单轴控制函数
        tool_serve_tool = [0,0,0,0,0,0]
        if T :
            if jop > 2:
                tool_serve_tool[jop] = 0.2
            else :
                tool_serve_tool[jop] = 0.5
        else :
            if jop > 2:
                tool_serve_tool[jop] = -0.2
            else :
                tool_serve_tool[jop] = -0.5
        self.robot.ServoCart(2,tool_serve_tool,vel=60)
        time.sleep(0.01)      
    def _execute_callback(self,goal,server, controller):# 行为回调函数
        """带参数的回调函数"""
        feedback = MessageFeedback()
        result = MessageResult() 
        try:
            if goal.message == "Start":
                rospy.loginfo("收到启动请求")
                controller.start_task()
                feedback.status = "启动中"
                result.reply = "启动成功"
            elif goal.message == "Stop":
                rospy.loginfo("收到停止请求")
                controller.stop_task()
                feedback.status = "停止中"
                result.reply = "停止成功"   
            server.publish_feedback(feedback)
            server.set_succeeded(result)  # 明确设置终止状态
        except Exception as e:
            rospy.logerr("处理错误: %s", str(e))
            result.reply = f"错误: {str(e)}"
    def _execute_2_callback(self,goal,server):
        feedback = MessageFeedback()
        result = MessageResult() 
        if goal.message == "path_run":
            self.path_run = True
            feedback.status = "path run go go go !!!"
            result.reply = "path run go now"
        elif goal.message == "path_close":
            if self.path_run:
                feedback.status = "Path execution in progress"
                result.reply = "No Path Clearing Allowed !!!"
            else :
                self.now_path = None
                feedback.status = "close path running "
                result.reply = "close path successful"
        elif goal.message == "path_break":
            self.path_break = True
            feedback.status = "break path running"
            result.reply = "break path successful"
        else:
            self.starting()
            feedback.status = "go to"
            result.reply = "successful"
        server.publish_feedback(feedback)
        time.sleep(0.1)
        server.set_succeeded(result)  # 明确
    def _joy_control_callback(self,data):
        self.robot.ServoCart(2,data.data,vel=60)
    def action_long_press_init(self):# 长按的行为回调函数
        # 创建控制器实例
        task_controller1  = TaskController(task_function=lambda: self.servo_tool(0, True))
        task_controller2  = TaskController(task_function=lambda: self.servo_tool(0, False))
        task_controller3  = TaskController(task_function=lambda: self.servo_tool(1, True))
        task_controller4  = TaskController(task_function=lambda: self.servo_tool(1, False))
        task_controller5  = TaskController(task_function=lambda: self.servo_tool(2, True))
        task_controller6  = TaskController(task_function=lambda: self.servo_tool(2, False))
        task_controller7  = TaskController(task_function=lambda: self.servo_tool(3, True))
        task_controller8  = TaskController(task_function=lambda: self.servo_tool(3, False))
        task_controller9  = TaskController(task_function=lambda: self.servo_tool(4, True))
        task_controller10 = TaskController(task_function=lambda: self.servo_tool(4, False))
        task_controller11 = TaskController(task_function=lambda: self.servo_tool(5, True))
        task_controller12 = TaskController(task_function=lambda: self.servo_tool(5, False))
        # 创建动作服务器（使用lambda传递参数）
        server1  = actionlib.SimpleActionServer('x_add',  MessageAction, lambda goal: self._execute_callback(goal, server1,  task_controller1),auto_start=False)
        server2  = actionlib.SimpleActionServer('x_subtrack',  MessageAction, lambda goal: self._execute_callback(goal, server2,  task_controller2),auto_start=False)
        server3  = actionlib.SimpleActionServer('y_add',  MessageAction, lambda goal: self._execute_callback(goal, server3,  task_controller3),auto_start=False)
        server4  = actionlib.SimpleActionServer('y_subtrack',  MessageAction, lambda goal: self._execute_callback(goal, server4,  task_controller4),auto_start=False)
        server5  = actionlib.SimpleActionServer('z_add',  MessageAction, lambda goal: self._execute_callback(goal, server5,  task_controller5),auto_start=False)
        server6  = actionlib.SimpleActionServer('z_subtrack',  MessageAction, lambda goal: self._execute_callback(goal, server6,  task_controller6),auto_start=False)
        server7  = actionlib.SimpleActionServer('rx_add', MessageAction, lambda goal: self._execute_callback(goal, server7,  task_controller7),auto_start=False)
        server8  = actionlib.SimpleActionServer('rx_subtrack', MessageAction, lambda goal: self._execute_callback(goal, server8,  task_controller8),auto_start=False)
        server9  = actionlib.SimpleActionServer('ry_add', MessageAction, lambda goal: self._execute_callback(goal, server9,  task_controller9),auto_start=False)
        server10 = actionlib.SimpleActionServer('ry_subtrack', MessageAction, lambda goal: self._execute_callback(goal, server10, task_controller10),auto_start=False)
        server11 = actionlib.SimpleActionServer('rz_add', MessageAction, lambda goal: self._execute_callback(goal, server11, task_controller11),auto_start=False)
        server12 = actionlib.SimpleActionServer('rz_subtrack', MessageAction, lambda goal: self._execute_callback(goal, server12, task_controller12),auto_start=False)
        server13 = actionlib.SimpleActionServer('go_to_start',  MessageAction, lambda goal: self._execute_2_callback(goal,server13) ,auto_start=False)
        server14 = actionlib.SimpleActionServer('run_the_path',  MessageAction, lambda goal: self._execute_2_callback(goal,server14) ,auto_start=False)
        server15 = actionlib.SimpleActionServer('clear_the_path',  MessageAction, lambda goal: self._execute_2_callback(goal,server15) ,auto_start=False)
        server16 = actionlib.SimpleActionServer('break_the_path',  MessageAction, lambda goal: self._execute_2_callback(goal,server16) ,auto_start=False)
        #开启动作执行器
        server1.start()
        server2.start()
        server3.start()
        server4.start()
        server5.start()
        server6.start()
        server7.start()
        server8.start()
        server9.start()
        server10.start()
        server11.start()
        server12.start()
        server13.start()
        server14.start()
        server15.start()
        server16.start()
        #开启伺服控制
        self.robot.ServoMoveStart()
    def pose_publish(self):# 机械臂位置发布函数
        # pub = rospy.Publisher('arm_pose', Float32MultiArray, queue_size=1)
        rate = rospy.Rate(5)  # 1Hz发布频率
        # 带3位小数的列表数据
        while not rospy.is_shutdown():
            try :
                if self.path_run:
                    self._path_run()
                # msg = Float32MultiArray(data=data)
                # pub.publish(msg)
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.logerr("ROS interrupt exception")
                break
            except Exception as e:
                rospy.logerr("Error in A node: %s", str(e))
                break
   
def main():
    sdk = SDK_control()
    # sdk.starting()
    sdk.pose_publish()
if __name__ == "__main__":
    main()