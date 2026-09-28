#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import actionlib
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
import time
import sys
import rospkg
import os
import signal
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')
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
import actionlib
from app_pkg.msg import MessageAction, MessageGoal ,MessageResult# type: ignore
class SDK_control():
    def __init__(self):
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.pub_run = rospy.Publisher('chat2', String, queue_size=10)
        self.robot = fr_Robot()
        rospy.init_node('action_server', anonymous=True)
        rospy.set_param("score_limit",0.65)
        self.robot.ServoMoveStart()
        self._running = True  # 运行状态标志
        self.run_state = True
        self.path_list = []
        self.path_form()
        self.path_index = 0
        self.client = actionlib.SimpleActionClient("data_save", MessageAction)
        
        self.goal = MessageGoal()
        self.goal.message = "data_save"
    def feedback_cb(self,feedback):
        print("接收到了反馈")
    def _text_callback(self, msg):
        self.run_state = msg.data
        rospy.loginfo(f"收到消息: {msg.data}")

    def signal_handler(self, sig, frame):
        """处理中断信号"""
        rospy.loginfo("接收到中断信号，正在停止机器人...")
        self._running = False
        self.run_state = False
        self.robot.ServoMoveEnd()
        rospy.signal_shutdown("用户中断")
        sys.exit(0)

    def change_180(self, go_pos):
        """角度归一化到[-180,180]"""
        for i in [3, 5]:  # 只处理第4和第6个元素（角度）
            while go_pos[i] > 180:
                go_pos[i] -= 360
            while go_pos[i] < -180:
                go_pos[i] += 360
        return go_pos

    def change_error(self, aim_pose, desc_pos, error_pose):
        """处理角度误差"""
        if aim_pose[3] * desc_pos[3] < 0:
            if aim_pose[3] < 0:
                error_pose[3] += 360
            else:
                error_pose[3] -= 360
        return error_pose

    def path_form(self):
        """生成平滑路径"""
        # 统一使用float64类型
        pos1 = np.array([-295.0, -382.0, 250.0, 180.0, 0.0, 180], dtype=np.float64)
        pos2 = np.array([-210.49687512 ,-459.13036902, 250.0, 180.0, 0.0, 180], dtype=np.float64)
        # -169.95963564 ,-496.13080237 
        # -178.16935653 -488.63736585
        # -187.11533281 -480.47191099
        # -192.93982721 -475.15559396
        # -201.79802403 -467.07025995
        # -210.49687512 -459.13036902
        # 离散化路径
        num_points = 100
        self.now_pos = pos1.copy()
        
        for t in np.linspace(0, 1, num_points):
            if not self._running:
                break
            aim_pose = pos1 + t * (pos2 - pos1)
            error_pose = aim_pose - self.now_pos
            self.change_error(aim_pose, self.now_pos, error_pose)
            
            magnitude = np.linalg.norm(error_pose)
            steps = max(1, int(magnitude / 0.05))  # 确保至少1步
            
            for i in range(steps):
                if not self._running:
                    break
                step_ratio = (i + 1) / steps
                intermediate_pos = self.now_pos + step_ratio * error_pose
                intermediate_pos = self.change_180(intermediate_pos)
                self.path_list.append(intermediate_pos.copy())
                print("路径点",intermediate_pos)
            self.now_pos = aim_pose.copy()

    def two_pose_run(self):
        """两点间运动，支持中断"""
        try:
            # 初始移动到起点（确保使用正确数据类型）
            start_pos = np.array([-295.0, -382.0, 250.0, 180.0, 0.0, 180], dtype=np.float64)
            error = self.robot.MoveCart(start_pos, 0, 0, vel=20)
            time.sleep(20)
            self.client.send_goal(self.goal, feedback_cb=self.feedback_cb)
            self.client.wait_for_result()
            result = self.client.get_result()
            for i in range(len(self.path_list)-1):
                self.path_index = i
                # 确保发送给SDK的数据类型正确
                self.robot.ServoCart(0, self.path_list[self.path_index])
                time.sleep(0.015)  # 控制速度 
            self.client.send_goal(self.goal, feedback_cb=self.feedback_cb)
            self.client.wait_for_result()
            result = self.client.get_result()

            start_pos = np.array([-295.0, -382.0, 250.0, 180.0, 0.0, 180], dtype=np.float64)
            error = self.robot.MoveCart(start_pos, 0, 0, vel=20)

            msg = String()
            msg.data = "start"
            self.pub_run.publish(msg)
            self.sub = rospy.Subscriber("chat", String, self._text_callback, queue_size=1)
            for i in range(len(self.path_list)-1):
                if self.run_state == "stop":
                    rospy.loginfo("运动被中断")
                    break
                self.path_index = i
                # 确保发送给SDK的数据类型正确
                self.robot.ServoCart(0, self.path_list[self.path_index])
                time.sleep(0.015)  # 控制速度 
            while not rospy.is_shutdown():
                if self.run_state == "stop":
                    pass
                elif self.run_state == "back":
                    if self.path_index<=0:
                        pass
                    else:
                        self.path_index -=1
                        self.robot.ServoCart(0, self.path_list[self.path_index])
                        time.sleep(0.015)
                elif self.run_state == "go":
                    self.path_index +=1
                    self.robot.ServoCart(0, self.path_list[self.path_index])
                    time.sleep(0.015)
            rospy.loginfo("运动完成")

        except Exception as e:
            rospy.logerr(f"运动过程中发生错误: {str(e)}")
            self.robot.ServoMoveEnd()
        finally:
            self.robot.ServoMoveEnd()

def main():
    try:
        sdk = SDK_control()
        sdk.two_pose_run()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
    except Exception as e:
        rospy.logerr(f"主函数错误: {str(e)}")
    finally:
        rospy.loginfo("节点关闭")

if __name__ == "__main__":
    main()