#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
import threading 
from app_pkg.msg import MessageAction, MessageFeedback, MessageResult  # type: ignore
import actionlib
from std_msgs.msg import Float32MultiArray
import time
import os
from std_msgs.msg import String
class joy_control:
    def __init__(self):
        rospy.init_node('control',anonymous=True)
        rospy.Subscriber('/joy',Joy,self._joy_callback) #定义订阅者
        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0
        self.desc_pos = [0,0,0,0,0,0]
        self.mode_speed = 0.6
        self.joy_server = actionlib.SimpleActionServer('joy',  MessageAction, self._execute_callback ,auto_start=False)
        self.pub = rospy.Publisher('chat', String, queue_size=10)
        self.joy_server.start()
        self.joy_state = False
        self.botton_state = True
    def check_joystick(self):
        # 检查是否存在 js* 设备
        joysticks = [f for f in os.listdir('/dev/input') if f.startswith('js')]
        if joysticks:
            return True
        else:
            return False
    def _execute_callback(self,goal):
        feedback = MessageFeedback()
        result = MessageResult() 
        self.joy_state = not self.joy_state
        if self.joy_state:
            feedback.status = "enable joystick control"
            result.reply = "enable successful"
            time_delat = 0
            while not self.check_joystick():
                if time_delat > 300:
                    self.joy_state = False
                    result.reply = "enable failed"
                    break
                time_delat += 1
                feedback.status = "please connect joystick"
                self.joy_server.publish_feedback(feedback)
                time.sleep(0.1)
        else :
            feedback.status = "disable joystick control"
            result.reply = "disable successful"
        self.joy_server.publish_feedback(feedback)
        time.sleep(0.1)
        self.joy_server.set_succeeded(result)  # 明确
    def _joy_callback(self,msg):
        message = String()
        if msg.axes[5] !=0 and msg.axes[4] != 0 :
            self.botton_state = False
        else :
            if msg.axes[5] != 0 and self.botton_state :
                self.z = ((msg.axes[5]-1)/2)*self.mode_speed
            if msg.axes[4] != 0 and self.botton_state :
                self.z = (-(msg.axes[4]-1)/2)*self.mode_speed
        if not self.botton_state :
            self.z = ((msg.axes[5]-1)/2-(msg.axes[4]-1)/2)*self.mode_speed
        self.start = True
        if msg.buttons[4] == 1 :
            self.Back_nall = True
        if msg.buttons[7] == 1 :
            # 发布消息
            message.data = f"speed gear {self.mode_speed//0.4}"
            self.pub.publish(message)
            self.mode_speed += 0.4
            if self.mode_speed > 1.3:
                self.mode_speed = 0.04
        self.x = msg.axes[0]*self.mode_speed
        self.y = msg.axes[1]*self.mode_speed
        self.rx = msg.axes[2]/2*self.mode_speed
        self.ry = msg.axes[3]/2*self.mode_speed
        print(msg.axes[5],msg.axes[4],self.z)
        self.rz = (msg.buttons[0]-msg.buttons[1])/2*self.mode_speed
        self.desc_pos = [self.x,self.y,self.z,self.rx,self.ry,self.rz]
    def joy_publish(self):
        self.joy_pub = rospy.Publisher('joy_control', Float32MultiArray, queue_size=10)
        msg = Float32MultiArray()
        rate = rospy.Rate(100)  # 1Hz发布频率
        while not rospy.is_shutdown():
            if self.joy_state: 
                # 设置数据部分
                msg.data = self.desc_pos 
                self.joy_pub.publish(msg)
                rate.sleep()
def main():
    joy = joy_control()
    joy.joy_publish()
if __name__ == "__main__":
    main()