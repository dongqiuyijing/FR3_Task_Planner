import sys
import os
import tkinter as tk
print(f"[DEBUG] tkinter path: {tk.__file__}")

from PIL import Image, ImageTk

import rospy
print(f"[DEBUG] rospy path: {rospy.__file__}")

from sensor_msgs.msg import Image as RosImage

from cv_bridge import CvBridge

import cv2
print(f"[DEBUG] OpenCV path: {cv2.__file__}")

import threading
import queue
import time
import numpy
print(f"NumPy路径: {numpy.__file__}")
import actionlib
print(f"[DEBUG] actionlib path: {actionlib.__file__}")

import rospkg
print(f"[DEBUG] rospkg path: {rospkg.__file__}")

from std_msgs.msg import String, Float32MultiArray

rospack = rospkg.RosPack()
from app_pkg.msg import MessageAction, MessageGoal ,MessageResult# type: ignore
# 获取当前ROS包的路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('app_pkg')  # 替换为你的包名
# 动态构建scripts目录路径
scripts_path = os.path.join(pkg_path, 'scripts')
sys.path.insert(0,scripts_path)
from gui import button
LongPressButton = button.LongPressButton
Button = button.Button
class RosTkViewer:
    def __init__(self):
        rospy.init_node('tk_image_viewer', anonymous=True)
        rospy.set_param("z_height",0.035)
        rospy.set_param("fit_line", True) 
        rospy.set_param("inhere_height", False)  
        rospy.set_param("all_direction", False)  
        rospy.set_param("inhere_height_data", 190)
        rospy.set_param("points_num", 4)
        self.root = tk.Tk()
        self.root.title("ROS-Tkinter Viewer")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 获取屏幕实际尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # 设置窗口大小为屏幕的 80%（示例比例）
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.root.geometry(f"{window_width}x{window_height}") 
        # 图像显示界面
        self.panel = tk.Label(self.root, bg='black')
        self.panel.grid(row=0, column=0,sticky="", padx=10, pady=10)
        # self.panel.config(width=500, height=150)
        # 状态历史记录文本框
        self.status_frame = tk.Frame(self.root)
        self.status_frame.grid(row=0, column=2, sticky="", padx=10, pady=10)  # 使用 grid 定位
        # self.status_frame.config(width=500, height=150)  # 固定尺寸

        self.scrollbar = tk.Scrollbar(self.status_frame)

        self.status_text = tk.Text(
            self.status_frame,
            wrap=tk.WORD,
            yscrollcommand=self.scrollbar.set,
            state='disabled',
            font=('song ti', 12),
            bg='#F5F5F5',
            spacing2=5,
            width=100,  # 字符宽度（非像素！）
            height=15  # 字符高度（非像素！）
        )
        # 将 Text 和 Scrollbar 放入网格
        self.status_text.grid(row=0, column=0, sticky="")  # 填充整个 Frame
        self.scrollbar.config(command=self.status_text.yview)
        self.scrollbar.grid(row=0, column=1, sticky="")     # 垂直滚动条靠右
        self.status_text.tag_config('success', foreground='#2E8B57')
        self.status_text.tag_config('error', foreground='#DC143C')
        self.status_text.tag_config('info', foreground='#1E90FF')
        
        self.image_queue = queue.Queue(maxsize=1)
        self.bridge = CvBridge()
        self.displaying = False
        self.waiting = False
        self._hand_eye_key()
        self._key_list()
        self._long_key_list()
        self.ros_thread = threading.Thread(target=self._ros_init)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        self.root.after(50, self._update_gui)

    def _long_key_list(self):
        """长按按键的设置"""
        button_frame = tk.Frame(self.root, bg="lightgray",height=30)
        button_frame.grid(row=0, column=1, sticky="", padx=10, pady=(10,0))  # 水平拉伸
        
        self.lbtn_action1 = LongPressButton(button_frame)
        self.lbtn_action2 = LongPressButton(button_frame)
        self.lbtn_action3 = LongPressButton(button_frame)
        self.lbtn_action4 = LongPressButton(button_frame)
        self.lbtn_action5 = LongPressButton(button_frame)
        self.lbtn_action6 = LongPressButton(button_frame)
        self.lbtn_action7 = LongPressButton(button_frame)
        self.lbtn_action8 = LongPressButton(button_frame)
        self.lbtn_action9 = LongPressButton(button_frame)
        self.lbtn_action10 = LongPressButton(button_frame)
        self.lbtn_action11 = LongPressButton(button_frame)
        self.lbtn_action12 = LongPressButton(button_frame)
        long_button_configs = [
            #       按键名称     action 开始消息   停止消息  显示名称  位置x  位置y
            (self.lbtn_action1,     "x_add",        "Start", "Stop",   "x+",    0   ,   0),
            (self.lbtn_action2,     "x_subtrack",   "Start", "Stop",   "x-",    0   ,   1),
            (self.lbtn_action3,     "y_add",        "Start", "Stop",   "y+" ,   1   ,   0),
            (self.lbtn_action4,     "y_subtrack",   "Start", "Stop",   "y-" ,   1   ,   1),
            (self.lbtn_action5,     "z_add",        "Start", "Stop",   "z+",    2   ,   0),
            (self.lbtn_action6,     "z_subtrack",   "Start", "Stop",   "z-",    2   ,   1),
            (self.lbtn_action7,     "rx_add",       "Start", "Stop",  "rx+" ,   3   ,   0),
            (self.lbtn_action8,     "rx_subtrack",  "Start", "Stop",  "rx-" ,   3   ,   1),
            (self.lbtn_action9,     "ry_add",       "Start", "Stop",  "ry+",    4   ,   0),
            (self.lbtn_action10,    "ry_subtrack",  "Start", "Stop",  "ry-",    4   ,   1),
            (self.lbtn_action11,    "rz_add",      "Start", "Stop",  "rz+" ,    5   ,   0),
            (self.lbtn_action12,    "rz_subtrack", "Start", "Stop",  "rz-" ,    5   ,   1)
        ]
        
        for lbtn, action_name, start_msg, stop_msg, btn_text, row, col in long_button_configs:
            lbtn.config(
                text=btn_text,
                press_cmd=lambda a=action_name, m=start_msg, b=lbtn: self._Laction_callback(a, m, b, True),  # 按下时发送开始指令
                release_cmd=lambda a=action_name, m=stop_msg, b=lbtn: self._Laction_callback(a, m, b, False),  # 松开时发送停止指令
                long_thresh=1.0,
                width=8,
                height=1,
                bg="#FFD700",
                activebackground="#FFEC8B",
                font=('Arial', 12) 
            )
            lbtn.grid(row=row, column=col, padx=5, pady=5, sticky="")  #   
        #   位置反馈标签
        self.label_x = tk.Label(button_frame)
        self.label_y = tk.Label(button_frame)
        self.label_z = tk.Label(button_frame)
        self.label_rx = tk.Label(button_frame)
        self.label_ry = tk.Label(button_frame)
        self.label_rz = tk.Label(button_frame)
        label_configs = [
            #   (Label对象, 显示文本, 位置x, 位置y)
            (self.label_x, "X: 0.000", 0, 2),
            (self.label_y, "Y: 0.000", 1, 2),
            (self.label_z, "Z: 0.000", 2, 2),
            (self.label_rx, "RX: 0.000", 3, 2),
            (self.label_ry, "RY: 0.000", 4, 2),
            (self.label_rz, "RZ: 0.000", 5, 2)
        ]
        # 创建并放置Label控件
        for label, text, row, col in label_configs:
            label.config(text=text,
                        font=('Arial', 12),
                        bg='white',
                        fg='black',
                        width=8,
                        height = 1,
                        relief=tk.RIDGE)
            label.grid(row=row, column=col, sticky="ew", padx=5, pady=5) 
    def _hand_eye_key(self):
        # 定义一个框架存放长按的按键
        self.hand_eye_key = tk.Frame(self.root,width=500,height=200)
        self.hand_eye_key.grid(row = 1,column = 1 ,sticky="", padx=10, pady=10)  
        self.hand_eye_key.grid_remove()
        btn_action1 = Button(self.hand_eye_key)
        btn_action2 = Button(self.hand_eye_key)
        btn_action3 = Button(self.hand_eye_key)
        button_configs = [
        {"btn":btn_action1, "action_name":"record_data",      "action_text":"record",      "btn_name":"记录一组数据",      "grid":(0,0),"close_frame": False},
        {"btn":btn_action2, "action_name":"record_data",      "action_text":"calculate",   "btn_name":"计算数据"  ,         "grid":(0,1),"close_frame": False},
        {"btn":btn_action3, "action_name":"aruco_get",          "action_text":"Stop",           "btn_name":"结束手眼标定"  ,    "grid":(0,2),"close_frame": True}]
        for button_config in button_configs:
            button_config["btn"].config(
                text=button_config["btn_name"],
                command=lambda a=button_config["action_name"], m=button_config["action_text"], b=button_config["btn"]: self._action_callback(a, m, b),
                width=10,
                height=3,
                bg="#FFD700",
                activebackground="#FFEC8B",
                font=('Arial', 15),
                close_frame = button_config["close_frame"]                
            )
            button_config["btn"].grid(row = button_config["grid"][0],column = button_config["grid"][1] ,sticky="", padx=10, pady=10)  
    def _key_list(self):
        # 定义一个框架存放按键
        frame_button = tk.Frame(self.root)
        frame_button.grid(row=2, column=0, sticky="", padx=10, pady=(10,0))  # 水平拉伸
        # frame_button.grid_propagate(False)  # 禁止根窗口自动调整
        self.buttons = []
        frame_config = [
            {"name": "screen1",  "row": 0, "col": 0, "width": 50, "height": 40},
            {"name": "screen2",  "row": 0, "col": 1, "width": 50, "height": 40},
            {"name": "screen3",  "row": 0, "col": 2, "width": 50, "height": 40},
            {"name": "screen4",  "row": 0, "col": 3, "width": 50, "height": 40},
            {"name": "screen5",  "row": 1, "col": 0, "width": 50, "height": 40},
            {"name": "screen6",  "row": 1, "col": 1, "width": 50, "height": 40},
            {"name": "screen7",  "row": 1, "col": 2, "width": 50, "height": 40},
            {"name": "screen8",  "row": 1, "col": 3, "width": 50, "height": 40},
            {"name": "screen9",  "row": 2, "col": 0, "width": 50, "height": 40},
            {"name": "screen10", "row": 2, "col": 1, "width": 50, "height": 40},
            {"name": "screen11", "row": 2, "col": 2, "width": 50, "height": 40},
            {"name": "screen12", "row": 2, "col": 3, "width": 50, "height": 40},
            {"name": "screen13", "row": 2, "col": 4, "width": 50, "height": 40},
            {"name": "screen14", "row": 2, "col": 5, "width": 50, "height": 40}
        ]
        for screen_config in frame_config:
            # 创建框架作为屏幕
            frame = tk.Frame(
                frame_button,
                width=screen_config["width"],
                height=screen_config["height"],
                bg="lightgray",
                bd=2,
                relief="ridge"
            )
            
            # 放置框架
            frame.grid(row=screen_config["row"], column=screen_config["col"], sticky="", padx=10, pady=10)  
            
            # 创建按钮框架
            button_frame = Button(frame)    
            # 保存屏幕信息
            self.buttons.append(button_frame)
        button_configs = [
            {"action_name":"aruco_get",     "action_text":"Start",          "button_name":"启动手眼标定",   "pressed_button_name":"启动手眼标定",  "close_frame":False,"open_frame":self.hand_eye_key},
            {"action_name":"joy",           "action_text":"JOY",            "button_name":"启动手柄控制",   "pressed_button_name":"结束手柄控制",  "close_frame":False,"open_frame":False},
            {"action_name":"go_to_start",   "action_text":"start",          "button_name":"回归原点",      "pressed_button_name":"回归原点",      "close_frame":False,"open_frame":False},
            {"action_name":"photo_get",     "action_text":"start",          "button_name":"获取点云路径",   "pressed_button_name":"获取点云路径",  "close_frame":False,"open_frame":False},
            {"action_name":"run_the_path",  "action_text":"path_run",       "button_name":"执行当前路径",   "pressed_button_name":"执行当前路径",  "close_frame":False,"open_frame":False},
            # {"action_name":"clear_the_path",  "action_text":"path_close",   "button_name":"放弃当前路径",   "pressed_button_name":"放弃当前路径",  "close_frame":False,"open_frame":False},
            {"action_name":"break_the_path",  "action_text":"path_break",   "button_name":"中断当前路径",   "pressed_button_name":"中断当前路径",  "close_frame":False,"open_frame":False},
            {"action_name":"data_save",        "action_text":"true",        "button_name":"开始保存数据",    "pressed_button_name":"结束保存数据",  "close_frame":False,"open_frame":False},
            {"action_name":"parameter",        "action_text":"up",          "button_name":"增加高度",       "pressed_button_name":"增加高度",     "close_frame":False,"open_frame":False},
            {"action_name":"parameter",        "action_text":"down",        "button_name":"降低高度",       "pressed_button_name":"降低高度",     "close_frame":False,"open_frame":False},
            {"action_name":"parameter",        "action_text":"fit",         "button_name":"放弃拟合曲线",    "pressed_button_name":"使用拟合曲线",  "close_frame":False,"open_frame":False},
            {"action_name":"parameter",        "action_text":"inhere_height",  "button_name":"固定高度",    "pressed_button_name":"结束固定高度","close_frame":False,"open_frame":False},
            {"action_name":"parameter",        "action_text":"all_direction",  "button_name":"开启全方位",   "pressed_button_name":"关闭全方位","close_frame":False,"open_frame":False},
            {"action_name":"parameter",        "action_text":"point_num_up",   "button_name":"增加打点数",   "pressed_button_name":"增加打点数","close_frame":False,"open_frame":False},
            {"action_name":"parameter",        "action_text":"point_num_down", "button_name":"减少打点数",   "pressed_button_name":"减少打点数","close_frame":False,"open_frame":False}
        ]
        for btn, button_config in zip(self.buttons,button_configs):
            btn.config(
                text=button_config["button_name"],
                press_text=button_config["button_name"],
                pressed_text = button_config["pressed_button_name"],
                command=lambda a=button_config["action_name"], m=button_config["action_text"], b=btn: self._action_callback(a, m, b),
                width=10,
                height=5,
                bg="#FFD700",
                activebackground="#FFEC8B",
                font=('Arial', 12) ,
                close_frame = button_config["close_frame"],
                open_frame = button_config["open_frame"]
            )
            btn.grid(row=0, column=0, sticky = "")
#################################回调函数#########################################
    def _Laction_callback(self, action_name, message, button, is_start):
        """处理长按按钮的开始/停止指令"""
        command_type = "开始" if is_start else "停止"
        self.root.after(0, lambda: 
                    self._append_status(f"{command_type}执行 {action_name}: {message}", 'info'))
        
        threading.Thread(
            target=self._execute_ros_laction,
            args=(action_name, message, button, is_start),
            daemon=True
        ).start()
    def _execute_ros_laction(self, action_name, message, button, is_start):
        """执行长按相关的ROS动作"""
        try:
            client = actionlib.SimpleActionClient(f'/{action_name}', MessageAction)
            
            if not client.wait_for_server(rospy.Duration(5)):
                self.root.after(0, lambda: self._append_status("服务器等待超时！", 'error'))
                return
            def feedback_cb(feedback):
                status = getattr(feedback, 'status', '执行中')
                self.root.after(0, lambda: self._append_status(f"{action_name}状态: {status}", 'info'))
            if is_start:
                # 发送开始指令
                goal = MessageGoal()
                goal.message = message
                client.send_goal(goal,feedback_cb=feedback_cb)
                self.root.after(0, lambda: 
                    self._append_status(f"{action_name} 开始指令已发送", 'success'))
            else:
                # 发送停止指令
                goal = MessageGoal()
                goal.message = message
                client.send_goal(goal,feedback_cb=feedback_cb)
                self.root.after(0, lambda: 
                    self._append_status(f"{action_name} 停止指令已发送", 'success'))
   
        except Exception as e:
            self.root.after(0, lambda: 
                self._append_status(f"{action_name} 操作失败: {str(e)}", 'error'))
    def _action_callback(self, action_name, message, button): # 所有普通按键处理函数
        """所有普通按键处理函数"""
        if action_name == "parameter":
            z_height = rospy.get_param("z_height")
            fit_line = rospy.get_param("fit_line")
            all_direction = rospy.get_param("all_direction")
            inhere_height = rospy.get_param("inhere_height")
            inhere_height_data = rospy.get_param("inhere_height_data")
            points_num = rospy.get_param("points_num")
            button.button_state = not button.button_state #按键状态取反            
            if message == "fit":
                fit_line = not fit_line
                self.root.after(0, lambda: self._append_status(f"是否使用拟合曲线: {fit_line}...", 'info'))
                rospy.set_param("fit_line",fit_line)
            elif message == "up":
                z_height += 0.001
                self.root.after(0, lambda: self._append_status(f"当前z轴高度为: {z_height:.3f}...", 'info'))
                rospy.set_param("z_height",z_height)
            elif message == "down":
                z_height -= 0.001
                self.root.after(0, lambda: self._append_status(f"当前z轴高度为: {z_height:.3f}...", 'info'))
                rospy.set_param("z_height",z_height)
            elif message == "inhere_height":
                z_height = rospy.get_param("z_height")
                inhere_height = not inhere_height
                self.root.after(0, lambda: self._append_status(f"是否使用固定高度扫描: {inhere_height},当前高度为{(inhere_height_data+z_height*1000)}...", 'info'))
                rospy.set_param("inhere_height",inhere_height)
            elif message == "all_direction":
                all_direction = not all_direction
                self.root.after(0, lambda: self._append_status(f"是否使用全方位扫描: {all_direction}...", 'info'))
                rospy.set_param("all_direction",all_direction)
            elif message == "point_num_up":
                points_num += 1
                self.root.after(0, lambda: self._append_status(f"当前打点数量为: {points_num:.3f}...", 'info'))
                rospy.set_param("points_num",points_num)
            elif message == "point_num_down":
                points_num -= 1
                self.root.after(0, lambda: self._append_status(f"当前打点数量为: {points_num:.3f}...", 'info'))
                rospy.set_param("points_num",points_num)

        elif action_name == None:
            pass
        else:
            button.config(state=tk.DISABLED)
            self.root.after(0, lambda: 
                        self._append_status(f"开始执行 {action_name}...", 'info'))
            threading.Thread(
                target=self._execute_ros_action,
                args=(action_name, message, button),
                daemon=True
            ).start()
    def _execute_ros_action(self, action_name, message, button): # 所有支线actin函数
        """所有支线action执行函数"""
        try:
            client = actionlib.SimpleActionClient(f'/{action_name}', MessageAction)

            if not client.wait_for_server(rospy.Duration(5)): # 5秒警告超时
                self.root.after(0, lambda: self._append_status("服务器等待超时！", 'error'))
                button.config(state=tk.NORMAL)
                return

            def feedback_cb(feedback):
                status = getattr(feedback, 'status', '执行中')
                if 'fail' in status:
                    self.root.after(0, lambda: self._append_status(f"{action_name} 失败: {status}", 'error'))
                    if button.open_frame != None:
                        button.open_frame.grid_forget()# 关闭本应打开的界面
                else :
                    self.root.after(0, lambda: 
                    self._append_status(f"{action_name}状态: {status}", 'info'))
            goal = MessageGoal()
            goal.message = message
            client.send_goal(goal, feedback_cb=feedback_cb)
            client.wait_for_result()
            result = client.get_result()
            reply = str(result.reply)
            if any(keyword in reply.lower() for keyword in ['fail', 'error', '无法']):
                self.root.after(0, lambda: 
                self._append_status(f"{action_name} 失败: {result}", 'error'))
            else:
                self.root.after(0, lambda: 
                self._append_status(f"{action_name} 完成: {result}", 'success'))
                button.button_state = not button.button_state #按键状态取反
            
                
        except Exception as e:
            self.root.after(0, lambda e=e:  # 通过默认参数捕获当前e的值
            self._append_status(f"{action_name} 失败: {str(e)}", 'error'))
        finally:
            button.config(state=tk.NORMAL)
    def _ros_init(self):
        rospy.Subscriber("/camera/cropped_image", RosImage, self._image_callback, queue_size=1)
        rospy.Subscriber('arm_pose', Float32MultiArray, self._pose_callback)
        rospy.Subscriber("chat", String, self._text_callback, queue_size=1)
        # 显示初始图片作为初始界面
        pkg_path = rospack.get_path('app_pkg')
        image_path = os.path.join(pkg_path, 'config/start.jpg')
        start_img = cv2.imread(image_path)
        start_img = cv2.cvtColor(start_img, cv2.COLOR_BGR2RGB)
        start_img = cv2.resize(start_img, (1000, 1000))
        img = Image.fromarray(start_img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

        rospy.spin()
    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (1000, 1000))
            try:
                self.image_queue.put_nowait(rgb_image)
                # self._append_status("成功接收并缓存图像帧", 'success')
            except queue.Full:
                # self._append_status("图像队列已满，跳过当前帧", 'error')
                pass
        except Exception as e:
            error_msg = f"图像处理失败: {str(e)}"
            rospy.logerr(error_msg)
            self._append_status(error_msg, 'error')
    def _pose_callback(self, data):
        self.label_x.config(text=f"X: {data.data[0]:.3f}")
        self.label_y.config(text=f"Y: {data.data[1]:.3f}")
        self.label_z.config(text=f"Z: {data.data[2]:.3f}")
        self.label_rx.config(text=f"RX: {data.data[3]:.3f}")
        self.label_ry.config(text=f"RY: {data.data[4]:.3f}")
        self.label_rz.config(text=f"RZ: {data.data[5]:.3f}")
    def _text_callback(self,msg):
        self.root.after(0, lambda: 
                    self._append_status(msg.data))
    def _append_status(self, message, tag='info'):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {message}\n"
        print(message)
        self.status_text.configure(state='normal')
        self.status_text.insert(tk.END, full_msg, tag)
        self.status_text.see(tk.END)
        self.status_text.configure(state='disabled')
    def _update_gui(self):
        try:
            if  not self.image_queue.empty():
                img_data = self.image_queue.get_nowait()
                img = Image.fromarray(img_data)
                imgtk = ImageTk.PhotoImage(image=img)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)
        except queue.Empty:
            pass
        
        self.root.after(50, self._update_gui)
    def _clear_queue(self):
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
    def _kill_node(self, node_name):
        import subprocess
        try:
            full_name = subprocess.check_output(
                f"rosnode list | grep {node_name}", 
                shell=True
            ).decode().strip()
            
            if full_name:
                subprocess.run(["rosnode", "kill", full_name])
                print(f"已关闭节点: {full_name}")
            else:
                print(f"未找到节点: {node_name}")
        except Exception as e:
            print(f"关闭失败: {str(e)}")
    def _on_close(self):
        from tkinter import messagebox
        if messagebox.askokcancel("退出程序", "确定要关闭所有ROS节点和窗口吗？"):
            try:
                self._kill_node("TaskController")
                self._kill_node("interface_gui")
                self._kill_node("rvc_aruco_get")
                self._kill_node("eye_to_hand")
                self._kill_node("RVC_camera")
                self._kill_node("rvc_path")
                self._kill_node("data_save")
                self.displaying = False
                self._clear_queue()
                rospy.signal_shutdown("用户请求关闭")
                time.sleep(1)
                
                if hasattr(self, 'ros_thread') and self.ros_thread.is_alive():
                    rospy.loginfo("等待ROS线程结束...")
                    self.ros_thread.join(timeout=2.0)
                    
                self.root.destroy()
                rospy.loginfo("程序已安全退出")
                
            except Exception as e:
                rospy.logerr(f"关闭过程中出错: {e}")
                self.root.destroy()
        else:
            rospy.loginfo("用户取消了关闭操作")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    viewer = RosTkViewer()
    viewer.run()