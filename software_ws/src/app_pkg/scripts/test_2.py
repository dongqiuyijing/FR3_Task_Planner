#!/usr/bin/env python3
import sys
import rospy
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                           QWidget, QPushButton, QLabel, QLineEdit, QHBoxLayout)
from PyQt5.QtCore import QTimer
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget
from rviz import bindings as rviz

class RvizEmbeddedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        rospy.init_node('embedded_rviz_py_path_display')
        
        self.setWindowTitle('Rviz Python 嵌入式路径显示')
        self.setGeometry(300, 300, 1000, 800)
        
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        self.topic_label = QLabel('路径话题:')
        self.topic_edit = QLineEdit('/path')
        control_layout.addWidget(self.topic_label)
        control_layout.addWidget(self.topic_edit)
        
        self.display_button = QPushButton('显示路径')
        self.display_button.clicked.connect(self.display_path)
        control_layout.addWidget(self.display_button)
        
        self.status_label = QLabel('状态: 等待操作')
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        
        # 右侧Rviz面板
        try:
            self.rviz_widget = rviz.VisualizationFrame()
            self.rviz_widget.setSplashPath("")
            self.rviz_widget.initialize()
            self.rviz_widget.setStatusBar(None)
            self.rviz_widget.setMenuBar(None)
            
            manager = self.rviz_widget.getManager()
            manager.setFixedFrame("map")
            
            # 添加Path显示
            path_display = manager.createDisplay("rviz/Path", "Path", True)
            path_display.subProp("Topic").setValue(self.topic_edit.text())
            path_display.subProp("Color").setValue("255; 0; 0")
            
            # 设置初始视角
            view_manager = manager.getViewManager()
            view_manager.setCurrentViewControllerType("rviz/Orbit")
            view = view_manager.getCurrent()
            view.subProp("Distance").setValue(10)
            
        except Exception as e:
            self.rviz_widget = QLabel(f"无法初始化Rviz: {str(e)}")
            self.status_label.setText(f"错误: {str(e)}")
        
        # 组合布局
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(self.rviz_widget, stretch=4)
        main_widget.setLayout(main_layout)
        
        # 路径发布者
        self.path_pub = None
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"
        
        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_path)
    
    def display_path(self):
        """开始显示路径"""
        topic_name = self.topic_edit.text().strip()
        if not topic_name:
            self.status_label.setText('状态: 请输入有效的话题名称')
            return
        
        self.path_pub = rospy.Publisher(topic_name, Path, queue_size=10)
        self.status_label.setText(f'状态: 正在发布路径到 {topic_name}')
        
        # 更新Rviz中的话题
        try:
            manager = self.rviz_widget.getManager()
            for display in manager.getRootDisplayGroup().getDisplays():
                if display.getName() == "Path":
                    display.subProp("Topic").setValue(topic_name)
        except:
            pass
        
        self.path_msg.poses = []
        self.timer.start(100)
    
    def update_path(self):
        """更新路径"""
        if not self.path_pub:
            return
            
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        
        t = rospy.Time.now().to_sec()
        pose.pose.position.x = 2.0 * math.cos(t)
        pose.pose.position.y = 2.0 * math.sin(t)
        pose.pose.orientation.w = 1.0
        
        self.path_msg.poses.append(pose)
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path_msg)
    
    def closeEvent(self, event):
        self.timer.stop()
        if self.path_pub:
            self.path_pub.unregister()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RvizEmbeddedWindow()
    window.show()
    
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(50)
    
    sys.exit(app.exec_())