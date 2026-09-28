#!/bin/bash
LOG_FILE="$HOME/software_ws/src/app_pkg/log/data.log"
exec &> >(tee -a "$LOG_FILE")

echo "=== 启动时间: $(date) ==="

# 设置关键环境变量
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/
export DISPLAY=:0
# 这里是产生 No module named 'rospkg'的主要原因
export PATH=$PATH:/home/admind/anaconda3/bin/activate
export PYTHONPATH=/home/admind/anaconda3/envs/dvt/lib/python3.9/site-packages:$PYTHONPATH
# ROS初始化
source /opt/ros/noetic/setup.bash
source ~/software_ws/devel/setup.bash
echo $PYTHONPATH
# 进程管理
{
    echo "正在启动ROS核心..."
    roscore &
    CORE_PID=$!
    sleep 2  # 确保核心启动完成
    
    echo "正在启动节点..."
    roslaunch app_pkg other_test.launch &
    NODE_PID=$!
    echo "正在启动GUI..."
    roslaunch app_pkg interface_gui.launch
    NODE2_PID=$!
    # 清理
    kill  $CORE_PID 
    kill  $NODE_PID 
    kill  $NODE2_PID 
    wait
}

echo "=== 结束时间: $(date) ==="