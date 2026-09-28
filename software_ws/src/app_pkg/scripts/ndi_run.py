import serial
import time
import struct
from typing import NamedTuple
import geometry_msgs.msg
import nav_msgs.msg
import rospy

class AuroraToolData(NamedTuple):
    handle: int
    q0: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    error: float
    port_status: int
    frame_number: int
class AuroraResponse(NamedTuple):
    tools: list[AuroraToolData]
    system_status: int
    crc: str
def decode_tx_response(raw_data: str) -> AuroraResponse:
    """解码NDI Aurora TX命令响应"""
    if not raw_data.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
        raise ValueError("无效的响应格式")

    lines = raw_data.splitlines()
    if len(lines) < 2:
        raise ValueError("响应数据不完整")

    # 解析工具数据
    tools = []
    num_handles = int(lines[0][:2], 16)  # 第一行前两位是工具数量
    
    for i in range(num_handles):
        tool_line = lines[i]
        handle = int(tool_line[2:4], 16)  # 句柄号
        
        # 四元数和平移数据
        q0 = float(tool_line[4:10]) / 10000
        qx = float(tool_line[10:16]) / 10000
        qy = float(tool_line[16:22]) / 10000
        qz = float(tool_line[22:28]) / 10000
        tx = float(tool_line[28:35]) / 100
        ty = float(tool_line[35:42]) / 100
        tz = float(tool_line[42:49]) / 100
        error = float(tool_line[49:55]) / 10000
        
        # 端口状态和帧号
        port_status = int(tool_line[55:63], 16)
        frame_number = int(tool_line[63:71], 16)
        
        tools.append(AuroraToolData(
            handle, q0, qx, qy, qz, tx, ty, tz, 
            error, port_status, frame_number
        ))
    # 系统状态和CRC
    system_status = int(lines[-1][:4], 16)
    crc = lines[-1][4:8] if len(lines[-1]) >= 8 else ""
    return AuroraResponse(tools, system_status, crc)
def parse_port_status(status: int) -> dict:
    """解析端口状态位"""
    return {
        'occupied': bool(status & 0x1),
        'initialized': bool(status & 0x10),
        'enabled': bool(status & 0x20),
        'out_of_volume': bool(status & 0x40),
        'partially_out': bool(status & 0x80),
        'broken_coil': bool(status & 0x100)
    }
def num_show(sample_data):
    response = decode_tx_response(sample_data)
    print(f"系统状态: 0x{response.system_status:04X}")
    print(f"CRC校验: {response.crc}")
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal = geometry_msgs.msg.PoseStamped()
    pose_goal.header.frame_id = "base_link"
    pose_goal.header.stamp = rospy.Time.now()
    for tool in response.tools:
        pose_goal.pose.position.x = tool.tx*0.001
        pose_goal.pose.position.y = tool.ty*0.001
        pose_goal.pose.position.z = -tool.tz*0.001
        pose_goal.pose.orientation.x = tool.qx
        pose_goal.pose.orientation.y = tool.qy
        pose_goal.pose.orientation.z = tool.qz
        pose_goal.pose.orientation.w = tool.q0
        print(f"\n工具 {tool.handle}:")
        print(f"  位置: X={tool.tx:.2f}mm, Y={tool.ty:.2f}mm, Z={tool.tz:.2f}mm")
        print(f"  旋转: Q0={tool.q0:.4f}, QX={tool.qx:.4f}, QY={tool.qy:.4f}, QZ={tool.qz:.4f}")
        print(f"  误差指标: {tool.error:.1f}")
        status = parse_port_status(tool.port_status)
        print("  状态:", 
                "已启用" if status['enabled'] else "未启用",
                "| 在体积内" if not status['out_of_volume'] else "| 超体积")
    return pose_goal
class ndi_aurora:
    def __init__(self):
        rospy.init_node('ndi', anonymous=True)
        # 配置串口参数
        self.SERIAL_PORT = '/dev/ttyUSB0'  # 根据实际情况修改（如 /dev/ttyUSB0）
        self.BAUD_RATE = 9600
        self.TIMEOUT = 1  # 秒
        self.path_msgs = nav_msgs.msg.Path()
        self.path_msgs.header.frame_id = "base_link"
        self.path_msgs.header.stamp = rospy.Time.now()
        self.pub = rospy.Publisher('path_cyl',nav_msgs.msg.Path,queue_size=10)
    def send_command(self,command,ser):
        """发送指令 指令间隔用空格，末尾用\r"""
        ser.write(command.encode('ascii'))
        # print(f"已发送命令: {command.strip()}")
        while True :
            if ser.in_waiting > 0:
                time.sleep(0.1)
                data = ser.read(ser.in_waiting)
                break
        received_data = data.decode('ascii')
        # print("收到：",received_data)
        return received_data # 原样输出
    def serial_init(self): # 初始化

        try:
            # 打开串口连接
            with serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=self.TIMEOUT) as ser:
                print(f"已连接到 {self.SERIAL_PORT}，波特率 {self.BAUD_RATE}")
                ###############重置####################
                echo = self.send_command("RESET 0\r",ser)
                if 'ERROR' in echo:
                    print("重置失败，再次尝试")
                    echo = self.send_command("RESET 1\r",ser)
                    if 'ERROR' in echo:
                        print("重置失败，请手动重启")
                        return 
                print("重置成功，开始初始化")
                #############初始化##################
                echo = self.send_command("INIT \r",ser)
                if 'ERROR' in echo:
                    print("初始化失败")
                    return
                print("初始化成功，开始检查通道")
                ##############检测通道##################
                echo = self.send_command("PHSR 01\r",ser)
                if '001414' in echo:
                    print("没有通道")
                    echo = self.send_command("PHF 0A\r",ser)
                    if 'ERROR' in echo:
                        print("通道设置失败")
                        return
                print("设置通道成功")
                #############通道初始化####################
                echo = self.send_command("PHSR 02\r",ser)
                if '010A001C1B5' in echo:
                    print("通道没有初始化")
                    echo = self.send_command("PINIT 0A\r",ser)
                    if 'ERROR' in echo:
                        print("通道初始化失败")
                        return
                print("通道初始化成功")
                ################使能通道###############
                echo = self.send_command("PHSR 03\r",ser)
                if '010A01F77F4' in echo:
                    print("通道没有使能")
                    echo = self.send_command("PENA 0AD\r",ser)
                    if 'ERROR' in echo:
                        print("通道使能失败")
                        return
                print("通道使能成功")
                #################开启轨迹模式#################
                echo = self.send_command("TSTART 40\r",ser)
                if 'ERROR' in echo:
                    print("开启失败")
                    return
                print("轨迹模式开启成功")
                try:
                    while not rospy.is_shutdown():
                        echo = self.send_command("TX 0001\r",ser)
                        if 'MISSING' in echo :
                            print("超出活动范围了")
                            continue
                        pose = num_show(echo)
                        if len(self.path_msgs.poses) >= 50:
                            self.path_msgs.poses.pop(0)
                        self.path_msgs.poses.append(pose)
                        self.pub.publish(self.path_msgs)
                        time.sleep(0.025)  # 避免过高频率（100ms间隔）
                except KeyboardInterrupt:
                    print("\n检测到 Ctrl+C，正在安全退出...")
                
        except serial.SerialException as e:
            print(f"串口错误: {e}")
        except Exception as e:
            print(f"其他错误: {e}")
    def run(self):
        try:
            while True:
                echo = self.send_command("TX 0001\r")
                self.num_show(echo)
                time.sleep(0.1)  # 避免过高频率（100ms间隔）
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在安全退出...")
def main():
    ndi = ndi_aurora()
    ndi.serial_init()
if __name__ == "__main__":
    main()