import serial
import time
import struct
from typing import NamedTuple
import rospy
from typing import NamedTuple, List  # 添加List以兼容旧版Python
import geometry_msgs.msg
import nav_msgs.msg

path_pub = rospy.Publisher('path_cyl',nav_msgs.msg.Path,queue_size=10)
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
    tools: List[AuroraToolData]  # 使用typing.List替代list[]以兼容旧版Python
    system_status: int
    crc: str

def decode_tx_response(raw_data: str) -> AuroraResponse:
    """解码NDI Aurora TX命令响应"""
    if not raw_data.startswith(tuple(str(i) for i in range(10))):  # 更Pythonic的数字检查方式
        raise ValueError("无效的响应格式")

    lines = raw_data.splitlines()
    if len(lines) < 2:
        raise ValueError("响应数据不完整")

    # 解析工具数据
    tools = []
    try:
        num_handles = int(lines[0][:2], 16)  # 第一行前两位是工具数量
    except ValueError:
        raise ValueError("无效的工具数量格式")

    for i in range(min(num_handles, len(lines)-1)):  # 防止索引越界
        tool_line = lines[i]
        if len(tool_line) < 71:  # 确保有足够长度
            continue
            
        try:
            handle = int(tool_line[2:4], 16)
            
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
        except (ValueError, IndexError) as e:
            print(f"工具{i}数据解析错误: {e}")
            continue

    # 系统状态和CRC
    try:
        system_status = int(lines[-1][:4], 16) if lines[-1] else 0
        crc = lines[-1][4:8] if len(lines[-1]) >= 8 else ""
    except (ValueError, IndexError):
        system_status = 0
        crc = ""

    return AuroraResponse(tools, system_status, crc)

def parse_port_status(status: int) -> dict:
    """解析端口状态位（增强版）"""
    return {
        'occupied': bool(status & 0x1),
        'gpio1': bool(status & 0x2),
        'gpio2': bool(status & 0x4),
        'gpio3': bool(status & 0x8),
        'initialized': bool(status & 0x10),
        'enabled': bool(status & 0x20),
        'out_of_volume': bool(status & 0x40),
        'partially_out': bool(status & 0x80),
        'broken_coil': bool(status & 0x100),
        'current_sensing': bool(status & 0x200)
    }

def num_show(sample_data: str) -> None:
    """增强版数据显示函数"""
    try:
        response = decode_tx_response(sample_data)
        print(f"\n=== NDI Aurora 数据报告 ===")
        print(f"系统状态: 0x{response.system_status:04X}")
        print(f"CRC校验: {response.crc or '无'}")
        
        if not response.tools:
            print("\n警告: 未检测到有效工具数据")
            return
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
            print(f"  位置: X={tool.tx:>7.2f}mm  Y={tool.ty:>7.2f}mm  Z={tool.tz:>7.2f}mm")
            print(f"  旋转: Q0={tool.q0:.4f}  QX={tool.qx:.4f}  QY={tool.qy:.4f}  QZ={tool.qz:.4f}")
            print(f"  误差指标: {tool.error:.1f} μm")
            
            status = parse_port_status(tool.port_status)
            print("  状态:")
            print(f"    • 启用状态: {'已启用' if status['enabled'] else '未启用'}")
            print(f"    • 体积状态: {'在体积内' if not status['out_of_volume'] else '超体积'}")
            print(f"    • 初始化: {'已完成' if status['initialized'] else '未完成'}")
            if status['broken_coil']:
                print("    ! 线圈故障")
            if status['partially_out']:
                print("    ! 部分超出体积")
        return pose_goal  
    except ValueError as e:
        print(f"\n数据解析错误: {e}")
class ndi_aurora:
    def __init__(self):
        # 配置串口参数
        rospy.init_node('ndi', anonymous=True)
        self.SERIAL_PORT = '/dev/ttyUSB0'  # 根据实际情况修改（如 /dev/ttyUSB0）
        self.BAUD_RATE = 9600
        self.TIMEOUT = 1  # 秒
        self.path_msgs = nav_msgs.msg.Path()
        self.path_msgs.header.frame_id = "base_link"
        self.path_msgs.header.stamp = rospy.Time.now()
        self.pub = rospy.Publisher('path_cyl',nav_msgs.msg.Path,queue_size=10)
    def send_command(self,command,ser,times = 0.001,hex = False):
        """发送指令 指令间隔用空格，末尾用\r"""
        ser.write(command.encode('ascii'))
        print(f"已发送命令: {command.strip()}")
        while True :
            if ser.in_waiting > 0:
                time.sleep(0.1)
                time.sleep(times)
                data = ser.read(ser.in_waiting)
                break
        if hex:
            print("收到原始数据(16进制):", data.hex())  # 关键：以16进制显示
            return data.hex()  # 返回字节对象，不解码
        received_data = data.decode('ascii')
        print("收到：",received_data)
        return received_data # 原样输出
    def serial_init(self): # 初始化

        try:
            # 打开串口连接
            with serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=self.TIMEOUT) as ser:
                print(f"已连接到 {self.SERIAL_PORT}，波特率 {self.BAUD_RATE}")
                ###############重置####################
                echo = self.send_command("RESET 0\r",ser)
                echo = self.send_command("APIREV \r",ser)
                echo = self.send_command("VER:4A6EF\r",ser)
                echo = self.send_command("COMM 70001\r",ser)
                echo = self.send_command("VER:5662E\r",ser)
                echo = self.send_command("GET:Device.*722D\r",ser)
                echo = self.send_command("INIT:E3A5\r",ser)
                echo = self.send_command("PHSR:0020FF\r",ser)
                echo = self.send_command("PHRQ:*********1****A4C1\r",ser)
                echo = self.send_command("PDIS:011266\r",ser)
                echo = self.send_command("PVWR:0100004E444900AF12000001000000000000010000000002DC32355A00000004000000040000000000403F000000000000000000000000000000000000000000000000913D\r",ser)
                echo = self.send_command("PVWR:0100400000204100000000000000000000000000000000000000001F853D4285EBE74100000000000000003333B24200000000A4700DC2A4700D420000000000000000DB4C\r",ser)
                echo = self.send_command("PVWR:01008000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ADDC\r",ser)
                echo = self.send_command("PVWR:0100C0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009EC7\r",ser)
                echo = self.send_command("PVWR:01010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000097B9\r",ser)
                echo = self.send_command("PVWR:010140000000000000803F00000000000000000000803F00000000000000000000803F00000000000000000000000000000000000000000000000000000000000000007503\r",ser)
                echo = self.send_command("PVWR:010180000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006DE4\r",ser)
                echo = self.send_command("PVWR:0101C0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000005EFF\r",ser)
                echo = self.send_command("PVWR:0102000000000000000000000000000000000000000000000000000000000000000000000000000000000000010203000000000000000000000000000000001F1F1F1F30CC\r",ser)
                echo = self.send_command("PVWR:010240090000004E44490000000000000000003837303034343900000000000000000000000000090101010100000000000000000000000000000000010101010000007763\r",ser)
                echo = self.send_command("PVWR:010280000000000000000000000000008000290000000000000000000080BF000000000000000000000000000000000000000000000000000000000000000000000000DF87\r",ser)
                echo = self.send_command("PVWR:0102C0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000005EB4\r",ser)
                echo = self.send_command("PINIT:0131EA\r",ser)
                echo = self.send_command("PHINF:010035FCAD\r",ser)
                echo = self.send_command("PENA:01D6D3B\r",ser)
                echo = self.send_command("GET:*AC73\r",ser,times=3)
                echo = self.send_command("GETINFO:Info.Status.New Alerts33A3\r",ser)
                echo = self.send_command("GETINFO:Info.Status.Alerts340A\r",ser)
                echo = self.send_command("TSTART:5423\r",ser)
                # echo = self.send_command("TX 0801\r",ser,times=0.5)
                # echo = self.send_command("BX:080BE5AD\r",ser,times=0.5,hex=True)
                # echo = self.send_command("BX:080BE5AD\r",ser,times=0.5,hex=True)
                try:
                    while not rospy.is_shutdown():
                        echo = self.send_command("TX 0801\r",ser)

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
def main():
    ndi = ndi_aurora()
    ndi.serial_init()
if __name__ == "__main__":
    main()