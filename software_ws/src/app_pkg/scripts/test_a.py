import serial
import time


class serial_communication:
    def __init__(self):
        port = '/dev/ttyUSB0'  # 修改为你的串口设备
        baudrate = 115200
        # 初始化串口
        try :
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1  # 读超时1秒
            )
        except serial.SerialException as e:
            print(f"串口错误: {e}")
        except ValueError as e:
            print(f"数据格式错误: {e}")
        finally:
            if 'ser' in locals() and self.ser.is_open:
                self.ser.close()
                print("串口已关闭")
    def serial_run(self):
        #####################################################
        print("设置波长")
        HEX_DATA = "BB0101160199669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00011699669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        if num == 1 :
            sum = 1064
        elif num == 2:
            sum = 755
        print("设置波长，当前波长",sum)
        ####################################################
        print("设置能量")
        HEX_DATA = "BB01021101F499669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00021199669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("能量设置情况，当前能量为：",num)
        #####################################################
        print("设置频率")
        HEX_DATA = "BB0101130A99669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00011399669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("频率设置情况，当前频率为：",num)
        #####################################################
        print("工作准备信号")
        HEX_DATA = "BB0101600199669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00016099669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("工作准备信号反馈",num)
        #####################################################
        HEX_DATA = "BB00015F99669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data) 
        print("预燃开启情况",num)
        time.sleep(5)
        #####################################################
        HEX_DATA = "BB00016299669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data) 
        print("故障报警信息",num)
        #####################################################
        print("脚踏出光")
        HEX_DATA = "BB0101610099669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00016199669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("脚踏出光",num)
        time.sleep(10)
        #####################################################
        print("脚踏出光")
        HEX_DATA = "BB0101610199669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00016199669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("脚踏出光",num)
        #####################################################
        print("脚踏出光")
        HEX_DATA = "BB0101610199669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00016199669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("脚踏出光",num)
        #####################################################
        print("脚踏出光")
        HEX_DATA = "BB0101610199669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00016199669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("脚踏出光",num)
        #####################################################
        print("脚踏出光")
        HEX_DATA = "BB0101610199669669"  # 要发送的16进制数据
        self.send_hex(HEX_DATA)
        HEX_DATA = "BB00016199669669"  # 要发送的16进制数据
        data = self.send_hex(HEX_DATA)
        num = self.parse_hex_frame(data)
        print("脚踏出光",num)
        time.sleep(10)
        




    def parse_hex_frame(self,hex_str):
        """解析自定义16进制协议帧并输出10进制结果"""
        if len(hex_str) < 10 or not hex_str.startswith('BB'):
            raise ValueError("无效数据帧格式")

        try:
            # 提取各字段（16进制原始值）
            header = hex_str[0:2]
            rw_flag = hex_str[2:4]
            data_len_hex = hex_str[4:6]
            address = hex_str[6:8]
            data_field = hex_str[8:8+int(data_len_hex, 16)*2]
            footer = hex_str[-8:]
            # 转换为10进制
            data_dec = int(data_field, 16)
    
            return data_dec
        except Exception as e:
            raise ValueError(f"解析失败: {str(e)}")
    def send_hex(self,hex_data_to_send):
        # 转换16进制字符串为字节数据
        byte_data = bytes.fromhex(hex_data_to_send.replace(" ", ""))
        
        # print(f"发送 (16进制): {' '.join(f'{b:02X}' for b in byte_data)}")
        # 发送数据
        self.ser.write(byte_data)
        # 等待数据发送完成
        time.sleep(1)
        # 接收数据
        received_data = self.ser.read(self.ser.in_waiting or 1)
        if received_data:
            data =  ''.join(f'{b:02X}' for b in received_data)
            # print(data)
            return data
        else:
           pass


if __name__ == "__main__":
    uart = serial_communication()
    uart.serial_run()
   
    