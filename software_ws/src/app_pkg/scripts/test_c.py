import struct
from typing import NamedTuple

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

# 示例使用
if __name__ == "__main__":
    sample_data = """010A+03395+08404-04223+00000-005820-001514-014122+000000002003F00004326
000039BC"""
    
    try:
        response = decode_tx_response(sample_data)
        print(f"系统状态: 0x{response.system_status:04X}")
        print(f"CRC校验: {response.crc}")
        
        for tool in response.tools:
            print(f"\n工具 {tool.handle}:")
            print(f"  位置: X={tool.tx:.2f}mm, Y={tool.ty:.2f}mm, Z={tool.tz:.2f}mm")
            print(f"  旋转: Q0={tool.q0:.4f}, QX={tool.qx:.4f}, QY={tool.qy:.4f}, QZ={tool.qz:.4f}")
            print(f"  误差指标: {tool.error:.1f}")
            
            status = parse_port_status(tool.port_status)
            print("  状态:", 
                  "已启用" if status['enabled'] else "未启用",
                  "| 在体积内" if not status['out_of_volume'] else "| 超体积")
            
    except ValueError as e:
        print(f"解码错误: {e}")