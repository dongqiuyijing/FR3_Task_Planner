def parse_hex_frame(hex_str):
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
        data_len_dec = int(data_len_hex, 16)
        data_dec = [int(data_field[i:i+2], 16) for i in range(0, len(data_field), 2)]

        return data_dec if len(data_dec) > 1 else data_dec[0]
    except Exception as e:
        raise ValueError(f"解析失败: {str(e)}")

# 使用示例
frame = "BB0101131099669669"
parsed = parse_hex_frame(frame)
print(parsed)
