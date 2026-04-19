import serial

# 配置串口参数，根据你的实际串口设置进行修改
ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=1)  # 串口名称、波特率和超时


# 函数用于计算校验字节
def calculate_checksum(data):
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


# 函数用于发送命令
def send_command(command_bytes):
    ser.write(command_bytes)
    ser.flush()


if __name__ == "__main__":
    try:
        # 命令示例: 设置当前位置为零点
        # zero_position_command = bytes.fromhex("01 0A 6D 6B")
        # send_command(zero_position_command)

        # 命令示例: 读取编码器值
        # read_encoder_value_command = bytes.fromhex("01 30 6B")
        # send_command(read_encoder_value_command)
        # response = ser.read(4).hex()  # 读取4字节响应数据，根据实际情况调整字节数

        # 命令示例: 读取输入脉冲数
        # read_input_pulse_command = bytes.fromhex("01 33 6B")
        # send_command(read_input_pulse_command)
        # response = ser.read(6).hex()  # 读取6字节响应数据，根据实际情况调整字节数

        # 命令示例: 读取电机实时位置
        # read_motor_position_command = bytes.fromhex("01 36 6B")
        # send_command(read_motor_position_command)
        # response = ser.read(6).hex()  # 读取6字节响应数据，根据实际情况调整字节数

        # 命令示例: 控制闭环电机的正反转速度
        # 地址 + 0xF6 + 方向和速度(共用 2 个字节) + 加速度 + 校验字节
        # 其中,第3，4两个字节表示方向和速度,最高的半字节 0x1（0x0） 表示方向,剩下的
        # 0x4FF 表示速度档位,最大为 4FF,即 1279 个速度档位
        # control_motor_command = bytes.fromhex("01 F6 00 00 00 6B")
        # send_command(control_motor_command)

        # 命令示例: 控制闭环电机的正反转位置
        # 地址 + 0xFD + 方向和速度(2 字节) + 加速度 + 脉冲数(3 字节) + 校验字节
        # 其中,第3，4两个字节表示方向和速度,最高的半字节 0x1（0x0） 表示方向,剩下的
        # 0x4FF 表示速度档位,最大为 4FF,即 1279 个速度档位
        # 加速度档位后面的 3 个字节是脉冲数,例如 16 细分下发送 3200（16*200） 个脉冲就可以让 1.8°（360/200）的电机转动一圈
        control_motor_command = bytes.fromhex("01 FD 10 0F 00 00 0C 80 6B")
        send_command(control_motor_command)

        # print(response)
    except KeyboardInterrupt:
        pass
    finally:
        # 关闭串口
        ser.close()
