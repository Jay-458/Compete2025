
import time
import serial





def angle_to_bytes(angle):
    """将角度转换为符号、度数和小数部分的元组"""
    sign = 1 if angle >= 0 else 0
    degrees = int(abs(angle))
    decimal = int(round((abs(angle) - degrees) * 100))
    return sign, degrees, decimal

def pack_angles(angle1, angle2):
    """按照协议打包两个角度为字节格式"""
    # 帧头：固定为0xAA
    packet = bytearray([0xFE])
    
    # 处理第一个角度
    sign1, degrees1, decimal1 = angle_to_bytes(angle1)
    packet.append(sign1)
    packet.append(degrees1)
    packet.append(decimal1)
    
    # 处理第二个角度
    sign2, degrees2, decimal2 = angle_to_bytes(angle2)
    packet.append(sign2)
    packet.append(degrees2)
    packet.append(decimal2)
    
    # 帧尾：固定为0x55
    packet.append(0xFD)
    
    return bytes(packet)

# 示例使用
if __name__ == "__main__":
    ser = serial.Serial("COM11",9600)
    # 输入两个角度
    angle1 = 45.01
    angle2 = 54.96
    
    # 打包成字节
    packet = pack_angles(angle1, angle2)
    # print(packet)
    
    # 输出结果
    print(f"打包后的字节: {packet.hex(' ')}")
    # print(f"角度1: {angle1}, 符号: {1 if angle1 >= 0 else 0}, 度数: {int(abs(angle1))}, 小数: {int(round((abs(angle1) - int(abs(angle1))) * 100))}")
    # print(f"角度2: {angle2}, 符号: {1 if angle2 >= 0 else 0}, 度数: {int(abs(angle2))}, 小数: {int(round((abs(angle2) - int(abs(angle2))) * 100))}")    
    while True:
        ser.write(packet)
        if ser.in_waiting > 0:
            get_data = ser.readline(ser.in_waiting).decode('utf-8').strip()
            print(f"收到的原始数据为{get_data}")
        print(packet)
        time.sleep(1)
    print(result.decode('utf-16').strip())



