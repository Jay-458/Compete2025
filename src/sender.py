from parma import Baseparma
from pakgage import angle_to_bytes,pack_angles

import serial
import time
import queue




class Sender :
    def __init__(self,port,baudrate):
        self.Baud_rate = baudrate
        self.COM = port
        self.ser = serial.Serial(port=port,baudrate=baudrate)



    def INFO(self):
        print(Baseparma.COM)

    def send(self,date):
        self.ser.write(date)
        # print(f"已发送:{date}")
    def send_bytes(self, data):
        self.ser.write(data.encode('utf-8')) 
    def get(self):
        while True:
            if self.ser.in_waiting > 0:
                    # 读取数据
                    get_data = self.ser.readline(self.ser.in_waiting)
                    print(f"收到的原始数据为{get_data}")

                    try:
                        decoded_data = get_data.decode('utf-8').strip()
                        print(f"收到数据: {decoded_data}")
                        # return decoded_data
                    except UnicodeDecodeError:
                        print(f"解码错误: 数据无法以 {decode} 格式解码")
                        print(f"原始数据 (十六进制): {data.hex()}")
                        return None

    def serial_listener(self,queue):
        try:
            while True:
                if self.ser.in_waiting > 0:
                    message = self.ser.readline().decode('utf-8').strip()
                    queue.put(message)
                    print(f"收到消息:{message}")
                time.sleep(0.1)
        
        except serial.SerialException as e:
            print(f"串口错误:{e}")

    def send_angles(self,angle1,angle2):
        
        angles = pack_angles(angle1,angle2)
        self.send(angles)
        # print(f"已发送:{angles}")
                    
        


if __name__ == "__main__":
    message_queue = queue.Queue()
    sender1 = Sender(Baseparma.COM,Baseparma.Baud_rate)
    sender1.INFO()
    # date = Baseparma.ID[0]
    sender1.send(date)
    sender1.get()
