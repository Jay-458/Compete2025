from parma import Baseparma
from sender import Sender

import serial
import time
import threading
import queue

#创建消息通道
message_queue1 = queue.Queue()

#创建串口
try:
    ser_up = Sender(Baseparma.COM_UP, Baseparma.Baud_rate_up)
    
    print(f"串口{Baseparma.COM_UP} 已打开")
except serial.SerialException as e:
    print(f"串口错误:{e}")
    ser_up = None



#串口监听
def serial_listener(queue):
    """
    串口监听线程，持续监听串口数据并将数据放入消息队列。
    """
    global ser_up
    if ser_up is None:
        print("串口未初始化，监听线程退出")
        return
    try:
        while True:
            if ser_up.ser.in_waiting:
                message = ser_up.ser.readline(ser_up.ser.in_waiting).decode('utf-8',errors='replace').strip()
                message_queue1.put(message)
                print(f"收到消息: {message}")
            time.sleep(0.1)  # 避免CPU占用过高
    except serial.SerialException as e:
        print(f"串口错误: {e}")


#信息处理
def message_processor(ser,queue):
    try:
        while True:
            message = queue.get()
            queue.queue.clear() 
            if message[0] == 'm':
                ser.send(Baseparma.ID[0])
                print(f"进入mode1")
                message = None
                while True:
                    message = queue.get()
                    if message[0] == 'd':
                        ser.send(Baseparma.ID[2])
                        message = None                        
                        break
                    elif message[0] == 'm':
                        print(f"收到m")
                        ser.send(Baseparma.ID[3])  
                        message = None
                    else:
                        continue   
            elif message[0] == 'u':
                ser.send(Baseparma.ID[1])
                
            message = None
        






        time.sleep(0.1)
    except Exception as e:
        print(f"消息处理错误:{e}")

def main():
    #启动串口监听线程
    serial_thread = threading.Thread(target=serial_listener,args=(message_queue1,))
    serial_thread.daemon = True
    serial_thread.start()

    #启动消息处理线程
    processor_thread = threading.Thread(target=message_processor,args=(ser_up,message_queue1))
    processor_thread.daemon =True
    processor_thread.start()

    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 清理资源
        if ser_up and ser_up.ser.is_open:
            ser_up.ser.close()
            print("串口已关闭")

if __name__ =="__main__":
    main()
    # serial_listener(message_queue1)


