import serial
import threading
import queue
import time

# 串口配置
SERIAL_PORT = 'COM1'  # 根据实际情况修改
BAUD_RATE = 9600

# 全局串口对象
try:
    global_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"串口 {SERIAL_PORT} 已打开")
except serial.SerialException as e:
    print(f"串口错误: {e}")
    global_serial = None

# 全局消息队列
message_queue = queue.Queue()

def serial_listener(queue):
    """
    串口监听线程，持续监听串口数据并将数据放入消息队列。
    """
    global global_serial
    if global_serial is None:
        print("串口未初始化，监听线程退出")
        return

    try:
        while True:
            if global_serial.in_waiting:
                message = global_serial.readline().decode('utf-8').strip()
                queue.put(message)
                print(f"收到消息: {message}")
            time.sleep(0.1)  # 避免CPU占用过高
    except serial.SerialException as e:
        print(f"串口错误: {e}")
    finally:
        if global_serial is not None and global_serial.is_open:
            global_serial.close()
            print("串口已关闭")

def message_processor(queue):
    """
    消息处理线程，从消息队列中获取消息并进行处理。
    """
    while True:
        try:
            message = queue.get()
            print(f"消息处理线程处理消息: {message}")
            # 在这里添加你的消息处理逻辑
            if message == "你好":
                print("收到问候，回复：你好！")
            elif message == "再见":
                print("收到告别，回复：再见！")
            else:
                print("收到未知消息，无法处理。")
            time.sleep(0.1)  # 避免CPU占用过高
        except Exception as e:
            print(f"消息处理错误: {e}")

def main():
    """
    主线程，负责启动串口监听线程和消息处理线程。
    """
    # 启动串口监听线程
    serial_thread = threading.Thread(target=serial_listener, args=(message_queue,))
    serial_thread.daemon = True  # 设置为守护线程，主线程退出时自动结束
    serial_thread.start()

    # 启动消息处理线程
    processor_thread = threading.Thread(target=message_processor, args=(message_queue,))
    processor_thread.daemon = True  # 设置为守护线程，主线程退出时自动结束
    processor_thread.start()

    try:
        while True:
            time.sleep(0.1)  # 避免CPU占用过高
    except KeyboardInterrupt:
        print("程序退出")
        if global_serial is not None and global_serial.is_open:
            global_serial.close()

if __name__ == "__main__":
    main()