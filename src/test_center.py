from find import Finder
from parma import Baseparma, CAPparma, Findparma
from Draw_circle import draw_circle
from tracker import calculate_relative_angles
from sender import Sender


import cv2
import serial
import queue
import time
import threading
import numpy as np

queue_up = queue.Queue(maxsize=1)
queue_down = queue.Queue(maxsize=1)
def identify():
    """
    识别图像中的目标点并计算相对角度。
    """
    finder = Finder()
    cap = cv2.VideoCapture(1)
    
    theta = 0.0  # 初始角度，避免第一次使用时未定义

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧！")
            break

        arrived1 = False
        arrived2 = False
        has_goal = False
        center_angle1, center_angle2 = 0.0, 0.0
        circle_angle1, circle_angle2 = 0.0, 0.0
        rect_points = [(0, 0), (0, 0), (0, 0), (0, 0)]
        center, has_goal, rect_points = finder.find_frame_center(frame, Findparma.hsv_lower, Findparma.hsv_upper)

        # 非阻塞更新 theta
        try:
            theta = queue_up.get_nowait()
        except queue.Empty:
            pass  # 如果没有新值，就用上一次的 theta

        if has_goal:
            center_angle1, center_angle2, arrived1 = calculate_relative_angles(center, CAPparma.image_size, CAPparma.h_fov, (0, 0.04, 0), 10)
            circle_point = draw_circle.transform_and_inverse(rect_points, theta)
            circle_angle1, circle_angle2, arrived2 = calculate_relative_angles(circle_point, CAPparma.image_size, CAPparma.h_fov, (0, 0.04, 0), 10)
            center_draw = (int(center[0]), int(center[1]))
            cv2.circle(frame, center_draw, radius=5, color=(0, 0, 255), thickness=-1)

        # 清空旧数据，仅保留最新一帧的数据
        if not queue_down.empty():
            try:
                queue_down.get_nowait()
            except queue.Empty:
                pass

        try:
            queue_down.put_nowait((center_angle1, center_angle2, arrived1, has_goal, circle_angle1, circle_angle2, arrived2))
        except queue.Full:
            pass

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def send_angle():
    sender_up = Sender(Baseparma.COM_UP, Baseparma.Baud_rate_up)
    sender_down = Sender(Baseparma.COM_DOWN, Baseparma.Baud_rate_down)
    last_sent_time = 0
    current_flag = 'm'  # 初始为0，表示未设置或同时发送
    thetas = np.linspace(0, 2*np.pi, 100)
    i = 0
    theta = thetas[i]
    queue_up.put(theta)
    while True:
        # 获取当前标志位（建议每次循环都检查）
        try:
            if sender_up.ser.in_waiting > 0:
                flag_byte = sender_up.ser.read().decode('utf-8').strip()




























                
                print(f"接收到标志位: {flag_byte}")
            if flag_byte in ('u', 'n','c'):
                current_flag = flag_byte
        except Exception as e:
            pass

        # 等待队列数据
        center_angle1, center_angle2, arrived1, has_goal, circle_angle1, circle_angle2, arrived2 = queue_down.get()
        now = time.time()

        if arrived1 and (now - last_sent_time > 2):
            try:
                center_angle1, center_angle2 = 0, 0
                sender_up.send_bytes(Baseparma.ID[0])
                last_sent_time = now
            except serial.SerialException as e:
                print(f"发送center标志失败: {e}")

        if arrived2 and (now - last_sent_time > 2):
            try:
                circle_angle1, circle_angle2 = 0, 0
                sender_up.send_bytes(Baseparma.ID[0])
                i += 1
                if i >= len(thetas):
                    i = 0
                queue_up.put(thetas[i])
                last_sent_time = now
            except serial.SerialException as e:
                print(f"发送circle标志失败: {e}")

        # 根据标志位选择发送
        if current_flag == 'u':
            sender_down.send_angles(center_angle1, center_angle2)
            print(f"[flag=1] 发送center角度: {center_angle1}, {center_angle2}")
        elif current_flag == 'n':
            sender_down.send_angles(circle_angle1, circle_angle2)
            print(f"[flag=2] 发送circle角度: {circle_angle1}, {circle_angle2}")
        elif current_flag == 'c':
            pass  # 如果标志位不是1或2，则跳过发送

def main():
    # 启动识别线程
    identify_thread = threading.Thread(target=identify)
    identify_thread.daemon = True
    identify_thread.start()

    # 启动发送角度线程
    send_angle_thread = threading.Thread(target=send_angle)
    send_angle_thread.daemon = True
    send_angle_thread.start()

    # 阻塞主线程
    identify_thread.join()
    send_angle_thread.join()

    
if __name__ == "__main__":
    main()
