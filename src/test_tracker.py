from sender import Sender
from parma import Baseparma,CAPparma,Findparma
from find import Finder
from tracker import calculate_relative_angles

import serial 
import time
import threading
import cv2
import numpy as np

angle1 = 0
angle2 = 0

def smooth_angle(prev_angle, current_angle, alpha):
    """
    对已归一到 [-180°, 180°] 范围的角度进行平滑处理。
    
    prev_angle: 上一时刻平滑后的角度（单位：度）
    current_angle: 当前的角度测量值（单位：度）
    alpha: 平滑系数，范围 0~1（越小越平滑）
    """
    diff = current_angle - prev_angle
    # 保证最小角度差值（考虑 -180 到 180 跨越）
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    
    smoothed = prev_angle + alpha * diff

    # 保持输出在 [-180, 180]
    if smoothed > 180:
        smoothed -= 360
    elif smoothed < -180:
        smoothed += 360

    return smoothed

def find():
    global angle1,angle2,prev_angle1,prev_angle2
    angle1,angle2 = 0,0
    prev_angle1,prev_angle2 = 0,0
    finder = Finder()
    cap = cv2.VideoCapture(1)
    lower_color = np.array([115,63,92])
    upper_color = np.array([164,226,255])
    min_area = 100
    while True:
        # 读取帧
        ret, frame = cap.read()  # ret: 是否成功, frame: 图像帧
        
        if not ret:
            print("无法读取帧！")
            break
        contours,color_mask = finder.find_color(frame,lower_color,upper_color)
        has_valid_point = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # 计算轮廓的矩
                M = cv2.moments(contour)
                
                # 计算中心坐标
                if M["m00"] != 0:  # 避免除以零（处理极小或空轮廓）
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    has_valid_point = True
                    point = (cx, cy)
                    pre_angle1,pre_angle2 = angle1,angle2
                    angle1,angle2 = calculate_relative_angles(point, (Findparma.frame_width,Findparma.frame_height), 100, (0,0.03,0))
                    # 在图像上绘制中心点
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 绿色实心圆
                    cv2.putText(frame, f"({cx}, {cy})", (cx+10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # base_frame = finder.base_process(frame)
        # 显示帧
        if not has_valid_point:
            # 或者设置为默认值
            pre_angle1,pre_angle2 = 0,0
            angle1, angle2 = 0, 0
            
        cv2.imshow('Camera', frame)
        cv2.imshow('mask',color_mask)
        # time.sleep(0.5)
        
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def send_angle():
    global angle1,angle2,prev_angle1,prev_angle2
    
    ser = Sender(Baseparma.COM,Baseparma.Baud_rate)
    while True:
        if abs(angle1) < 1.0 :
            angle1 = 0
        if abs(angle2) < 1.0 :
            angle2 =0

         
        send_angle1 = smooth_angle(prev_angle1,angle1,0.2)
        send_angle2 = smooth_angle(prev_angle2,angle2,0.2)
        ser.send_angles(send_angle1,send_angle2)
        # ser.send_angles(0,angle2)
        # ser.send_angles(angle1,0)
        # print(angle1,angle2)
        print(send_angle1,send_angle2)
        # time.sleep(0.5)


def main():
    find_thread = threading.Thread(target=find)
    find_thread.daemon = True
    find_thread.start()

    send_thread = threading.Thread(target=send_angle)
    send_thread.daemon = True
    send_thread.start()
   
   # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序被用户中断")

    
   
    
    
if __name__ == "__main__":
    main()
