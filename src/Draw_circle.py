from parma import Baseparma,CAPparma,Findparma
from find import Finder

import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_circle_allpoints(center, radius, num_points=100):
    """
    生成圆形上的点集

    参数:
        center: tuple (a, b)，圆心坐标
        radius: float，半径
        num_points: int，要生成的点数（默认100）

    返回:
        numpy.ndarray，形状为 (num_points, 2) 的点集
    """
    a, b = center
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = a + radius * np.cos(theta)
    y = b + radius * np.sin(theta)
    return np.stack((x, y), axis=-1)  # shape: (num_points, 2)

# theta = np.linspace(0, 2*np.pi, 100)
def generate_circle_points(center, radius, theta):
    a,b = center
    x = a + radius * np.cos(theta)
    y = b + radius * np.sin(theta)
    circle_points = (x,y)

    return circle_points

    
def get_radius(points, scale_x=0.23, scale_y=0.34):
    """
    计算点集中相邻点之间的特定坐标差值并返回平均值
    
    参数:
        points: 点集，格式为 [(x1, y1), (x2, y2), ...]，至少包含4个点
        scale_x: x方向的比例系数
        scale_y: y方向的比例系数
    
    返回:
        float: 各段差值的平均值，即:
               (|x2 - x1|*scale_x + |y3 - y2|*scale_y + 
                |x4 - x3|*scale_x + |y1 - y4|*scale_y) / 4
    """
    if len(points) < 4:
        raise ValueError("点集至少需要包含4个点")
    
    # 提取各点坐标
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]
    
    # 按规则计算各段差值并应用比例系数
    diff1 = abs(x2 - x1) * scale_x  # 点1与点2的x坐标差
    diff2 = abs(y3 - y2) * scale_y  # 点2与点3的y坐标差
    diff3 = abs(x4 - x3) * scale_x  # 点3与点4的x坐标差
    diff4 = abs(y1 - y4) * scale_y  # 点4与点1的y坐标差
    
    # 返回平均值（总和除以段数4）
    return (diff1 + diff2 + diff3 + diff4) / 4
    



if __name__ == "__main__":
    findner = Finder()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    i = 0
    thetas = np.linspace(0, 2*np.pi,100)
    while True:
        # 读取帧
        ret, frame = cap.read()  # ret: 是否成功, frame: 图像帧
        
        if not ret:
            print("无法读取帧！")
            break
        
        center,_,points = findner.find_frame_center(frame,Findparma.frame_lowerr,Findparma.frame_upperr)
        radius = get_radius(points)
        print(f"半径: {radius}")
        
        theta = thetas[i]
        if center != (0,0):
            i += 1
            circle_points = generate_circle_points(center, radius, theta)
            x, y = circle_points
            center = (int(x),int(y))
            # print(circle_points)
            cv2.circle(frame, center,5,(0, 0, 255),-1)  # 绘制圆心
            cv2.imshow("Frame", frame)
            print("已画点")
            if i >= len(thetas):
                i = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    
    