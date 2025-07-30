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
def generate_circle_points(theta):
    center = (14, 9)
    a,b = center
    radius = get_radius(points = np.float32([[0, 0], [27, 0], [27, 18], [0, 18]]), scale_y=0.34)
    x = a + radius * np.cos(theta)
    y = b + radius * np.sin(theta)
    circle_points = (x,y)

    return circle_points

    
def get_radius(points = np.float32([[0, 0], [27, 0], [27, 18], [0, 18]]), scale_y=0.34):
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
    
    diff2 = abs(y3 - y2) * scale_y  # 点2与点3的y坐标差
    diff4 = abs(y1 - y4) * scale_y  # 点4与点1的y坐标差
    
    # 返回平均值（总和除以段数4）
    return ( diff2 +  diff4) / 2
    

def transform_and_inverse(src_points,theta):
    """
    对输入的4个点进行透视变换，计算目标点后逆变换回原始坐标系
    （目标点集在函数内部定义为固定矩形）
    
    参数:
        src_points: 原始坐标系的4个点，格式 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        calc_func: 计算函数，输入目标坐标系的点集，返回目标点 (x, y)
        *calc_args: 传递给 calc_func 的额外参数
        dst_width: 目标矩形的宽度（默认400）
        dst_height: 目标矩形的高度（默认400）
    
    返回:
        original_target: 逆变换后的原始坐标系目标点 (x, y)
        transformed_target: 目标坐标系中的目标点 (x, y)
    """
    # 1. 在函数内部定义固定的目标点集（矩形）
    dst_points = np.float32([[0, 0], [27, 0], [27, 18], [0, 18]])
    
    # 2. 转换原始点为numpy数组
    src = np.float32(src_points)
    
    # 3. 计算透视变换矩阵（src -> dst）
    M = cv2.getPerspectiveTransform(src, dst_points)
    
    # 4. 计算逆变换矩阵（dst -> src）
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        raise ValueError("变换矩阵不可逆，请检查输入点是否共线")
    
    # 5. 在目标坐标系中计算目标点
    transformed_target = generate_circle_points(theta)
    
    # 6. 逆变换回原始坐标系
    x, y = transformed_target
    # 透视变换需用齐次坐标 (x, y, 1)
    homogeneous = np.array([x, y, 1], dtype=np.float32)
    original_homogeneous = M_inv @ homogeneous  # 矩阵乘法
    # 除以齐次分量w，得到原始坐标
    original_x = original_homogeneous[0] / original_homogeneous[2]
    original_y = original_homogeneous[1] / original_homogeneous[2]
    
    return (original_x, original_y)
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
        
        
        theta = thetas[i]
        if center != (0,0):
            i += 1
            circle_points = transform_and_inverse(points,theta)
            x, y = circle_points
            center = (int(x),int(y))
            print(center)
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

    
    