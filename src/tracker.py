from parma import CAPparma


import math
import numpy as np




import math

import numpy as np

import numpy as np

def calculate_relative_angles(pixel_point, image_size, fov, ref_point):
    """
    计算从参考点指向图像中某个像素的相对俯仰角和偏航角
    
    参数:
    pixel_point: 图像中的像素坐标 (u, v)
    image_size: 图像尺寸 (width, height)
    fov: 水平视场角（单位：度）
    ref_point: 相机坐标系下的参考点位置 (X_ref, Y_ref, Z_ref)
    
    返回:
    relative_pitch: 相对于参考点的俯仰角（度）
    relative_yaw: 相对于参考点的偏航角（度）
    """
    u, v = pixel_point
    width, height = image_size
    cx, cy = width / 2, height / 2

    # 归一化图像坐标（范围：[-1, 1]）
    norm_x = (u - cx) / (width / 2)
    norm_y = (v - cy) / (height / 2)

    # 估算垂直视角
    fov_h = fov
    fov_v = fov * height / width

    # 方向向量在相机坐标系下
    x_dir = np.tan(np.radians(fov_h / 2)) * norm_x
    y_dir = np.tan(np.radians(fov_v / 2)) * norm_y
    z_dir = 1.0

    # 构造方向向量
    direction = np.array([x_dir, y_dir, z_dir])
    ref = np.array(ref_point)
    delta = direction - ref

    # 偏航角（绕Y轴）
    relative_yaw = np.degrees(np.arctan2(delta[0], delta[2]))
    # 俯仰角（绕X轴）
    relative_pitch = np.degrees(np.arctan2(delta[1], np.sqrt(delta[0]**2 + delta[2]**2)))

    return relative_pitch, relative_yaw


# 使用示例
if __name__ == "__main__":
    image_size = (640, 480)
    fov = 100.0
    pixel_point = (320, 240)
    ref_point = (0, 0, 0)
    
    pitch, yaw = calculate_relative_angles(pixel_point, image_size, fov, ref_point)
    print(f"像素坐标: ({pixel_point[0]}, {pixel_point[1]})")
    print(f"参考点坐标: ({ref_point[0]:.2f}, {ref_point[1]:.2f}, {ref_point[2]:.2f}) m")
    print(f"相对角度: 俯仰角 = {pitch:.2f}°, 偏转角 = {yaw:.2f}°")    