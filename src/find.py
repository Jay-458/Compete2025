from value import parse_date
from parma import Baseparma,CAPparma,Findparma

import cv2
import serial
import numpy as np
class Finder:
    def __init__(self):
        self.width = Findparma.frame_width
        self.height = Findparma.frame_height
        

    def base_process(self,frame):
        resized_image = cv2.resize(frame, (self.width, self.height))
        #BGR转HSV色彩空间
        # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 转换为灰度图像
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # 图像二值化
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        #腐蚀
        #kernel = np.ones((5,5),np.uint8)
        #erosion = cv.erode(img,kernel,iterations = 1)
        #膨胀
        #dilation = cv.dilate(img,kernel,iterations = 1)
        #开运算，腐蚀后再膨胀
        # opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        #闭运算，先膨胀后腐蚀
        # closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        #形态学梯度，结果将看起来像对象的轮廓
        # gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
        # 在原图上绘制轮廓
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        base_frame = binary_image
        return base_frame

#利用hsv色彩空间找单一颜色
    def find_color(self,frame,lower_color,upper_color):
        frame = cv2.resize(frame, (self.width, self.height))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        kernel = np.ones((5,5),np.uint8)
        sigma = 5.0
        #sigma越小，对中心像素的权重越大，去噪能力越强。去除轻微噪声：使用较小的 σ（如 0.8-1.5），去除严重噪声：增大 σ（如 2-5），但需权衡模糊程度。
        kernel_size = 5
        # 3. 中值滤波
        # median_filtered = cv2.medianBlur(mask, kernel_size)
        
        # 2. 高斯滤波
        
        gaussian_filtered = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        # 1. 均值滤波
       
        # mean_filtered = cv2.blur(gaussian_filtered, (kernel_size, kernel_size))
        
        filtered_mask = gaussian_filtered

       

        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours,filtered_mask

    # def find_color_with_RGB(self,frame,):
    #     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #     bgr = frame

    #     # 拆分通道
    #     h, s, v = cv2.split(hsv)  # HSV的H、S、V通道
    #     b, g, r = cv2.split(bgr)  # BGR的B、G、R通道（对应RGB的R、G、B反转）

    #     # 1. HSV的H通道：紫色的H值约125-150（根据实际调整）
    #     _, mask_h = cv2.threshold(h, 125, 150, cv2.THRESH_BINARY)  # 仅保留125-150的区域

    #     # 2. B通道（紫色含高蓝分量）：排除低蓝值区域
    #     _, mask_b = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY)  # B值>100的区域

    #     # 3. G通道（紫色含低绿分量）：排除高绿值区域
    #     _, mask_g = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY_INV)  # G值<50的区域

    #     mask_combined = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_b, mask_g))

    #     contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     #高斯滤波

    #     #中值滤波

    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def Judge_the_shape(self,frame,lower_color,upper_color):
        frame = cv2.resize(frame, (self.width, self.height))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        kernel = np.ones((5,5),np.uint8)
        sigma = 5.0
        #sigma越小，对中心像素的权重越大，去噪能力越强。去除轻微噪声：使用较小的 σ（如 0.8-1.5），去除严重噪声：增大 σ（如 2-5），但需权衡模糊程度。
        kernel_size = 5
        # 3. 中值滤波
        # median_filtered = cv2.medianBlur(mask, kernel_size)
        
        # 2. 高斯滤波
        
        gaussian_filtered = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        filtered_mask = gaussian_filtered
        # 3. 查找轮廓（使用RETR_CCOMP以获取层次结构）
        contours, hierarchy = cv2.findContours(filtered_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # 4. 区分内外轮廓
        outer_contours = []
        inner_contours = []
        for i, contour in enumerate(contours):
            # hierarchy[i][3]表示父轮廓的索引，-1表示无父轮廓（外轮廓）
            if hierarchy[0][i][3] == -1: 
                outer_contours.append(contour)
            else:
                inner_contours.append(contour)

        # # 5. 绘制轮廓
        # frame = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像
        # # 绘制外轮廓（绿色）
        # for contour in outer_contours:
        #     cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        #     print(f"外轮廓面积: {cv2.contourArea(contour)}")
        # 绘制内轮廓（红色）
        # for contour in inner_contours:
        #     cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
        #     print(f"内轮廓面积: {cv2.contourArea(contour)}")

        


       
        for contour in inner_contours:
            area = cv2.contourArea(contour, oriented=False)
            epsilon = 0.02 * cv2.arcLength(contour, True)  # 阈值参数，可调整
            approx = cv2.approxPolyDP(contour, epsilon, True)  # 多边形逼近

            num_sides = len(approx)  # 边的数量
            if (num_sides == 4 and area>100) or  (num_sides == 3 and area>100) or (num_sides > 20 and area>100):
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                print(f"内轮廓面积: {cv2.contourArea(contour)}")


         
# 6. 显示结果
        cv2.imshow("Original Mask",filtered_mask)
        cv2.imshow("Contours (Green: Outer, Red: Inner)", frame)

   



if __name__ == "__main__":
    finder = Finder()
    cap = cv2.VideoCapture(0)
    lower_color = np.array([0,0,0])
    upper_color = np.array([180,180,80])
    min_area = 100
    while True:
        # 读取帧
        ret, frame = cap.read()  # ret: 是否成功, frame: 图像帧
        
        if not ret:
            print("无法读取帧！")
            break
        finder.Judge_the_shape(frame,lower_color,upper_color)
        # contours,color_mask = finder.find_color(frame,lower_color,upper_color)

        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area > min_area:
        #         # 计算轮廓的矩
        #         M = cv2.moments(contour)
                
        #         # 计算中心坐标
        #         if M["m00"] != 0:  # 避免除以零（处理极小或空轮廓）
        #             cx = int(M["m10"] / M["m00"])
        #             cy = int(M["m01"] / M["m00"])
                    
        #             # 在图像上绘制中心点
        #             cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # 绿色实心圆
        #             cv2.putText(frame, f"({cx}, {cy})", (cx+10, cy), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # # base_frame = finder.base_process(frame)
        # # 显示帧
        # cv2.imshow('Camera', frame)
        # cv2.imshow('mask',color_mask)
        
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
