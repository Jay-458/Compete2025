import cv2
import numpy as np
def empty(x):
    print(x)


#创建一个命名窗口
cv2.namedWindow('trackbar',cv2.WINDOW_AUTOSIZE)
#创建色调滑动条，控制最小色调值
cv2.createTrackbar('hue-min','trackbar',0,180,empty)
#创建色调滑动条，控制最大色调值
cv2.createTrackbar('hue-max','trackbar',180,180,empty)
#创建饱和度滑动条，控制最小值
cv2.createTrackbar('sat-min','trackbar',0,255,empty)
#创建饱和度滑动条，控制最大值
cv2.createTrackbar('sat-max','trackbar',255,255,empty)
#创建亮度滑动条，控制最小亮度
cv2.createTrackbar('val-min','trackbar',0,255,empty)
#创建亮度滑动条，控制最大亮度\
cv2.createTrackbar('val-max','trackbar',255,255,empty)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
while True:
    # 读取一帧视频
    ret, frame = cap.read()
    cv2.resize(frame, (720, 480))
        # 检查是否成功读取帧
    if not ret:
        print("无法获取帧，退出...")
        break
    try:
        h_min = cv2.getTrackbarPos('hue-min','trackbar')
        h_max = cv2.getTrackbarPos('hue-max','trackbar')
        s_min = cv2.getTrackbarPos('sat-min','trackbar')
        s_max = cv2.getTrackbarPos('sat-max','trackbar')
        val_min = cv2.getTrackbarPos('val-min','trackbar')
        val_max = cv2.getTrackbarPos('val-max','trackbar')
        
        lower = np.array([h_min,s_min,val_min])
        upper = np.array([h_max,s_max,val_max])
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,lower,upper)

        result = cv2.bitwise_and(frame,frame,mask=mask)
        cv2.imshow('mask',mask)
        cv2.imshow('trackbar',result)


    except Exception as e:
                # 捕获异常并打印错误信息
                print(f"发生错误: {e}")
                # 继续循环
                continue
        # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()