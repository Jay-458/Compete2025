
from value import parse_date
from parma import Baseparma,CPPparma


import cv2
import serial

# 打开相机（通常 0 表示内置摄像头，1 表示外接摄像头,初始化串口
cap = cv2.VideoCapture(0)
ser = serial.Serial()

while True:
    if ser.in_waiting > 10 :
        cap_value = parse_date(ser.readline(ser1.in_waiting).decode('utf-8',errors='replace').strip())
        try:

            CPPparma.fps = cap_value["fps"]
            CPPparma.brightness = cap_value["brightness"]
            CPPparma.contrast = cap_value["contrast"]
            CPPparma.saturation = cap_value["saturation"]
            CPPparma.hue = cap_value["hue"]
            CPPparma.gain = cap_value["gain"]
            CPPparma.exposure = cap_value["exposure"]
            CPPparma.auto_exposure = cap_value["auto_exposure"]

            # 设置参数（参数ID, 参数值）
            cap.set(cv2.CAP_PROP_FPS, CPPparma.fps)               # 帧率
            cap.set(cv2.CAP_PROP_BRIGHTNESS,CPPparma.brightness)       # 亮度 (0-1)
            cap.set(cv2.CAP_PROP_CONTRAST, CPPparma.contrast)         # 对比度 (0-1)
            cap.set(cv2.CAP_PROP_SATURATION, CPPparma.saturation)       # 饱和度 (0-1)
            cap.set(cv2.CAP_PROP_HUE, CPPparma.hue)              # 色调 (0-1)
            cap.set(cv2.CAP_PROP_GAIN, CPPparma.gain)             # 增益 (0-1)
            cap.set(cv2.CAP_PROP_EXPOSURE, CPPparma.exposure)          # 曝光 (-10-10)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CPPparma.auto_exposure)   # 自动曝光 (0=关闭, 1=开启, 0.25=手动模式)
        except:
            continue

    # 读取并显示帧
    ret, frame = cap.read()
    cv2.imshow('Camera', frame)

# 释放资源
cap.release()
cv2.destroyAllWindows()