import numpy as np

class Baseparma :
    COM = "COM11"
    Baud_rate = 9600
    ID = [ "$1;", "$2;", "$3;", "$4;", "$5;", "$6;", "$7;", "$8;", "$9;", "$10;", "$11;"]
   
    

class CAPparma :
    fps        = 30
    brightness = 0.5
    contrast   = 0.5
    saturation = 0.5
    hue        = 0.5
    gain       = 0.5 
    exposure   = -3
    auto_exposure = 0.25
    image_size = (640, 480)
    h_fov      = 100.0
    v_fov      = 100.0

class Findparma:
    frame_width  = 640
    frame_height = 480
    frame_lowerr = np.array([0, 0, 0])
    frame_upperr = np.array([180, 137, 97])
    threshold_value = 140
    
