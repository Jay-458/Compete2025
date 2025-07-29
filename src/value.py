from parma import Baseparma



import serial


def parse_date(date_str):
    #初始化字典，通过“，”分割信息
    cap_result = {}
    paris = date_str.split(';')
    # print(paris)

    for pari in paris:
        if '=' in pari:
            
            key,value = pari.split("=",1)

            key = key.strip()
            value = value.strip()

            cap_result[key] = value

    return cap_result


if __name__ == "__main__":

    ser1 = serial.Serial(port = Baseparma.COM,baudrate = Baseparma.Baud_rate)
    print(f"串口已成功创建")
    while True:
        if ser1.in_waiting > 0:
            str_date = ser1.readline(ser1.in_waiting).decode('utf-8',errors='replace').strip()
            # print(f"已收到信息")

            value_date = parse_date(str_date)
            print(value_date)

