import time
import serial
import struct
import ctypes
import threading
import numpy as np

lock = threading.Lock()

force = [0, 0, 0]
torque = [0, 0, 0]


# forcearray = []
# torquearray = []

class Sensor(threading.Thread):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                This CLASS is used for keep listening the FORCE INFORMATION from force sensor                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __init__(self):
        threading.Thread.__init__(self)
        # sudo chmod 777 /dev/ttyUSB0   # 赋予权限
        # sudo cutecom   # 打开软件  需要输入AT+GSD 获取实时数据
        # dmesg|grep ttyS*  # 查看开机以来哪个端口插拔过
        port = '/dev/ttyUSB0'  # port number
        bps = 115200  # bundrate
        timeO = 5  # time out
        self.ser = serial.Serial(port, bps, timeout=timeO)  # open port

        self.package_flag = False  # header flag
        self.rev = ''  # data package

    def run(self):
        global force
        global torque
        # global forcearray
        # global torquearray

        while self.ser.isOpen():
            data = self.ser.read().hex()  # recive data to hex
            if self.package_flag:
                self.rev = self.rev + data
                if len(self.rev) == 62:
                    sri_data = np.zeros(6)
                    check = self.uchar_checksum(bytes.fromhex(self.rev[12:-2]),
                                                24)  # group number*2*CHANNUM+11 (length untill check bit)
                    if check[-2:] == self.rev[-2:]:
                        package = self.rev[12:-2]
                        for num in range(6):
                            channel = '0x' + package[8 * num + 6:8 * num + 8] + package[
                                                                                8 * num + 4:8 * num + 6] + package[
                                                                                                           8 * num + 2:8 * num + 4] + package[
                                                                                                                                      8 * num:8 * num + 2]
                            sri_value = self.hex2float(channel)
                            sri_data[num] = sri_value

                    if not np.all(sri_data == 0):
                        force = sri_data[:3]
                        torque = sri_data[3:]
                        # forcearray.append(force)
                        # torquearray.append(torque)
                        # if len(forcearray)>3  or len(torquearray)>3:
                        #    del(forcearray[0])
                        #    del(torquearray[0])

                    self.ser.flushInput()
                    self.rev = ''
                    self.package_flag = False
                    # print('hi')
                    # time.sleep(0.00001)

            if data == 'aa' and not self.package_flag:  # detect header
                if self.ser.read().hex() == '55':
                    self.package_flag = True
                    self.rev = 'aa55'

            # print(time.perf_counter() - start)
            # lock.release()
        self.ser.close()

    def uchar_checksum(self, data, length, byteorder='little'):
        checksum = 0
        for i in range(0, length):
            checksum += int.from_bytes(data[i:i + 1], byteorder, signed=False)
            checksum &= 0xFF  # leave 16 bits

        return hex(checksum)

    def hex_to_float(self, h):
        i = int(h, 16)
        return struct.unpack('<f', struct.pack('<I', i))[0]

    def hex2float(self, h):
        i = int(h, 16)
        cp = ctypes.pointer(ctypes.c_int(i))
        fp = ctypes.cast(cp, ctypes.POINTER(ctypes.c_float))
        return fp.contents.value


def GetForce():
    global force
    # num = forcearray
    # totalforce = 0
    # for i in range(len(num)):
    #    totalforce += num[i]
    # cur_force = totalforce/len(num)
    return force


def GetTorque():
    global torque
    # num = torquearray
    # totaltorque = 0
    # for i in range(len(num)):
    #    totaltorque += num[i]
    # cur_torque = totaltorque/len(num)
    return torque


if 1:
    Sensor().start()
    while True:
        print('force:', force)
        print('torque:', torque, '\n')
