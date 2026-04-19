from socket import *
import struct

# import rospy
# from tsms_control_rc.msg._ftSensed import ftSensed

import numpy as np

Ex=[5.012264,5.012264,5.012264,5.012264,5.012264,5.012264]
AmpZero=[32846.000000,32711.000000,32838.000000,32862.000000,32757.000000,32837.000000]
Gain=[123.767562,123.713668,123.653037,123.737246,123.750720,123.629458]
Decoupled_r0 = [0.599355,0.857297,0.008628,123.554476,1.472792,-124.840125]
Decoupled_r1 = [3.072252,-142.374254,-0.660497,72.106101,1.273655,70.840278]
Decoupled_r2 = [414.722036,-4.445533,413.279384,-3.364040,411.545367,-2.807879]
Decoupled_r3 = [-0.115987,-0.044253,-3.292908,0.050730,2.963563,-0.009766]
Decoupled_r4 = [3.776915,0.038207,-2.005137,0.140784,-1.757471,0.008970]
Decoupled_r5 = [-0.028952,1.252525,-0.001470,1.092244,-0.007079,1.020931]
Decoupled=np.mat([Decoupled_r0,Decoupled_r1,Decoupled_r2,Decoupled_r3,Decoupled_r4,Decoupled_r5])
# Decoupled=[[0.599355,0.857297,0.008628,123,554476,1.472792,-124.840125],[3.072252,-142.374254,-0.660497,72.106101,1.273655,70.840278],[414.722036,-4.445533,413.279384,-3.364040,411.545367,-2.807879],[-0.115987,-0.044253,-3.292908,0.050730,2.963563,-0.009766],[3.776915,0.038207,-2.005137,0.140784,-1.757471,0.008970],[-0.028952,1.252525,-0.001470,1.092244,-0.007079,1.020931]]
# print(Decoupled)
# HOST = '192.168.0.108' #default Address
HOST = '192.168.1.101'
PORT = 4008
BUFSIZ = 38
ADDR = (HOST, PORT)

tcpClientSock = socket(AF_INET, SOCK_STREAM)
tcpClientSock.connect(ADDR)

data_ft=[0, 0, 0, 0, 0, 0]

def ft_pub():
    pub = rospy.Publisher('SRI_force_topic',ftSensed,queue_size=8)
    rospy.init_node('SRI_node',anonymous=True)
    rate = rospy.Rate(500)

    while not rospy.is_shutdown():

        tcpClientSock.send(b'AT+GOD\r\n')
        data_raw = tcpClientSock.recv(BUFSIZ)

        if not data_raw:
            break

        # print(struct.unpack('>HHHHHH',data_raw[6:18]))
        AD_Count = struct.unpack('>HHHHHH',data_raw[6:18])
        #
        # data_ft = [0,0,0,0,0,0]

        data_ft[0] = 1000 * (AD_Count[0] - AmpZero[0]) / 65535 * 5 / Gain[0] / Ex[0]
        data_ft[1] = 1000 * (AD_Count[1] - AmpZero[1]) / 65535 * 5 / Gain[1] / Ex[1]
        data_ft[2] = 1000 * (AD_Count[2] - AmpZero[2]) / 65535 * 5 / Gain[2] / Ex[2]
        data_ft[3] = 1000 * (AD_Count[3] - AmpZero[3]) / 65535 * 5 / Gain[3] / Ex[3]
        data_ft[4] = 1000 * (AD_Count[4] - AmpZero[4]) / 65535 * 5 / Gain[4] / Ex[4]
        data_ft[5] = 1000 * (AD_Count[5] - AmpZero[5]) / 65535 * 5 / Gain[5] / Ex[5]
        DATA_FT = np.mat([[data_ft[0]],[data_ft[1]],[data_ft[2]],[data_ft[3]],[data_ft[4]],[data_ft[5]]])
        FT = Decoupled*DATA_FT
        fs = ftSensed()
        fs.f_x_tool = float(FT[0])
        fs.f_y_tool = float(FT[1])
        fs.f_z_tool = float(FT[2])
        fs.t_x_tool = float(FT[3])
        fs.t_y_tool = float(FT[4])
        fs.t_z_tool = float(FT[5])
        pub.publish(fs)
        print(FT)
        # print(np.transpose(FT))
        # print(ftSensed.t_z_tool)
        rate.sleep()

    tcpClientSock.close()

if __name__=='__main__':
    try:
        ft_pub()
    except rospy.ROSInterruptException:
        pass

