# -*- coding: utf-8 -*-
"""
###########################################################################################
#  @filename:  FEM_Main.py
#  @brief:     main function
#  @author:    Hao Chen
#  @version:   1.2
#  @update:    add dynamics and co-rotation
#  @date:      2026.03.25
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
import numpy as np
import os
import sys
import time
import multiprocessing as mp
import socket
import pickle
import traceback

# 外部引用文件
# import udpcomm
# from first_person_perspective.src import Project_moudle
# from udpcomm import UDPComm

# 内部fem文件
from fem_code.fem_run import fem
from fem_code.fem_opengl_draw import draw
from udp_connect.udp_recv_steer import udp_recv_steer
# from camera.DetectObject import DetectObject, getXYZlocation
# from camera.markcali import markcali
from fem_code.fem_show_exp import monitor_state
os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])


def udp_run(shared_udp):
    udp = udpcomm.pUDPComm()
    udp.run(shared_udp)


def udp_random(shared_udp, shared_draw):
    fr = False
    fx = False
    i = 0
    while (shared_draw['event'][0] != 1 or shared_draw['event'][1] != 1) and shared_draw['event'][0] != 5:
        if shared_draw['event'][0] == 1 and shared_draw['event'][1] == 32:
            fr = True
            fx = not fx
        if shared_draw['event'][0] == 1 and shared_draw['event'][1] == 2:
            fr = False
        if fr:
            i += 0.1
            time.sleep(0.1)
            if fx:
                x = 30 * np.cos(i)
                y = 30 * np.sin(i)
            else:
                y = 30 * np.cos(i)
                x = 30 * np.sin(i)

            shared_udp['datafloat'] = np.array([x, y, 0, 0, 1])

    shared_udp['datafloat'] = np.array([0, 0, 0, 0, 0])


def udp_recv_location(shared_location, shared_draw):
    recv_addr = ('192.168.31.48', 8080)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(recv_addr)
    s.settimeout(1)
    while (shared_draw['event'][0] != 1 or shared_draw['event'][1] != 1) and shared_draw['event'][2] != 5:
        try:
            pose__, addr = s.recvfrom(2048)
            pose_recv = pickle.loads(pose__)
            shared_location['location'] = np.array(pose_recv)  # .reshape((4, 4))
        except (Exception, BaseException) as errorstring:
            # print('%s' % str(errorstring))  # 打印错误
            pass
    s.close()


def fem1(trans_matrix, tip_matrix, mode, world_point, Is_warm_start):
    # mp.set_start_method('spawn')
    # cuda.init()
    # serialtest.SerialCom().start()
    velocity_InsertionUnit = 1  # 鞘和插入部插入速度,该值越大速度越慢，为除法,注意该值不能小于1/self.la
    velocity_sheath = 1  # 该值暂时无意义

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    shared_fps = manager.dict({})
    shared_camera_matrix = manager.dict({})
    shared_camera_matrix['camera_matrix'] = manager.list([0, 0, 0, 1, 1, 1, 0, 1, 0])
    shared_camera_matrix['trans_matrix'] = np.eye(4, dtype=np.float32)
    shared_draw = manager.dict({})
    event = np.array([0, 0, 0])
    shared_draw['event'] = event
    shared_draw['F'] = np.zeros(100 * 6, dtype=np.float32)
    shared_draw['constraint_triangle'] = np.zeros((1000, 1), dtype=np.float32)
    shared_draw['ResultNode'] = np.zeros((100, 3), dtype=np.float32)
    shared_draw['l1'] = 99
    shared_draw['na'] = 0
    shared_draw['la'] = 1
    shared_draw['velocity_InsertionUnit'] = velocity_InsertionUnit
    shared_draw['camera_matrix'] = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0], dtype=np.float32)
    shared_draw['camera_T_start'] = np.eye(3, dtype=np.float32)
    shared_draw['set_locate'] = 0
    shared_draw['fps'] = 0
    shared_draw['scaled_factor'] = 1.0
    manager1 = ctx.Manager()
    shared_mem = manager1.dict({})
    ss = manager1.list([0])
    # ab = np.zeros(1)
    shared_fps['fps'] = manager.list([0])
    shared_state = manager.dict({})
    shared_state['x'] = manager.list([0.])
    shared_state['v'] = manager.list([0.])
    shared_state['a'] = manager.list([0.])
    shared_state['dE'] = manager.list([0.])
    shared_state['t'] = manager.list([0.])
    # fem(shared_mem, shared_fps, None, velocity_InsertionUnit, velocity_sheath)
    # multiprocessing.set_start_method('spawn')
    shared_udp = manager.dict({})
    shared_udp['datafloat'] = np.array([0, 0, 0, 0, 0])

    shared_location = manager1.dict({})
    # 内参， [[fx,0,cx],[0,fy,cy],[0,0,1]]
    shared_location['intrinsic'] = np.array([[320.604884143610, 0, 323.645615703525],
                                            [0, 321.838038649933, 238.098053365179],
                                            [0, 0, 1]], dtype=np.float32)
    # 定位和欧拉转角，x,y,z,rx,ry,rz
    # shared_location['location'] = np.array([[-0.999059796,-0.042072266,0.010458697,-0.916783988],[0.003219745,0.168573037,0.985683858,9.344343185],[-0.043233007,0.984790862,-0.168279082,-1.924159646],[0,0,0,1]], dtype=np.float32)
    shared_location['location'] = 10 * np.array([-0.916783988, 9.344343185, -1.924159646, -1.739889264, 0.010458888, 3.099505663]) # 单位改成mm

    shared_RL = manager.dict({})
    shared_RL['state'] = np.zeros((100, 3), dtype=np.float32)
    shared_RL['F'] = np.zeros(100 * 6, dtype=np.float32)
    shared_RL['U'] = manager.list([0])
    shared_RL['draw'] = manager.list([1])
    shared_RL['action'] = None
    process = [ctx.Process(target=fem, args=(
                shared_mem, shared_fps, velocity_InsertionUnit, velocity_sheath, shared_camera_matrix, trans_matrix,
                tip_matrix, shared_draw, shared_udp, mode, shared_location, shared_RL, Is_warm_start, shared_state)),  # 有限元主函数
               ctx.Process(target=draw, args=(trans_matrix, shared_draw, mode, shared_location, shared_RL, world_point)),  # 绘制主函数
               # ctx.Process(target=Project_moudle.AR_show, args=(shared_camera_matrix,)),  # 输出到AR
               # ctx.Process(target=udp_recv_location, args=(shared_location, shared_draw,)),  # 读取定位
               # ctx.Process(target=udp_run, args=(shared_udp,)),  # 改用新的和主手的通信函数udp_recv_steers
               # ctx.Process(target=udp_recv_steer, args=(shared_udp,)),  # 主手通信
               # ctx.Process(target=your_RL_code, args=(shared_draw, shared_RL)),  # RL
               ctx.Process(target=monitor_state, args=(shared_state,)),  # monitor state
               ]
    # Project_moudle.AR_show(shared_camera_matrix)
    [p.start() for p in process]
    [p.join() for p in process]


def your_RL_code(shared_draw, shared_RL):
    pass
    try:
        pass
        while not shared_draw['event'][2] == 5:
            print(shared_RL['state'])
            print(shared_RL['F'])
            print(shared_RL['U'])  # potential energy, 有wen ti
            shared_RL['action'] = 'RIGHT'
            # shared_RL['draw'] = 1
    except (Exception, BaseException) as errorstring:
        print('RL退出，原因是：（systemExit为代码正常退出）%s' % str(errorstring))
        print(traceback.format_exc())
    finally:
        event = np.array([0, 0, 5])
        shared_draw['event'] = event


def get_exp_data():
    intr_matrix = np.array([[557.9329, 0, 244.1769], [0, 557.5108, 243.7359], [0, 0, 1]])
    intr_coeffs = np.array([0.1425, -0.0519, 0.0074, -0.0325, -0.1166])
    img_point, coeff = DetectObject()  # img_point
    print(img_point)
    pose_R, pose_t = markcali(intr_matrix, intr_coeffs)
    world_point = getXYZlocation(pose_R, pose_t, np.array(img_point, dtype=np.float32))
    world_point *= 1000
    print(world_point)
    np.save('./data/exp_data/right/world_point.npy', world_point)
    np.save('./data/exp_data/right/pose_R.npy', pose_R)
    np.save('./data/exp_data/right/pose_t.npy', pose_t)
    return world_point, pose_R, pose_t


def reload_exp_data():
    world_point = np.load('./data/exp_data/right/world_point.npy')
    pose_R = np.load('./data/exp_data/right/pose_R.npy')
    pose_t = np.load('./data/exp_data/right/pose_t.npy')
    return world_point, pose_R, pose_t


if __name__ == '__main__':
    # world_point, pose_R, pose_t = get_exp_data()
    world_point, pose_R, pose_t = reload_exp_data()
    # XYZ_point = world_point @ pose_R - 1000 * pose_t[:, 0]
    # print(XYZ_point)
    trans_matrix = np.eye(4, dtype=np.float32)
    tip_matrix = np.eye(4, dtype=np.float32)
    tip_matrix[0:3, 0:3] = pose_R
    tip_matrix[0:3, 3] = pose_t[0:3, 0]
    mode = 1
    tip_matrix = None
    path_root = os.getcwd()
    Is_warm_start = False
    if len(sys.argv) > 1:
        Is_warm_start = sys.argv[1]
    # trans_matrix = np.load(path_root + "/data/npy/trans_matrix.npy")
    # tip_matrix = np.load(path_root + "/data/npy/tip2cam.npy")

    fem1(trans_matrix, tip_matrix, mode, world_point, Is_warm_start)



