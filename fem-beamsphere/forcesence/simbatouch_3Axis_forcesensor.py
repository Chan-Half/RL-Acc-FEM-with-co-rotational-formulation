import minimalmodbus
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
# Define instrument object for Modbus communication
instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1, mode='rtu')
instrument.serial.baudrate = 9600
instrument.serial.bytesize = 8
while_cont = True
is_zero_get = False
is_stop = True
origin_data = np.zeros(3)
force = np.zeros(3)
force_zero = np.zeros(3)
max_force = np.zeros(3)
# Read register using Modbus RTU
# set_zero
pygame.init()
viewport = (400, 400)
screen = pygame.display.set_mode(viewport, pygame.OPENGL | pygame.DOUBLEBUF)
pygame.key.set_repeat(1, 10)
print('please set zero')
while while_cont:
    if is_stop:
        register_value = np.array(instrument.read_registers(450, 6))
        for i in range(3):
            if register_value[i*2] > 65536/2:
                register_value[i*2+1] -= 65535
            elif register_value[i*2] == 0:
                pass
            else:
                print('the force is too large!!!!')
                print(register_value)
                while_cont = False
        origin_data = register_value[1:6:2]
        force = origin_data - force_zero
        if is_zero_get and np.linalg.norm(max_force) < np.linalg.norm(force):
            max_force = force
            print("max_force:", max_force)
        if is_zero_get:
            print("force:", force)
        else:
            print("origin data:", origin_data)
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            force_zero = origin_data
            is_zero_get = True
            print('get force zero:', force_zero)
        if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
            print(np.linalg.norm(max_force) * 9.8e-3)
            max_force = np.zeros(3)
            is_stop = False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            is_stop = True
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            print(np.linalg.norm(max_force)*9.8e-3)
            while_cont = False


# while_cont = True
# while while_cont:
#     register_value = np.array(instrument.read_registers(450, 6))
#     for i in range(3):
#         if register_value[i*2] > 65536/2:
#             register_value[i*2+1] -= 65535
#         elif register_value[i*2] == 0:
#             pass
#         else:
#             print('the force is too large!!!!')
#             print(register_value)
#             while_cont = False
#     if np.linalg.norm(force) < np.linalg.norm(register_value[1:6:2] - force_zero):
#         force = register_value[1:6:2] - force_zero
#     print(force)
#     for e in pygame.event.get():
#         if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
#             print(np.linalg.norm(force)*9.8e-3)
#             while_cont = False

pygame.quit()
