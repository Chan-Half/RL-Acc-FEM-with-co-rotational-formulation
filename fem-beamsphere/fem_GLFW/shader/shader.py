"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:  shader.py
#  @brief:     fem shader
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2023.04.03
###########################################################################################
"""
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import numpy as np
import scipy.linalg as linalg


class Shader:
    def __init__(self, vertex_path, fragment_path):
        with open(vertex_path, mode='r', encoding='utf-8') as vertex_stream:
            vertex_code = vertex_stream.readlines()
        with open(fragment_path, mode='r', encoding='utf-8') as fragment_stream:
            fragment_code = fragment_stream.readlines()

        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_code)
        glCompileShader(vertex_shader)
        status = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
        if not status:
            print("[ERROR]: " + bytes.decode(glGetShaderInfoLog(vertex_shader)))

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_code)
        glCompileShader(fragment_shader)
        status = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
        if not status:
            print("[ERROR]: " + bytes.decode(glGetShaderInfoLog(fragment_shader)))

        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)
        status = glGetProgramiv(shader_program, GL_LINK_STATUS)
        if not status:
            print("[ERROR]: " + bytes.decode(glGetProgramInfoLog(shader_program)))

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        self.shaderProgram = shader_program

    def use(self):
        glUseProgram(self.shaderProgram)

    def delete(self):
        glDeleteProgram(self.shaderProgram)


class camera1:
    def __init__(self, view):
        self.view = view
        self.position = np.transpose(self.view[0:3, 3])  # np.array([0.0, 0.0, 0.0])
        self.T = self.view[0:3, 0:3]  # np.eye(3)
        # self.front = np.array([1.0, 0.0, 0.0])
        # self.right = np.array([0.0, 1.0, 0.0])
        # self.up = np.array([0.0, 0.0, 1.0])

    # 旋转矩阵 欧拉角
    def rotate(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        self.T = np.dot(self.T, rot_matrix)
        self.view[0:3, 0:3] = self.T

    def translate(self, transposition):
        self.position += transposition
        self.view[0:3, 3] = np.transpose(self.position)

    def getview(self):
        return self.view


# class camera:
#     origin = [0.0, 0.0, 0.0]
#     length = 1.
#     yangle = 0.
#     zangle = 0.
#     __bthree = False
#
#     def __init__(self):
#         self.mouselocation = [0.0, 0.0]
#         self.offest = 0.01
#         self.zangle = 0. if not self.__bthree else math.pi
#
#     def setthree(self, value):
#         self.__bthree = value
#         self.zangle = self.zangle + math.pi
#         self.yangle = -self.yangle
#
#     def eye(self):
#         return self.origin if not self.__bthree else self.direction()
#
#     def target(self):
#         return self.origin if self.__bthree else self.direction()
#
#     def direction(self):
#         if self.zangle > math.pi * 2.0:
#             self.zangle -= math.pi * 2.0
#         elif self.zangle < 0.:
#             self.zangle += math.pi * 2.0
#         len = 1. if not self.__bthree else self.length if 0. else 1.
#         xy = math.cos(self.yangle) * len
#         x = self.origin[0] + xy * math.sin(self.zangle)
#         y = self.origin[1] + len * math.sin(self.yangle)
#         z = self.origin[2] + xy * math.cos(self.zangle)
#         return [x, y, z]
#
#     def move(self, x, y, z):
#         sinz, cosz = math.sin(self.zangle), math.cos(self.zangle)
#         xstep, zstep = x * cosz + z * sinz, z * cosz - x * sinz
#         if self.__bthree:
#             xstep = -xstep
#             zstep = -zstep
#         self.origin = [self.origin[0] + xstep, self.origin[1] + y, self.origin[2] + zstep]
#
#     def rotate(self, z, y):
#         self.zangle, self.yangle = self.zangle - z, self.yangle + y if not self.__bthree else -y
#
#     def setLookat(self):
#         ve, vt = self.eye(), self.target()
#         # print ve,vt
#         glLoadIdentity()
#         gluLookAt(ve[0], ve[1], ve[2], vt[0], vt[1], vt[2], 0.0, 1.0, 0.0)
#
#     def keypress(self, key, x, y):
#         if key in ('e', 'E'):
#             self.move(0., 0., 1 * self.offest)
#         if key in ('f', 'F'):
#             self.move(1 * self.offest, 0., 0.)
#         if key in ('s', 'S'):
#             self.move(-1 * self.offest, 0., 0.)
#         if key in ('d', 'D'):
#             self.move(0., 0., -1 * self.offest)
#         if key in ('w', 'W'):
#             self.move(0., 1 * self.offest, 0.)
#         if key in ('r', 'R'):
#             self.move(0., -1 * self.offest, 0.)
#         if key in ('v', 'V'):
#             # this.__bthree = not this.__bthree
#             self.setthree(not self.__bthree)
#         if key == glfw.KEY_UP:
#             self.offest = self.offest + 0.1
#         if key == glfw.KEY_DOWN:
#             self.offest = self.offest - 0.1
#
#     def mouse(self, x, y):
#         rx = (x - self.mouselocation[0]) * self.offest * 0.1
#         ry = (y - self.mouselocation[1]) * self.offest * -0.1
#         self.rotate(rx, ry)
#         self.mouselocation = [x, y]
