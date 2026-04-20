"""
###########################################################################################
#  @copyright: Institute of Automation,Chinese Academy of Sciences
#  @filename:  fem_opengl_draw.py
#  @brief:     fem draw function
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2026.03.01
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""

import random
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import os
import sys
import time
import trimesh
# import skeletor
import scipy.linalg as linalg
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *

from fem_code.fem_Exclude_Function import fa_vector
from fem_code.RotationEularQuaternion import rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix, rotateToQuaternion, QuaternionToAxialAngle
os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])


class param(object):
    pass


def set_light0():
    glEnable(GL_LIGHT0)
    light_position = [1.0, 1.0, 1.0, 0.0]
    light_ambient = [0.1, 0.1, 0.1, 1.0]
    light_diffuse = [1.0, 1.0, 1.0, 1.0]
    light_specular = [1.0, 1.0, 1.0, 1.0]

    # glLightfv(GL_LIGHT0, GL_POSITION, light_position)  # 指定位置
    # glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)  # 设置环境光
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)  # 设置漫反射光
    # glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)  # 设置镜面反射光

    mat_specular = [0.1, 0.1, 0.1, 1.0]
    mat_ambient = [0.5, 0.5, 0.5, 1.0]
    mat_emission = [0.1, 0.1, 0.1, 1.0]
    mat_diffuse = [0.5, 0.5, 0.5, 1.0]
    mat_shininess = [50]

    # glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular)
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient)
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess)

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE)
    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SINGLE_COLOR)

    glEnable(GL_LIGHTING)  # 开灯
    glEnable(GL_LIGHT0)
    glDisable(GL_LIGHT1)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_NORMALIZE)


def set_light1(location, camera_location):
    glEnable(GL_LIGHT1)
    light_position = [location[0], location[1], location[2], 1.0]
    light_ambient = [0.1, 0.1, 0.1, 1.0]
    light_diffuse = [0.5, 0.5, 0.5, 1.0]
    light_specular = [0.5, 0.5, 0.5, 1.0]

    glLightfv(GL_LIGHT1, GL_POSITION, light_position)  # 指定位置
    glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient)  # 设置环境光
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse)  # 设置漫反射光
    glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular)  # 设置镜面反射光

    glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, [camera_location[0, 2], camera_location[1, 2], camera_location[0, 2]]) # 光照方向
    glLightfv(GL_LIGHT1, GL_SPOT_EXPONENT, [0])  # 光照聚集程度
    glLightfv(GL_LIGHT1, GL_SPOT_CUTOFF, [30])  # 光照扩散角度

    mat_specular = [0.1, 0.1, 0.1, 1.0]
    mat_ambient = [0.5, 0.5, 0.5, 1.0]
    mat_emission = [0.05, 0.05, 0.05, 1.0]
    mat_diffuse = [0.5, 0.5, 0.5, 1.0]
    mat_shininess = [10]

    # glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess)

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE)
    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SINGLE_COLOR)

    glEnable(GL_LIGHTING)  # 开灯
    glEnable(GL_LIGHT1)
    glDisable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_NORMALIZE)

class camera():
    class Ortho:
        # left, right, bottom, top, near, far
        params = np.array([-1, 1, -1, 1, -1, 1], np.float64)
        # params = np.array([-1, 1, -1, 1, 1, -1], np.float64)
        bbox = params[0:4]
        nf = params[4:]  # near far


def pos_get_pos3d(pos):
    x = pos[0]
    y = param.viewport[1] - pos[1]
    # 这个是说读出一个矩形 左下角， 长宽
    z = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)  # 获得的是 NDC中的坐标
    # print(x, y, z)

    # 获得必要的矩阵
    # 4 16 16
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)

    # print(viewport)
    # print(modelview)

    modelview[3, 3] = 0.0005
    # print(projection)
    xyz = gluUnProject(x, y, z, modelview, projection, viewport)
    # print(xyz)
    return xyz


def draw_pos(pos3d, size=3, color=None):
    if color is None:
        color = [0, 0, 1]
    glPointSize(size)
    glBegin(GL_POINTS)
    # print(pos3d)

    glColor3f(*color)
    glVertex3f(*pos3d)

    glEnd()
    glColor3f(1, 1, 1)


def pos_get_pos3d_show(pos):
    # p(pos)
    pos3d = pos_get_pos3d(pos)
    param.pos3d = pos3d
    # draw_pos(pos3d)
    # p("pos3d", pos3d)


def Rendercylinder(x0, y0, z0, x1, y1, z1, radius):
    dir_x = x1 - x0
    dir_y = y1 - y0
    dir_z = z1 - z0
    bone_length = np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)

    quad_obj = gluNewQuadric()
    gluQuadricDrawStyle(quad_obj, GLU_FILL)
    gluQuadricNormals(quad_obj, GLU_SMOOTH)
    glPushMatrix()
    #  平移到起始点
    glTranslated(x0, y0, z0)
    #  计算长度

    length = np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    if length < 0.000001:
        length = 0.0001
    # print(length)
    dir_x /= length
    dir_y /= length
    dir_z /= length

    up_x = 0.0
    up_y = 1.0
    up_z = 0.0

    side_x = up_y * dir_z - up_z * dir_y
    side_y = up_z * dir_x - up_x * dir_z
    side_z = up_x * dir_y - up_y * dir_x
    length = np.sqrt(side_x * side_x + side_y * side_y + side_z * side_z)
    if length < 0.000001:
        length = 0.0001
    side_x /= length
    side_y /= length
    side_z /= length
    up_x = dir_y * side_z - dir_z * side_y
    up_y = dir_z * side_x - dir_x * side_z
    up_z = dir_x * side_y - dir_y * side_x
    #  计算变换矩阵
    m = [side_x, side_y, side_z, 0.0,
         up_x, up_y, up_z, 0.0,
         dir_x, dir_y, dir_z, 0.0,
         0.0, 0.0, 0.0, 1.0]
    glMultMatrixd(m)
    #  圆柱体参数
    slices = 8  # 每个圆柱多少面
    stack = 2  # 2段圆柱

    gluCylinder(quad_obj, radius, radius, bone_length, slices, stack)
    glPopMatrix()


# 通过内参构造投影矩阵
def InitProjectMat(intrinsic, w, h, znear, zfar):
    proj = np.zeros((4, 4), dtype=np.float64)

    proj[0, 0] = 2 * intrinsic[0, 0] / w
    proj[1, 1] = 2 * intrinsic[1, 1] / h

    proj[0, 2] = (w - 2 * intrinsic[0, 2]) / w
    proj[1, 2] = (h - 2 * intrinsic[1, 2]) / h
    proj[2, 2] = (-zfar - znear) / (zfar - znear)
    proj[3, 2] = -1.0

    proj[2, 3] = -2.0 * zfar * znear / (zfar - znear)

    return proj


def drawText(x, y, text, font):
    textSurface = font.render(text, True, (255, 255, 66, 255), (0, 66, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

# @profile
# def DisplayResults(x, xtriangle, spherelist_1, spherelist_2, facelist, shared_mem, shared_fps):
def DisplayResults(x, xtriangle, path0, spherelist_1, spherelist_2, F, rx, ry, tx, ty, zpos,
                   constraint_triangle, ResultNode, l1, facelist, na, la, velocity_InsertionUnit, camera_matrix, camera_T_copy, gl_list, gl_list1, proj, location, vNormal, font, fps, scaled_factor):
    #  显示计算结果

    glNewList(gl_list, GL_COMPILE)

    # glDisable(GL_TEXTURE_2D)
    # glEnable(GL_TEXTURE_2D)
    # glDisable(GL_DEPTH_TEST)
    # glDepthFunc(GL_ALWAYS)
    # 绘制局部坐标系
    # glLineWidth(3)
    # glBegin(GL_LINES)
    # glColor3f(1, 0, 0)
    # glVertex3fv(ResultNode[-1])
    # glColor3f(1, 0, 0)
    # glVertex3fv(camera_matrix[3:6])
    # glEnd()
    # glBegin(GL_LINES)
    # glColor3f(0, 0, 1)
    # glVertex3fv(ResultNode[-1])
    # glColor3f(0, 0, 1)
    # glVertex3fv(ResultNode[-1] + camera_matrix[6:9])
    # glEnd()

    # 绘制定位输入局部坐标系
    # cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * 3.1415926535 / 2))
    # camera_location = eulerAnglesToRotationMatrix(location[3:6], 'zyx') @ cy
    # glLineWidth(3)
    # glBegin(GL_LINES)
    # glColor3f(0, 0, 1)
    # glVertex3f(location[0], location[1], location[2])
    # glColor3f(0, 0, 1)
    # glVertex3f(location[0] + camera_location[0, 2], location[1] + camera_location[1, 2],
    #            location[2] + camera_location[2, 2])
    # glEnd()
    # glBegin(GL_LINES)
    # glColor3f(0, 1, 0)
    # glVertex3f(location[0], location[1], location[2])
    # glColor3f(0, 1, 0)
    # glVertex3f(location[0] + camera_location[0, 1], location[1] + camera_location[1, 1],
    #            location[2] + camera_location[2, 1])
    # glEnd()
    # glBegin(GL_LINES)
    # glColor3f(1, 0, 0)
    # glVertex3f(location[0], location[1], location[2])
    # glColor3f(1, 0, 0)
    # glVertex3f(location[0] + camera_location[0, 0], location[1] + camera_location[1, 0],
    #            location[2] + camera_location[2, 0])
    # glEnd()
    # 绘制柔性机器人
    if la < 1:
        na = 0
        la = 1
    glLineWidth(3)
    for i in range(len(ResultNode) - 1):
        if i > int(l1):
            # glColor3f(0, 0.7, 0.7)
            glColor4f(0.1, 0.1, 0.7, 1.0)
            Rendercylinder(ResultNode[i][0],
                           ResultNode[i][1],
                           ResultNode[i][2],
                           ResultNode[i + 1][0],
                           ResultNode[i + 1][1], ResultNode[i + 1][2], 0.3)
        elif i == int(l1):
            glColor4f(0.1, 0.1, 0.7, 1.0)
            Rendercylinder(ResultNode[i][0], ResultNode[i][1], ResultNode[i][2],
                           ResultNode[i][0] + (l1 - int(l1)) * (
                                   ResultNode[i + 1][0] - ResultNode[i][0]),
                           ResultNode[i][1] + (l1 - int(l1)) * (
                                   ResultNode[i + 1][1] - ResultNode[i][1]),
                           ResultNode[i][2] + (l1 - int(l1)) * (
                                   ResultNode[i + 1][2] - ResultNode[i][2]), 0.5)
            glColor4f(0.1, 0.1, 0.7, 1.0)
            Rendercylinder(
                ResultNode[i][0] + (l1 - int(l1)) * (ResultNode[i + 1][0] - ResultNode[i][0]),
                ResultNode[i][1] + (l1 - int(l1)) * (ResultNode[i + 1][1] - ResultNode[i][1]),
                ResultNode[i][2] + (l1 - int(l1)) * (ResultNode[i + 1][2] - ResultNode[i][2]),
                ResultNode[i + 1][0],
                ResultNode[i + 1][1], ResultNode[i + 1][2], 0.3)
        elif i < int(l1):
            glColor4f(0.1, 0.1, 0.7, 1.0)
            Rendercylinder(ResultNode[i][0], ResultNode[i][1], ResultNode[i][2],
                           ResultNode[i + 1][0],
                           ResultNode[i + 1][1],
                           ResultNode[i + 1][2], 0.5)

    # 绘制受力
    # for i in range(int(len(F) / 6)):
    #     # print(i)
    #     glBegin(GL_LINES)
    #     glColor3f(0., 0.8, 0.8)
    #     glVertex3f(ResultNode[i][0], ResultNode[i][1], ResultNode[i][2])
    #     glColor3f(0., 0.8, 0.8)
    #     glVertex3f(ResultNode[i][0] + F[i * 6 + 0], ResultNode[i][1] + F[i * 6 + 1],
    #                ResultNode[i][2] + F[i * 6 + 2])
    #     # glVertex3f(ResultNode[i + 1][0], ResultNode[i + 1][1], ResultNode[i + 1][2])
    #     glEnd()
    # 绘制碰撞面片
    # glNewList(gl_list1, GL_COMPILE)
    for i in constraint_triangle:
        vertices1 = xtriangle[int(i)]
        glBegin(GL_POLYGON)
        for j in range(len(vertices1)):
            glColor4f(1.0, 0., 0., 1.0)
            glNormal3fv(vNormal[int(i)])
            glVertex3fv(x[vertices1[j]])
        glEnd()
    # glEndList()
    # glDepthFunc(GL_LESS)
    glEndList()

    glClearColor(1.0, 1.0, 1.0, 1.0)  # 背景颜色
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    for i in range(2):
        if i == 0:
            # glviewport实现窗口分屏， 全局视角
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            set_light0()
            '''cam.Ortho.bbox[:] = cam.Ortho.bbox * 13
            cam.Ortho.nf[:] = cam.Ortho.nf * 200'''
            # glOrtho(*cam.Ortho.params)
            # gluPerspective(fovy=60, aspect=1, zNear=0.1, zFar=1000)
            # gluPerspective(60, 1, 0.1, 10000)
            # gluPerspective(60, 1, 20000, 0.001)

            # glOrtho(3,-3,-4,4,1,50)

            glViewport(0, 0, 1920, 1080)
            gluPerspective(45, 1920/1080, 0.01, 1000)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
            # gluLookAt(0, 0, 800, 0, 0, 0, 0, 1, 0)

            # glClearColor(0.6, 0.6, 0.6, 0.0)

            # glLoadIdentity()
            # RENDER OBJECT
            glTranslate(tx / 3., ty / 3., -5 * zpos)
            glRotate(ry / 3, 1, 0, 0)
            glRotate(rx / 3, 0, 1, 0)
            # glFrontFace(GL_CCW) # 默认为CCW正方向
            glShadeModel(GL_SMOOTH)  # 设置着色模式为：平滑底纹

            # glEnable(GL_LIGHT0)

            # glEnable(GL_LIGHTING)  # 开启光照
            # glBlendFunc(GL_SRC_ALPHA, GL_DST_COLOR)
            # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendFunc(GL_ONE, GL_ZERO)
            # glBlendFunc(GL_ONE, GL_ONE)
            # glEnable(GL_LIGHT0)
            # gluLookAt(0, 0, 0, 0, 0, 1, 0, 1, 0)
            # glRotate(100, 0, 0, 1)

            # glRotate(180, 0, 1, 0)  # 这个是通过观察得到的
            # ps = np.array(x)

            # vmin = x.min(axis=0)
            # vmax = x.max(axis=0)
            # print(vmin)
            # print(vmax)
            # [7.932819 10.268884  4.915966]
            # #
            # bbox_center = (vmax + vmin) / 2
            # print(bbox_center)
            # bbox_half_r = np.max(vmax - vmin) / 2
            #
            # s = [2 / bbox_half_r] * 3
            # glScale(*s)

            '''t = -bbox_center
            glTranslate(*t)'''
            # glCallList(facelist)
            # glCallList(spherelist_1)
            # ---> 新增：利用 GPU 的矩阵运算直接缩放静态球体表面，0性能损耗 <---
            glPushMatrix()
            glTranslated(50.0, 5.0, 80.0)  # 3. 移回原位
            glScalef(scaled_factor, scaled_factor, scaled_factor)  # 2. 缩放
            glTranslated(-50.0, -5.0, -80.0)  # 1. 将球心移到世界原点
            glCallList(spherelist_1)
            glPopMatrix()
            # -------------------------------------------------------------

            glCallList(gl_list1)
            # glDeleteLists(gl_list1)

            glCallList(spherelist_2)
            glCallList(gl_list)

            if hasattr(param, 'pos3d') and param.sel_pos:
                draw_pos(param.pos3d)
        # if i == 1:
        #     # 第一视角
        #     glViewport(1280, 600, 640, 480)
        #     glMatrixMode(GL_PROJECTION)
        #     glLoadIdentity()
        #
        #     # gluPerspective(69, 640/480, 0.001, 10000)
        #     glLoadMatrixf(np.transpose(proj))
        #
        #     glMatrixMode(GL_MODELVIEW)
        #     glLoadIdentity()
        #
        #     # RENDER OBJECT
        #     # glTranslate(tx / 20., ty / 20., -zpos)
        #     # glRotate(ry / 5, 1, 0, 0)
        #     # glRotate(rx / 5, 0, 1, 0)
        #     # lr = np.linalg.norm(ResultNode[-2])
        #     gluLookAt(camera_matrix[0],camera_matrix[1],camera_matrix[2],camera_matrix[3],camera_matrix[4],camera_matrix[5],camera_matrix[6],camera_matrix[7],camera_matrix[8])
        #     # gluLookAt(location[0], location[1], location[2],
        #     #           location[0] + camera_location[0, 0], location[1] + camera_location[1, 0],
        #     #           location[2] + camera_location[2, 0],
        #     #           -camera_location[0, 1], -camera_location[1, 1], -camera_location[2, 1])
        #     '''ps = np.array(x)
        #     vmin = ps.min(axis=0)
        #     vmax = ps.max(axis=0)
        #
        #     bbox_center = (vmax + vmin) / 2
        #     bbox_half_r = np.max(vmax - vmin) / 2
        #
        #     s = [2 / bbox_half_r] * 3
        #     glScale(*s)
        #
        #     t = -bbox_center
        #     glTranslate(*t)'''
        #
        #     glCallList(spherelist_1)
        #     glCallList(gl_list1)
        #     glCallList(spherelist_2)
        #     glCallList(gl_list)
        #     # glCallList(facelist)
        #
        #     if hasattr(param, 'pos3d') and param.sel_pos:
        #         draw_pos(param.pos3d)
    # rotatebool = 1
    drawText(100, 120, "%d" % int(fps), font)
    pygame.display.flip()

    return

class LoadObject:
    def __init__(self, trans_matrix, shared_draw, mode, shared_location, world_point):
        self.shared_RL_draw = 1
        # try:
        #     skullPath = "D:/陈浩/2022年研究/FEM_bronch/cada/cada/model/skull_move.stl"
        #     skull = trimesh.load(skullPath)
        # except:
        #     skullPath = "/home/cair/Documents/arproject/fem/model/skull_move.stl"
        #     skull = trimesh.load(skullPath)
        # self.skull = np.array(skull.vertices, dtype=np.float64)  # 读取点
        # self.skulltriangle = np.array(skull.faces, dtype=np.int32)  # 读取面
        # # try:
        # self.skull = self.skull @ np.transpose(trans_matrix[0:3, 0:3])
        # self.skull += 1000 * np.transpose(trans_matrix[0:3, 3])
        try:
            # D:/陈浩/2022年研究/cada/cada/model
            # objPath = './data/Segmentation_no_hole_reduced.obj'
            # objPath = './data/Segmentation_8_3.stl'
            objPath = '/home/cairch/chenhao/project/FEM/data_fem/mesh/exp_CAMAE/sphere.STL'
            # objPath = './data/bronch_singlelayer.obj'
            # objPath = './data/br_main.STL'
            obj = trimesh.load(objPath)
        except:
            if mode == 2:
                objPath = "/home/cair/Documents/arproject/data/cadaver6/stl/HEAD2_hole_real.stl"
            elif mode == 1:
                objPath = "/home/cair/Documents/arproject/data/cadaver5/stl/HEAD2_hole_real.stl"
            elif mode == 0:
                objPath = "/home/cair/Documents/arproject/data/headtest/stl/HEAD2_hole_real.stl"
            else:
                print('the paragragh mode is non-precondition')
            # objPath = './model/brain+hole.stl'
            obj = trimesh.load(objPath)

        self.x = np.array(obj.vertices, dtype=np.float64)  # 读取点
        # 旋转90度
        # cx = linalg.expm(np.cross(np.eye(3), [-1., 0., 0.] / linalg.norm([1, 0, 0]) * 3.1415926535/2))
        # self.x = self.x @ np.transpose(cx)
        self.xtriangle = np.array(obj.faces, dtype=np.int32)  # 读取面
        self.x += np.array([10, -35, 40])
        self.original_x = self.x.copy()
        # 单位改成mm
        # self.x *= 10
        try:
            self.x = self.x @ np.transpose(trans_matrix[0:3, 0:3])
            self.x += 1000 * np.transpose(trans_matrix[0:3, 3])
        except:
            print('trans_matrix is', trans_matrix)
        # try:
        #     # SegPath = '/home/mingcong/arproject/fem/model/all_model/new_v123_hole.stl'
        #     SegPath = 'D:/陈浩/2022年研究/FEM_bronch/cada/cada/model/all_model/new_v123_hole.stl'
        #     # SegPath = 'model\\tube_ch.obj'
        #     Segobj = trimesh.load(SegPath)
        # except:
        #     # SegPath = '/home/mingcong/arproject/fem/model/all_model/new_v123_hole.stl'
        #     SegPath = '/home/cair/Documents/arproject/fem/model/all_model/new_v123_hole.stl'
        #     # SegPath = 'model\\tube_ch.obj'
        #     Segobj = trimesh.load(SegPath)
        # self.xSeg = np.array(Segobj.vertices, dtype=np.float64)  # 读取点
        # self.xtriangleSeg = np.array(Segobj.faces, dtype=np.int32)  # 读取面
        # try:
        #     self.xSeg = self.xSeg @ np.transpose(trans_matrix[0:3, 0:3])
        #     self.xSeg += 1000 * np.transpose(trans_matrix[0:3, 3])
        # except:
        #     print('trans_matrix is', trans_matrix)

        self.skel = np.zeros([100, 3], dtype=np.float64)
        # self.skeletonlist = np.array(self.skel.vertices, dtype=np.float64)
        self.skeletonlist = np.zeros([100, 3], dtype=np.float64)
        # 绘图预备
        param.obj = obj
        param.sel_pos = False
        pygame.init()
        viewport = (1920, 1080)
        param.viewport = viewport
        screen = pygame.display.set_mode(viewport, pygame.OPENGL | pygame.DOUBLEBUF)
        self.font = pygame.font.SysFont('arial', 32)
        self.fps = 0
        param.screen = screen
        pygame.key.set_repeat(1, 10)
        # set_light0()

        # font = pygame.font.SysFont("arial", 20)
        # text_surface = font.render("Hello！", True, (255, 0, 0))
        faces = self.xtriangle
        vertices = self.x
        # 设置gl列表
        self.spherelist_1 = glGenLists(1)
        self.spherelist_2 = glGenLists(1)
        self.facelist = glGenLists(1)

        # print("texture get")
        glNewList(self.facelist, GL_COMPILE)
        '''glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        for i in range(len(faces11)):
            vertices1 = faces11[i]
            tex = tex11[i]
            glBegin(GL_POLYGON)
            glColor4f(1.0, 1.0, 1.0, 0.5)
            for j in range(len(vertices1)):
                # print(vertices11[vertices1[j]])
                glTexCoord2f(texcoord11[tex[j]][0], texcoord11[tex[j]][1])
                glVertex3fv(vertices11[vertices1[j]])
            glEnd()'''
        glEndList()

        # 深度测试

        # glDepthMask(GL_FALSE)
        glEnable(GL_DEPTH_TEST)
        # glDepthFunc(GL_ALWAYS)
        # glEnable(GL_BLEND)
        # glDepthMask(GL_FALSE)
        #  打开混合模式
        # print(time.time())
        glNewList(self.spherelist_1, GL_COMPILE)
        # glEnable(GL_DEPTH_TEST)
        # glDepthMask(GL_TRUE)
        # glEnable(GL_BLEND)
        # self.skull = np.array(self.skull, dtype=np.float64)
        # self.skulltriangle = np.array(self.skulltriangle, dtype=np.int32)
        # self.skeletonlist = np.array(self.skeletonlist, dtype=np.float64)
        # self.skullNormal, self.dskull = fa_vector(self.skull, self.skulltriangle, self.skeletonlist)
        # glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # glCullFace(GL_BACK) #  剔除背面
        # glEnable(GL_CULL_FACE) #  启用剔除功能
        # glFrontFace(GL_CCW)
        # 显示头骨
        # for i in range(len(self.skulltriangle)):
        #     glBegin(GL_POLYGON)
        #     for j in range(3):
        #         glColor4f(0.7, 0.6, 0.4, 0.5)
        #         glNormal3f(self.skullNormal[i, 0], self.skullNormal[i, 1], self.skullNormal[i, 2])
        #         glVertex3fv(self.skull[self.skulltriangle[i][j]])
        #     glEnd()
        # 显示主要模型
        self.x = np.array(self.x, dtype=np.float64)
        self.xtriangle = np.array(self.xtriangle, dtype=np.int32)
        self.vNormal, self.distance0 = fa_vector(self.x, self.xtriangle, self.skeletonlist)
        for i in range(len(self.xtriangle)):
            glBegin(GL_POLYGON)
            for j in range(3):
                glColor4f(0.6, 0.6, 0.6, 0.5)
                glNormal3fv(-self.vNormal[i])
                glVertex3fv(self.x[self.xtriangle[i][j]])
            glEnd()

        # glDisable(GL_DEPTH_TEST)
        # glDepthMask(GL_FALSE)
        # glEnable(GL_BLEND)

        #  打开混合模式

        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        # glPolygonMode(GL_BACK, GL_LINE)
        # glEnable(GL_TEXTURE_2D)
        '''glLineWidth(1)
        # 画面片的边
        glColor3f(0.7, 0.6, 0.4)
        for vertices1 in faces:
            for i in range(2):
                glBegin(GL_LINES)
                glVertex3fv(vertices[vertices1[i]])
                glVertex3fv(vertices[vertices1[(i + 1) % 3]])
                glEnd()'''

        # # 显示孔
        '''glColor4f(0.1, 0.5, 0.5, 0.99)
        for vertices1 in self.xtriangleTube:
            glBegin(GL_POLYGON)

            # glColor4f(max(0, self.vNormal[vn][0]), max(self.vNormal[vn][1], 0), max(0, self.vNormal[vn][2]), 1.0)
            for j in range(len(vertices1)):
                glVertex3fv(self.xTube[vertices1[j]])
            glEnd()'''

        # # 显示脑室
        '''glColor4f(0.1, 0.5, 0.6, 0.6)
        for vertices1 in self.xtriangleSeg:
            glBegin(GL_POLYGON)

            # glColor4f(max(0, self.vNormal[vn][0]), max(self.vNormal[vn][1], 0), max(0, self.vNormal[vn][2]), 1.0)
            for j in range(len(vertices1)):
                glVertex3fv(self.xSeg[vertices1[j]])
            glEnd()'''

        # 绘制面片
        # glBegin(GL_TRIANGLES)
        # for vertices1 in faces:
        # glPolygonMode(GL_FRONT,GL_LINE)
        # glPolygonMode(GL_BACK,GL_FILL)
        # glEnable(GL_CULL_FACE)
        # glFrontFace(GL_CW)

        # glCullFace(GL_FRONT)
        # glFrontFace(GL_CW)
        # glEnable(GL_CULL_FACE)
        # for i in range(len(self.skulltriangle)):
        #     glBegin(GL_POLYGON)
        #     for j in [0,2,1]:
        #         # j = 2-j
        #         glColor4f(0.7, 0.1, 0.1, 0.5)
        #         glNormal3f(-self.skullNormal[i,0],-self.skullNormal[i,1],-self.skullNormal[i,2])
        #         glVertex3fv(self.skull[self.skulltriangle[i][j]])
        #     glEnd()
        # glDisable(GL_CULL_FACE)
        # vn = self.vNormal.copy()
        # for i in range(len(vn)):
        '''
        glColor4f(0.1, 0.5, 0.6, 0.3)
        for vertices1 in self.xtriangleSeg:
            glBegin(GL_POLYGON)

            # glColor4f(max(0, self.vNormal[vn][0]), max(self.vNormal[vn][1], 0), max(0, self.vNormal[vn][2]), 1.0)
            for j in range(len(vertices1)):
                glVertex3fv(self.xSeg[vertices1[j]])
            glEnd()
        '''

        #  关闭混合模式
        # glEnable(GL_DEPTH_TEST)
        # glDisable(GL_BLEND)
        # glDepthMask(GL_TRUE)
        glEndList()
        # print(time.time())
        # print("face get")

        glNewList(self.spherelist_2, GL_COMPILE)
        # 绘制实验曲线
        # glLineWidth(4)
        # for i in range(len(world_point) - 1):
        #     glColor4f(0.0, 0.0, 0.5, 1.0)
        #     Rendercylinder(world_point[i][0],
        #                    world_point[i][1],
        #                    world_point[i][2],
        #                    world_point[i + 1][0],
        #                    world_point[i + 1][1], world_point[i + 1][2], 1.44)
        # 画路径
        '''for i in range(len(self.path0) - 1):
            glBegin(GL_LINES)
            glColor3f(0, 1, 0)
            glVertex3f(self.path0[i][0], self.path0[i][1], self.path0[i][2])
            glColor3f(0, 1, 0)
            glVertex3f(self.path0[i + 1][0], self.path0[i + 1][1], self.path0[i + 1][2])
            glEnd()'''

        '''glLineWidth(2)

        # 画骨架
        for i in range(len(self.skeletonlist) - 1):
            glBegin(GL_POINTS)
            glColor3f(0, 1, 0)
            glVertex3f(self.skeletonlist[i][0], self.skeletonlist[i][1], self.skeletonlist[i][2])
            glEnd()'''

        # 画法向量
        # for i in range(len(faces)):
        #     vertices1 = faces[i]
        #     glBegin(GL_LINES)
        #     glColor3f(0, 0, 1)
        #     glVertex3fv(vertices[vertices1[0]])
        #     glVertex3fv(vertices[vertices1[0]] + self.vNormal[i])
        #     glEnd()

        glLineWidth(3)

        glBegin(GL_LINES)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 30)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 30, 0)
        glEnd()
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glColor3f(1, 0, 0)
        glVertex3f(30, 0, 0)
        glEnd()

        glEndList()

        self.rx, self.ry = (0, 0)
        self.tx, self.ty = (0, 0)
        self.zpos = 0
        self.path0 = None
        self.event = np.array([0, 0, shared_draw['event'][0]])
        shared_draw['event'] = self.event
        self.rotate = False
        self.move = False
        self.set_locate = False
        self.na = 0
        self.gl_list = glGenLists(1)
        self.gl_list1 = glGenLists(1)

        self.proj = InitProjectMat(shared_location['intrinsic'], 640, 480, 0.001, 10000)

    def DrawResult(self, shared_draw, shared_location, shared_RL):
        self.event = np.array([0, 0, shared_draw['event'][2]])
        self.F = shared_draw['F']
        self.constraint_triangle = shared_draw['constraint_triangle']
        self.ResultNode = shared_draw['ResultNode'] * 1000
        self.l1 = shared_draw['l1']
        self.na = shared_draw['na']
        self.la = shared_draw['la']
        self.velocity_InsertionUnit = shared_draw['velocity_InsertionUnit']
        self.camera_matrix = shared_draw['camera_matrix'] * 1000
        self.camera_T_start = shared_draw['camera_T_start']
        self.fps = shared_draw['fps']

        # 变化小球大小
        self.scaled_factor = shared_draw['scaled_factor']
        self.center = np.array([50.0, 5., 80.])
        # self.x = self.center + (self.original_x - self.center) * self.scaled_factor

        location = shared_location['location']
        if shared_RL['draw']:
            DisplayResults(self.x, self.xtriangle, self.path0, self.spherelist_1, self.spherelist_2, self.F, self.rx,
                           self.ry, self.tx, self.ty, self.zpos, self.constraint_triangle, self.ResultNode, self.l1,
                           self.facelist, self.na, self.la, self.velocity_InsertionUnit,
                           self.camera_matrix, self.camera_T_start, self.gl_list, self.gl_list1, self.proj, location, self.vNormal, self.font, self.fps, self.scaled_factor)  # cost 18ms
        if shared_draw['event'][2] == 5:  # e.type == pygame.QUIT:
            sys.exit()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.event[2] = 5
                shared_draw['event'] = self.event

                pygame.quit()
                sys.exit()

            elif e.type == pygame.KEYDOWN:
                self.event[0] = 1
                if e.key == pygame.K_ESCAPE:
                    self.event[1] = 1
                    shared_draw['event'] = self.event
                    pygame.quit()
                    sys.exit()

                if e.key == pygame.K_1:
                    self.event[1] = 2
                    break
                elif not shared_draw['set_locate']:
                    if e.key == pygame.K_2:
                        self.set_locate = True
                        self.event[1] = 3
                        break
                    elif e.key == pygame.K_w:
                        self.event[1] = 4
                        break
                    elif e.key == pygame.K_s:
                        self.event[1] = 5
                        break
                    elif e.key == pygame.K_z:
                        self.event[1] = 6
                        break
                    elif e.key == pygame.K_x:
                        self.event[1] = 7
                        break
                    elif e.key == pygame.K_a:
                        self.event[1] = 8
                        break
                    elif e.key == pygame.K_d:
                        self.event[1] = 9
                        break

                    elif e.key == pygame.K_q:
                        self.event[1] = 10
                        break
                    elif e.key == pygame.K_e:
                        self.event[1] = 11
                        break
                    elif e.key == pygame.K_UP:
                        self.event[1] = 12

                        break
                    elif e.key == pygame.K_DOWN:
                        self.event[1] = 13
                        break
                    elif e.key == pygame.K_LEFT:
                        self.event[1] = 14
                        break
                    elif e.key == pygame.K_RIGHT:
                        self.event[1] = 15
                        break
                    elif e.key == pygame.K_o:
                        self.event[1] = 16
                        break
                    elif e.key == pygame.K_i:
                        self.event[1] = 17
                        break

                    elif e.key == pygame.K_SPACE:
                        self.event[1] = 18
                        break
                    elif e.key == pygame.K_r:
                        self.event[1] = 32
                        break
                    elif e.key == pygame.K_j:
                        self.event[1] = 33
                        break
                    elif e.key == pygame.K_8:
                        self.shared_RL_draw = 1
                        break
                    elif e.key == pygame.K_9:
                        self.shared_RL_draw = 0
                        break

                elif shared_draw['set_locate']:
                    if e.key == pygame.K_3:
                        self.set_locate = False
                        self.event[1] = 19
                        break
                    elif e.key == pygame.K_w:
                        self.event[1] = 20
                        break
                    elif e.key == pygame.K_s:
                        self.event[1] = 21
                        break
                    elif e.key == pygame.K_a:
                        self.event[1] = 22
                        break
                    elif e.key == pygame.K_d:
                        self.event[1] = 23
                        break
                    elif e.key == pygame.K_q:
                        self.event[1] = 24
                        break
                    elif e.key == pygame.K_e:
                        self.event[1] = 25
                        break
                    elif e.key == pygame.K_i:
                        self.event[1] = 26
                        break
                    elif e.key == pygame.K_k:
                        self.event[1] = 27
                        break
                    elif e.key == pygame.K_j:
                        self.event[1] = 28
                        break
                    elif e.key == pygame.K_l:
                        self.event[1] = 29
                        break
                    elif e.key == pygame.K_u:
                        self.event[1] = 30
                        break
                    elif e.key == pygame.K_o:
                        self.event[1] = 31
                        break
            elif e.type == pygame.MOUSEBUTTONDOWN:

                # pressed_array = pygame.mouse.get_pressed()
                # if pressed_array[0]:  # 左键被按下
                #     if self.param.sel_pos:
                #         pos = pygame.mouse.get_pos()  # 获得鼠标位置
                #         # print("the point you select is ", pos)
                #         pos_get_pos3d_show(pos)

                if e.button == 4:
                    # zpos = max(1, zpos - 1)
                    self.zpos -= 1
                elif e.button == 5:
                    self.zpos += 1
                elif e.button == 1:
                    self.rotate = True
                elif e.button == 3:
                    self.move = True
                self.rotatebool = 0
                self.camulatebool = False
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    self.rotate = False
                elif e.button == 3:
                    self.move = False
                self.rotatebool = 0
                self.camulatebool = False
            elif e.type == pygame.MOUSEMOTION:
                i, j = e.rel
                if self.rotate:
                    self.rx -= i
                    self.ry -= j
                    self.rotatebool = 0
                    self.camulatebool = False
                if self.move:
                    self.tx += i
                    self.ty -= j
                    self.rotatebool = 0
                    self.camulatebool = False

        shared_draw['event'] = self.event
        shared_RL['draw'] = self.shared_RL_draw

def draw(trans_matrix, shared_draw, mode, shared_location, shared_RL, world_point):
    try:
        Object = LoadObject(trans_matrix, shared_draw, mode, shared_location, world_point)
        while 1:
            Object.DrawResult(shared_draw, shared_location, shared_RL)
    except SystemExit:
        print('OpenGL正常退出')
    except (Exception, BaseException) as errorstring:
        print('OpenGL退出，原因是：%s' % str(errorstring))
        print(traceback.format_exc())
    finally:
        event = np.array([0, 0, 5])
        shared_draw['event'] = event
