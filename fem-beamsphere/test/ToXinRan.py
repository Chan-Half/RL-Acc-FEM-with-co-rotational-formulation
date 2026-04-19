import random
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import scipy.linalg as linalg
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
import multiprocessing as mp
os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])


class camera1:
    def __init__(self, view):
        self.view0 = view.copy()
        self.view = view
        self.derect = self.view[3:6]-self.view[0:3]

    # 旋转矩阵 欧拉角
    def rotate(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        self.derect = np.dot(self.derect, np.transpose(rot_matrix))

        self.view[3:6] = self.view[0:3] + self.derect
        # self.view[6:9] = np.dot(rot_matrix, self.view0[6:9])

    def translate(self, transposition):
        self.view[0:3] = self.view0[0:3] + transposition
        self.view[3:6] = self.view0[3:6] + transposition

    def getview(self):
        return self.view

class param(object):
    pass


def set_light0():
    glEnable(GL_LIGHT0)
    light_position = [1000.0, 1000.0, 1000.0, 0.0]
    light_ambient = [0.5, 0.5, 0.5, 1.0]
    light_diffuse = [1.0, 1.0, 1.0, 1.0]
    light_specular = [1.0, 1.0, 1.0, 1.0]

    glLightfv(GL_LIGHT0, GL_POSITION, light_position)  # 指定位置
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)  # 设置环境光
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
    # glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess)

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    # glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SINGLE_COLOR)

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
        params = np.array([-1, 1, -1, 1, -1, 1], np.float32)
        # params = np.array([-1, 1, -1, 1, 1, -1], np.float32)
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
    stack = 1  # 1段圆柱

    gluCylinder(quad_obj, radius, radius, bone_length, slices, stack)
    glPopMatrix()


# 通过内参构造投影矩阵
def InitProjectMat(intrinsic, w, h, znear, zfar):
    proj = np.zeros((4, 4), dtype=np.float32)

    proj[0, 0] = 2 * intrinsic[0, 0] / w
    proj[1, 1] = 2 * intrinsic[1, 1] / h

    proj[0, 2] = (w - 2 * intrinsic[0, 2]) / w
    proj[1, 2] = (h - 2 * intrinsic[1, 2]) / h
    proj[2, 2] = (-zfar - znear) / (zfar - znear)
    proj[3, 2] = -1.0

    proj[2, 3] = -2.0 * zfar * znear / (zfar - znear)

    return proj


def DisplayResults(rx,ry, tx, ty, zpos, axislist, gl_list, shared_draw, cam):    #  显示计算结果
    glNewList(gl_list, GL_COMPILE)
    ResultNode = shared_draw['set_node']
    for i in range(len(ResultNode) - 1):
            glColor4f(0.5, 0.5, 0.5, 1.0)
            Rendercylinder(ResultNode[i][0], ResultNode[i][1], ResultNode[i][2],
                           ResultNode[i + 1][0],
                           ResultNode[i + 1][1],
                           ResultNode[i + 1][2], 1.6)
    glEndList()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    set_light0()
    glViewport(0, 0, 1200, 800)
    gluPerspective(45, 1200/800, 0.1, 10000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    vmin = ResultNode.min(axis=0)
    vmax = ResultNode.max(axis=0)
    bbox_center = (vmax + vmin) / 2
    gluLookAt(0, -400, bbox_center[2], 0, -390, bbox_center[2], -1, 0, 0)

    glRotate(ry / 3, 1, 0, 0)
    glRotate(rx / 3, 0, 1, 0)
    glTranslate(tx / 3., ty / 3., -5 * zpos)



    # cam.translate(np.array([tx / 3., ty / 3., -5 * zpos], dtype=np.float32))
    # cam.rotate([1, 0, 0], ry / 3)
    # cam.rotate([0, 1, 0], rx / 3)
    # look_view = cam.getview()
    # gluLookAt(look_view[0],look_view[1],look_view[2],look_view[3],look_view[4],look_view[5],look_view[6],look_view[7],look_view[8])
    # glShadeModel(GL_SMOOTH)  # 设置着色模式为：平滑底纹
    # glBlendFunc(GL_ONE, GL_ZERO)

    glCallList(axislist)
    glCallList(gl_list)
    pygame.display.flip()
    return


class LoadObject:
    def __init__(self):
        pygame.init()
        viewport = (1200, 800)
        param.viewport = viewport
        screen = pygame.display.set_mode(viewport, pygame.OPENGL | pygame.DOUBLEBUF)
        param.screen = screen
        pygame.key.set_repeat(1, 10)
        self.axislist = glGenLists(1)
        self.gl_list = glGenLists(1)
        glNewList(self.axislist, GL_COMPILE)
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
        self.event = np.array([0, 0])
        self.rotate = False
        self.move = False
        self.set_locate = False
        self.na = 0
        self.gl_list = glGenLists(1)
        self.gl_list1 = glGenLists(1)
        self.cam = camera1(np.array([0., -50., 0., 0., -49., 0., 0., 0., 1.], dtype=np.float32))

    def DrawResult(self, shared_draw):

        DisplayResults(self.rx,self.ry, self.tx, self.ty, self.zpos, self.axislist, self.gl_list, shared_draw, self.cam)  # cost 18ms

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 4:
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


def draw(shared_draw):
    Object = LoadObject()
    while 1:
        Object.DrawResult(shared_draw)


if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    manager = ctx.Manager()
    shared_draw = manager.dict({})
    # 我这里随便初始化了一条直线，注意调整视角是以原点为中心，鼠标左键旋转，右键和滑轮平移
    aa = np.zeros((200, 3), dtype=np.float32)
    for i in range(200):
        aa[i, 2] += 2 * i
    # 下面是你的那些点, 在你自己的函数中改变下面这条就可以了, 如shared_draw['set_node'] = 你的点，最好用np.array
    shared_draw['set_node'] = aa  # np.zeros((100, 3), dtype = np.float32)

    # 这里添加你的程序，把shared_draw作为参数
    process = [ctx.Process(target=draw, args=(shared_draw,)),
               # 把你的程序添加到这里，注意格式和上面一样，逗号不能省
               ]
    [p.start() for p in process]
    [p.join() for p in process]
