import numpy as np

import os
import sys
import time
import trimesh
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
import glfw

from shader import shader
import ctypes


def Draw(VAO, VBO, EBO):
    # glClear(GL_COLOR_BUFFER_BIT)
    glBindVertexArray(VAO)

    # glDrawArrays(GL_TRIANGLES, 0, 3)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    glBindVertexArray(0)


class LoadObject:
    def __init__(self, trans_matrix, shared_draw, mode):
        try:
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
        except:
            # D:/陈浩/2022年研究/cada/cada/model
            objPath = 'D:/陈浩/2022年研究/20221102尸体实验到管线碰撞/data/Segmentation_no_hole_reduced.obj'
            obj = trimesh.load(objPath)

        self.x = np.array(obj.vertices, dtype=np.float32)  # 读取点
        self.xtriangle = np.array(obj.faces, dtype=np.int32)  # 读取面
        try:
            self.x = self.x @ np.transpose(trans_matrix[0:3, 0:3])
            self.x += 1000 * np.transpose(trans_matrix[0:3, 3])
        except:
            print('trans_matrix is', trans_matrix)
        # try:
        #     # SegPath = '/home/mingcong/arproject/fem/model/all_model/new_v123_hole.stl'
        #     SegPath = 'D:/陈浩/2022年研究/cada/cada/model/all_model/new_v123_hole.stl'
        #     # SegPath = 'model\\tube_ch.obj'
        #     Segobj = trimesh.load(SegPath)
        # except:
        #     # SegPath = '/home/mingcong/arproject/fem/model/all_model/new_v123_hole.stl'
        #     SegPath = '/home/cair/Documents/arproject/fem/model/all_model/new_v123_hole.stl'
        #     # SegPath = 'model\\tube_ch.obj'
        #     Segobj = trimesh.load(SegPath)
        # self.xSeg = np.array(Segobj.vertices, dtype=np.float32)  # 读取点
        # self.xtriangleSeg = np.array(Segobj.faces, dtype=np.int32)  # 读取面
        # try:
        #     self.xSeg = self.xSeg @ np.transpose(trans_matrix[0:3, 0:3])
        #     self.xSeg += 1000 * np.transpose(trans_matrix[0:3, 3])
        # except:
        #     print('trans_matrix is', trans_matrix)

        self.skel = np.zeros([100, 3], dtype=np.float32)
        # self.skeletonlist = np.array(self.skel.vertices, dtype=np.float32)
        self.skeletonlist = np.zeros([100, 3], dtype=np.float32)


def CreateBuffer():  # 创建顶点缓存器

    # global VBO  # 设置为全局变量

    # vertex = np.array([[-1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    #
    #                    [1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    #
    #                    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    #                    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]], dtype="float32")  # 创建顶点数组

    trans_matrix = np.eye(4)
    shared_draw = None
    mode = 1
    OBJ = LoadObject(trans_matrix, shared_draw, mode)
    # vertex = OBJ.x
    # indices = OBJ.xtriangle

    vertex = np.array([[-1.0, -1.0, 0.0],

                       [1.0, -1.0, 0.0],

                       [0.0, 1.0, 0.0],
                       [0.8, 0.8, 0.0]], dtype="float32")  # 创建顶点数组
    indices = np.array([[1, 2, 3],
                       [0, 1, 2]], dtype=np.int32)  # 顶点索引

    # VBO = glGenBuffers(1)  # 创建缓存
    # glBindBuffer(GL_ARRAY_BUFFER, VBO)  # 绑定
    # glBufferData(GL_ARRAY_BUFFER, vertex.nbytes, vertex, GL_STATIC_DRAW)  # 输入数据
    # VAO = 0

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex.nbytes, vertex, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, 3 * sizeof(GLfloat), None)
    # glEnableVertexAttribArray(1)
    # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), ctypes.c_void_p(3 * sizeof(GLfloat)))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO, VBO, EBO


def key_callback(window, key, scancode, action, mode):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, GL_TRUE)




def display():
    #  Initialize GLEW to setup the OpenGL Function pointers
    glewExperimental = GL_TRUE
    glfw.init()
    window = glfw.create_window(800, 600, "ARProject", None, None)
    glfw.make_context_current(window)

    # #  Set the required callback functions
    glfw.set_key_callback(window, key_callback)
    # glfw.set_cursor_pos_callback(window, mouse_callback)
    # glfw.set_scroll_callback(window, scroll_callback)
    # #  Options
    glfw.set_input_mode(window, glfw.STICKY_KEYS, True)

    # Define the viewport dimensions
    glViewport(0, 0, 800, 600)

    #  Setup some OpenGL options
    # glEnable(GL_DEPTH_TEST)
    # glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    VAO, VBO, EBO = CreateBuffer()

    shaderProgram = shader.Shader("./shader/GLSL/vertex_shader.vs.glsl", "./shader/GLSL/fragment_shader.fs.glsl")

    shaderProgram.use()
    model = np.eye(4)
    view = np.eye(4)
    projection = np.eye(4)
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram.shaderProgram, 'view'), 1, GL_FALSE, view)
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram.shaderProgram, 'projection'), 1, GL_FALSE, projection)
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram.shaderProgram, 'model'), 1, GL_FALSE, model)


    lightPos = np.array([1.2, 1.0, -2.0])
    viewPos = np.transpose(view[0:3, 3])
    lightColor = np.array([1.0, 0.5, 0.3])
    objectColor = np.array([1.0, 1.0, 1.0])
    glUniform3fv(glGetUniformLocation(shaderProgram.shaderProgram, 'lightPos'), 1, GL_FALSE, lightPos)
    glUniform3fv(glGetUniformLocation(shaderProgram.shaderProgram, 'viewPos'), 1, GL_FALSE, viewPos)
    glUniform3fv(glGetUniformLocation(shaderProgram.shaderProgram, 'lightColor'), 1, GL_FALSE, lightColor)
    glUniform3fv(glGetUniformLocation(shaderProgram.shaderProgram, 'objectColor'), 1, GL_FALSE, objectColor)
    lastFrame = glfw.get_time()
    glClearColor(0.2, 0.3, 0.3, 1.0)
    camera = shader.camera1(view)
    while not glfw.window_should_close(window):
        a = time.time()
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT)
        # shaderProgram.use()
        Draw(VAO, VBO, EBO)

        glfw.swap_buffers(window)

        currentFrame = glfw.get_time()
        deltaTime = currentFrame - lastFrame
        lastFrame = currentFrame
        cameraSpeed = deltaTime
        print(cameraSpeed)
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            camera.translate(np.array([0., 0., 0.1])) #  上正

        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            camera.translate(np.array([0., 0., -0.1])) # 下负

        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            camera.translate(np.array([0.1, 0., 0.])) # 左正

        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            camera.translate(np.array([-0.1, 0., 0.])) # 右负


        # if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        #     xpos, ypos = glfw.get_cursor_pos(window)
        #     if lastXpos == 0.0 and lastYpos == 0.0:
        #         lastXpos = xpos
        #         lastYpos = ypos
        #
        #     else:
        #         double xoffset = lastXpos - xpos
        #         double yoffset = lastYpos - ypos
        #         lastXpos = xpos
        #         lastYpos = ypos
        #         double sensitivity = 0.02
        #         xoffset *= sensitivity
        #         yoffset *= sensitivity
        #         maincamera.moveUpDownward(yoffset)
        #         maincamera.moveLRward(xoffset)
        #
        #
        # if (glfwGetMouseButton(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS){
        # # ~~~~~
        # }
        # if (glfwGetMouseButton(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.RELEASE & & \
        #         glfwGetMouseButton(window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE) {
        # lastXpos = 0.0f;
        # lastYpos = 0.0f;
        # }
        # }
        # view = camera.getview()

        # print(view)

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.shaderProgram, 'view'), 1, GL_FALSE, np.transpose(view))

    shaderProgram.delete()
    glfw.terminate()
    sys.exit()


if __name__ == '__main__':
    display()
