"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:  fem_run.py
#  @brief:     fem main function
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2023.03.01
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
import os
import sys
import time
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg
from qpsolvers import solve_qp
import trimesh
import traceback
import cvxopt
from cvxopt import matrix, solvers
# import skeletor
from numba import jit
# from PIL import Image
# from objloader import objloader
# import pycuda.cumath as cumath
# import pycuda.autoinit
import pycuda.driver as cuda
from fem_code.fem_Exclude_Function import fa_vector, ldl_T
from fem_code.fem_cuda_SourceModule import get_cuda_SourceModule
from fem_code.fem_cuda_mesh import get_cuda_mesh
from fem_code.fem_cuda_MatrixStruct import MatrixStruct
from fem_code.fem_function import getnode, TransformMatrix, StiffnessMatrix, resetgK
from fem_code.RotationEularQuaternion import rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix, \
    rotateToQuaternion, QuaternionToAxialAngle
from fem_code.fem_model_mesh import Model
from fem_code.fem_numerical_integration_scheme import implicit_integration, newmark_beta_integration, \
    hht_alpha_integration, implicit_alpha_integration, implicit_beta_integration, newmark_beta_integration_x, \
    hht_alpha_integration_x, nnqp_hht
# from fem_code.fem_numerical_integration_scheme import py_osqp_dynamic_x, penalty_linear_solver

os.chdir(os.path.split(os.path.realpath(sys.argv[0]))[0])


# @jit(nopython=True)
def set_gravity(node_number, rou, A, g, L, T, f):
    # g = 9.8 # np.array([0, 0, 9.8])
    for i in range(node_number - 1):
        force = 0.5 * rou * g * A * L
        f[6 * i + 1] += force
        f[6 * i + 7] += force
        # 这里默认是选取了z轴正方向为重力方向，如果不是，需要修改T的选取
        # torgue = np.dot(T[i, 0:3, 0:3], np.transpose(np.array([0, rou / 12 * g * A * T[i,2,2]*L ** 2, rou / 12 * g * A * T[i,2,1]*L ** 2], dtype=np.float64)))
        # f[6*i + 3:6*i + 6] = torgue
        # f[6 * i + 9:6 * i + 12] = -torgue
        #  * T[i, 2, 1]
        torgue = np.array([0, rou / 12 * g * A * L ** 2, 0], dtype=np.float64)
        f[6 * i + 3:6 * i + 6] += torgue
        f[6 * i + 9:6 * i + 12] -= torgue
    return f


class FEM(object):
    def __init__(self, shared_mem, velocity_InsertionUnit, shared_camera_matrix, mod, trans_matrix, tip_matrix,
                 shared_draw, mode, shared_location, Is_warm_start, cuda_mesh):

        self.x_max = 0
        self.max_time = 0
        self.dirs_index = 0
        self.data_time_stream = ''
        self.is_save_data = False  # 是否保存数据
        self.save_data_CMAME = False  # 是否保存CAMAE数据
        self.CMAME_data_exp2 = []
        self.baseline_exp2 = []


        self.delta_h_time = 1e-3 # 2e-4  #  1e-3  # 模拟时间间隔
        # self.delta_h_time = 2e-4#1e-5  # 2e-4  # 模拟时间间隔
        self.center = np.array([50.0, 5.0, 80.0])
        self.scaled_factor = 1.0
        self.run_time = 0  # 模拟总时间
        self.give_c = False
        self.y = 0
        self.alpha = -0.3
        self.compute_time = []
        self.Nb_ave_constraint = []
        self.time_of_getCons = []
        self.time_of_getK = []
        self.time_of_update = []
        self.time_of_solve = []
        # trans_matrix 模型坐标转深度坐标
        # tip_matrix 末端
        self.hGimbalAngle = np.zeros(2, dtype=np.float64)
        self.hGimbalAngle_copy = np.zeros(2, dtype=np.float64)
        tip_matrix = np.array(tip_matrix, dtype=np.float64)
        trans_matrix = np.array(trans_matrix, dtype=np.float64)
        shared_camera_matrix['trans_matrix'] = trans_matrix
        # trans_matrix = np.transpose(trans_matrix)
        self.shared_camera_matrix = shared_camera_matrix
        self.velocity_InsertionUnit = velocity_InsertionUnit
        # skullPath = "/home/mingcong/arproject/fem/model/skull_move.stl"
        try:
            # D:/陈浩/2022年研究/cada/cada/model
            # objPath = 'D:/陈浩/2022年研究/FEM_bronch/20221104管线碰撞-肺部支气管/data/Segmentation_no_hole_reduced.obj'
            # objPath = './data/Segmentation_8_3.stl'
            objPath = '/home/cairch/chenhao/project/FEM/data_fem/mesh/exp_CAMAE/sphere.STL'
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
        self.xtriangle = np.array(obj.faces, dtype=np.int32)  # 读取面
        self.x += np.array([10, -35, 40])


        self.x_copy = self.x.copy()

        # 旋转90度
        # cx = linalg.expm(np.cross(np.eye(3), [-1., 0., 0.] / linalg.norm([1, 0, 0]) * 3.1415926535/2))
        # self.x = self.x @ np.transpose(cx)
        # 单位改成mm
        # self.x *= 10
        # try:
        #     self.x = self.x @ np.transpose(trans_matrix[0:3, 0:3])
        #     self.x += 1000 * np.transpose(trans_matrix[0:3, 3])
        # except:
        #     print('trans_matrix is None', trans_matrix)
        # get beam model
        # self.gMaterial = np.array([[510.0, 19.233, 9.3, 1.010e-6]])  # 管子号：1， 杨氏模量N/mm^2，惯性矩mm^4，横截面积mm^2, 密度kg/mm^3
        # self.gMaterial = np.array([[5e9, 0.019233, 0.283, 7.85e-6]])  # 管子号：1， 杨氏模量N/mm^2，惯性矩mm^4，横截面积mm^2, 密度kg/mm^3
        # self.gMaterial = np.array([[5e3, 4.24, 0.2827431, 7.85e-6]])  # 管子号：1， 杨氏模量N/mm^2，惯性矩mm^4，横截面积mm^2, 密度kg/mm^3
        self.gMaterial = np.array([[5e9, 4.24e-12, 0.2827431e-6, 7.85e3]])  # 管子号：1， 杨氏模量N/m^2，惯性矩m^4，横截面积m^2, 密度kg/m^3
        # 统一为m的单位
        # self.gMaterial = np.array([[5.1e8, 19.233e-12, 9.3e-6, 1.010e3]])  # 管子号：1， 杨氏模量N/m^2，惯性矩m^4，横截面积m^2, 密度kg/m^3

        self.node_number = 40
        mtype = 'beam'
        if mtype == 'beam':
            # topnode = 10 * np.array(
            #     [[-0.916783988 + 35 * 0.0001, 9.344343185 + 5 + 35 * 0.999999, -1.924159646 + 35 * (-0.0001)],
            #      [-0.916783988 - 5 * 0.0001, 9.344343185 + 5 - 5 * 0.999999, -1.924159646 - 5 * (-0.0001)]])
            topnode = 1e-3*np.array(
                [[0.001, 0.01, 0.0],
                 [-0.001, -0.01, 100.0]])
            # topnode = np.array(
            #     [[65.0 + 400 * 0.0001, -15. + 5. + 400 * 0.999999, 5. + 400 * (-0.0001)],
            #      [65.0, -15.0 + 5.0, 5.]])

        else:
            topnode = None
        self.model = Model(cuda_mesh, self.delta_h_time, self.gMaterial, tip_matrix, self.node_number, topnode,
                           path=None, mtype=mtype)
        # self.topnode_start = self.model.topnode.copy()

        # try:
        #     self.topnode = np.zeros((2, 3), dtype=np.float64)
        #     self.topnode[0] = np.transpose(1000 * tip_matrix[0:3, 3])
        #     self.topnode[1] = self.topnode[0] + 300 * np.transpose(tip_matrix[0:3, 0])
        # except:
        #     print('tip_matrix is None', tip_matrix)
        #     # 注意topnode的上下两端不能有参数一样，不能平行于xyz轴
        #     # self.topnode = np.array([[-24., -25., 31.], [88., 87., 87.]])
        #     # 10倍是厘米变成mm
        #     self.topnode = 10 * np.array([[-0.916783988 + 40 * 0.0001, 9.344343185 + 5 + 40 * 0.999999, -1.924159646 + 40 * (-0.0001)], [-0.916783988, 9.344343185 + 5, -1.924159646]])
        # node_num = 100
        # self.model.gNode, self.gElement = getnode(self.topnode, node_num, True)
        self.gBC1 = np.array([[0, 0, 0.0], [0, 1, 0.0], [0, 2, 0.0], [0, 3, 0.0], [0, 4, 0.0], [0, 5, 0.0]])
        # self.gBC1 = np.array([[0, 0, 0.0], [0, 2, 0.0], [0, 3, 0.0], [0, 4, 0.0], [0, 5, 0.0]])

        # self.model.ResultNode = np.zeros(np.shape(self.model.gNode), dtype=np.float64)
        # self.model.gNode = np.array(self.model.gNode, dtype=np.float64)
        # self.gElement = np.array(self.gElement, dtype=np.float64)
        self.f1 = np.zeros(self.node_number * 6, dtype=np.float64)
        self.gravity = np.zeros(self.node_number * 6, dtype=np.float64)
        self.gravity_copy = self.gravity.copy()
        # self.model.l1 = self.node_number - 1
        # T0 = TransformMatrix(self.model.gNode)
        # self.k0 = StiffnessMatrix(self.model.gNode, self.gMaterial)
        # self.model.gK = np.zeros((node_number * 6, node_number * 6), dtype=np.float64)
        self.U = 0
        # self.AssembleStiffnessMatrix(self.k0, T0, node_number)

        self.A1 = np.zeros([0, np.dot(6, np.shape(self.model.gNode)[0])], dtype=np.float64)
        self.b1 = np.zeros([0, 1], dtype=np.float64)

        self.X = MatrixStruct(self.x)
        self.X.send_to_gpu()
        self.Xtriangle = MatrixStruct(self.xtriangle)
        self.Xtriangle.send_to_gpu()
        self.adjacentTriangle = np.zeros((np.shape(self.xtriangle)[0], 10), dtype=np.float64)
        SearchAdjacentTriangle = mod.get_function("SearchAdjacentTriangle")
        self.AdjacentTriangle = MatrixStruct(self.adjacentTriangle)
        self.AdjacentTriangle.send_to_gpu()
        self.Adjx = np.zeros(np.shape(self.xtriangle)[0], dtype=np.int32)
        SearchAdjacentTriangle(self.Xtriangle.cptr, self.AdjacentTriangle.cptr, cuda.InOut(self.Adjx),
                               block=(1024, 1, 1),
                               grid=(self.xtriangle.shape[0], 1, 1))
        self.adjTri = self.AdjacentTriangle.d2h_row(self.adjacentTriangle.shape[0])
        self.SolveModel = mod.get_function("SolveModel")
        self.getConstraint = mod.get_function("getConstraint")
        self.getConstraint_x = mod.get_function("getConstraint_x")
        self.TransformMatrix = mod.get_function("TransformMatrix")
        self.skeletonlist = np.zeros([100, 3], dtype=np.float64)
        self.x = np.array(self.x, dtype=np.float64)
        self.xtriangle = np.array(self.xtriangle, dtype=np.int32)
        self.vNormal, self.distance0 = fa_vector(self.x, self.xtriangle, self.skeletonlist)
        # ----------------------------------------------------------------------------------------注意
        # self.vNormal = - self.vNormal

        self.VNormal = MatrixStruct(self.vNormal)
        self.Distance0 = MatrixStruct(self.distance0)
        self.VNormal.send_to_gpu()
        self.Distance0.send_to_gpu()

        start0 = [24.8284928, 4.0543492, -0.6314446]
        goal0 = [29.5939312, 18.0450401, 1.7955891]

        # self.path0 = pathplan11(self.skeletonlist, start0, goal0)
        # start0 = 2173
        # goal0 = 0
        self.path0 = None  # pathplan11(start0, goal0)

        [bc_number, dummy] = np.shape(self.gBC1)
        # bc_number初始约束数目，2个约束只能沿着轴向移动，6个约束跟随
        self.Aeq = np.zeros((bc_number, 6 * np.shape(self.model.gNode)[0]), dtype=np.float64)
        self.beq = np.zeros((bc_number, 1), dtype=np.float64)

        for ibc in range(bc_number):
            n = self.gBC1[ibc][0]
            d = self.gBC1[ibc][1]
            m = np.int64(n * 6 + d)
            # Aeq = np.row_stack((Aeq, np.zeros((1, 6 * np.shape(gNode)[0]))))
            self.Aeq[ibc][m] = 1
            # beq = np.row_stack((beq, np.zeros((1, 1))))
            self.beq[ibc] = self.gBC1[ibc][2]
        # for ieq in range(2):
        #     n = self.gBC1[0][0]
        #     m = np.int64((n - 1) * 6)
        #     self.Aeq[ieq + 6 + bc_number][m] = self.topnode[0, ieq + 1]-self.topnode[1, ieq + 1]
        #     self.Aeq[ieq + 6 + bc_number][m + ieq] = -(self.topnode[0, 0] - self.topnode[1, 0])
        #     self.beq[ieq + 6 + bc_number] = 0 # self.topnode[0, ieq + 1] * self.topnode[1, 0] - self.topnode[1, ieq + 1] * self.topnode[0, 0]

        # self.model.x1 = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)

        self.F = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)
        self.q = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)
        self.f2 = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)
        self.sigma_F = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)
        self.fc = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)

        self.t = 1
        # rotatebool = 0
        # self.rx, self.ry = (0, 0)
        # self.tx, self.ty = (0, 0)
        # self.zpos = 0
        # self.rotate = self.move = False
        # model start
        # self.gNode_start = self.model.gNode.copy()
        # self.model.gK_start = self.model.gK.copy()
        # self.model.x_start = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)

        # self.model.ResultNode = self.model.gNode.copy()
        # self.model.ResultNodedd = self.model.gNode.copy()

        # constrain start
        self.beq_start = self.beq.copy()
        self.Aeq_start = self.Aeq.copy()
        self.ff = 0.
        self.camulatebool = False
        self.set_locate = False
        # enviroment send to gpu
        self.constraint_triangle = np.zeros((1000, 1), dtype=np.float64)
        self.Constraint_triangle = MatrixStruct(self.constraint_triangle)
        self.Constraint_triangle.send_to_gpu()
        # constraint send to gpu
        self.A = np.zeros((1000, 6 * np.shape(self.model.gNode)[0]), dtype=np.float64)
        self.AA = MatrixStruct(self.A)
        self.AA.send_to_gpu()
        self.b = np.zeros((1000, 1), dtype=np.float64)
        self.B = MatrixStruct(self.b)
        self.B.send_to_gpu()
        self.a0a0 = np.zeros(1, dtype=np.int32)
        self.tttt = np.zeros(10, dtype=np.float32)
        # model send to gpu
        # self.Delta_h_time = MatrixStruct(np.array([self.delta_h_time]))
        # self.Delta_h_time.send_to_gpu()
        self.REsultNode = MatrixStruct(self.model.ResultNode * 1000)
        self.REsultNode.send_to_gpu()
        self.GNode = MatrixStruct(self.model.gNode)
        self.GNode.send_to_gpu()
        self.tT = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
        self.tT_zero = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
        self.T = MatrixStruct(self.tT)
        self.T.send_to_gpu()
        self.V = MatrixStruct(self.model.v * 1000)
        self.V.send_to_gpu()
        self.M_A = MatrixStruct(self.model.a * 1000)
        self.M_A.send_to_gpu()
        self.H = MatrixStruct(np.array([self.delta_h_time], dtype=np.float64))
        self.H.send_to_gpu()
        # result start
        # self.model.dv_copy = np.zeros_like(self.model.x)
        self.rotatebool = 0

        # some para
        self.na = 1
        self.la = 1
        self.sim_sheathinsert_copy = 0
        self.sim_tubeinsert_copy = 0
        self.forsum = 0
        self.alltime = 0
        self.Aeq_position = 0

        self.TransformMatrix(self.REsultNode.cptr, self.T.cptr, block=(12, 12, 1),
                             grid=(self.node_number - 1, 1, 1))  # cost 12ms，引入GPU后消除
        self.tT = self.T.d2h()
        # shared data to other process
        shared_draw['F'] = self.F
        shared_draw['constraint_triangle'] = self.constraint_triangle
        shared_draw['ResultNode'] = self.model.ResultNode
        shared_draw['l1'] = self.model.l1
        shared_draw['na'] = self.na
        shared_draw['la'] = self.la
        shared_draw['velocity_InsertionUnit'] = self.velocity_InsertionUnit
        # get camera para
        self.location_copy = shared_location['location'].copy()
        cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * 3.1415926535 / 2))
        camera_location = eulerAnglesToRotationMatrix(shared_location['location'][3:6], 'zyx') @ cy
        self.camera_location_copy = camera_location.copy()
        Quat = rotateToQuaternion(camera_location)
        self.Angles_copy = QuaternionToAxialAngle(Quat)
        # self.camera_location_copy_copy = self.camera_location_copy.copy()
        # self.Angles_copy_copy = self.Angles_copy.copy()
        self.d_locate = np.zeros(6, dtype=np.float64)

        # rt = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * 0.5))  # direction
        self.camera_T = self.tT[-1, 0:3, 0:3]  # @ rt  # self.camera_location_copy.copy()  #
        self.camera_T_start = self.camera_T.copy()
        self.camera_matrix = np.array(
            [self.model.ResultNode[-1][0], self.model.ResultNode[-1][1], self.model.ResultNode[-1][2],
             self.model.ResultNode[-1][0] + 10 * self.camera_T[0][0],
             self.model.ResultNode[-1][1] + 10 * self.camera_T[1][0],
             self.model.ResultNode[-1][2] + 10 * self.camera_T[2][0], 10 * self.camera_T[0][1],
             10 * self.camera_T[1][1], 10 * self.camera_T[2][1]])
        if Is_warm_start:
            self.model.warmstart()
            with open(os.getcwd() + "/data/run_data" + "/Is_warm_start.txt", "r") as f:
                for line in f:
                    data_line = line.strip("\n").split()
            with open(data_line[0], "rb") as f:
                self.camera_matrix = pickle.load(f)
                self.F = pickle.load(f)
                self.f1 = pickle.load(f)
        shared_draw['camera_matrix'] = self.camera_matrix
        shared_draw['camera_T_start'] = self.camera_T_start

    '''def init_tube(self, shared_draw):
        self.dirs_index = 0
        self.hGimbalAngle = np.zeros(2, dtype=np.float64)
        self.hGimbalAngle_copy = np.zeros(2, dtype=np.float64)
        node_num = 100
        self.model.gNode, self.gElement = getnode(self.topnode, node_num, True)
        gBC1 = np.array(
            [[node_num - 40, 4, 0.0],
             [node_num - 40, 5, 0.0], [node_num - 40, 6, 0.0]])
        self.gMaterial = np.array([[150.0, 0.01, 0.03, 1.010e-6]])
        [node_number, dummy] = np.shape(self.model.gNode)
        T0 = TransformMatrix(self.model.gNode)
        self.k0 = StiffnessMatrix(self.model.gNode, self.gMaterial)
        self.model.gK = np.zeros((node_number * 6, node_number * 6), dtype=np.float64)

        self.AssembleStiffnessMatrix(self.k0, T0, node_number)

        [bc_number, dummy] = np.shape(gBC1)
        self.Aeq = np.zeros((bc_number, 6 * np.shape(self.model.gNode)[0]), dtype=np.float64)
        self.beq = np.zeros((bc_number, 1), dtype=np.float64)

        for ibc in range(bc_number):
            n = gBC1[ibc][0]
            d = gBC1[ibc][1]
            m = np.int64((n - 1) * 6 + d - 1)
            self.Aeq[ibc][m] = 1
            self.beq[ibc] = gBC1[ibc][2]

        self.model.gNode_start = self.model.gNode.copy()
        self.beq_start = self.beq.copy()
        self.Aeq_start = self.Aeq.copy()
        # self.model.gK_start = self.model.gK.copy()
        # self.model.x_start = np.zeros(np.shape(self.model.gNode)[0] * 6, dtype=np.float64)
        self.ff = 0.
        self.rotatebool = 0
        self.camulatebool = True
        self.model.ResultNode = self.model.gNode.copy()
        self.set_locate = False

        self.REsultNode = MatrixStruct(self.model.ResultNode)
        self.REsultNode.send_to_gpu()
        self.gNode = MatrixStruct(self.model.gNode)
        self.gNode.send_to_gpu()
        self.a0a0 = np.zeros(1, dtype=np.int32)
        self.model.l1 = self.node_number - 1
        # self.model.dv_copy = np.zeros_like(self.model.x)

        self.na = 1
        self.la = 1
        self.sim_sheathinsert_copy = 0
        self.sim_tubeinsert_copy = 0

        self.forsum = 0
        self.alltime = 0
        self.Aeq_position = 0
        self.tT = np.zeros((node_number - 1, 12, 12))
        self.T = MatrixStruct(self.tT)
        self.T.send_to_gpu()
        self.camulatebool = False

        self.TransformMatrix(self.model.REsultNode._cptr, self.T._cptr, block=(12, 12, 1),
                             grid=(self.node_number - 1, 1, 1))  # cost 12ms，引入GPU后消除
        self.tT = self.T.get_from_gpu3()

        # rt = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * 0.5))  # direction
        self.camera_T = self.tT[-1, 0:3, 0:3]  # @ rt
        self.camera_T_start = self.camera_T.copy()

        self.camera_matrix = np.array([self.model.ResultNode[-1][0], self.model.ResultNode[-1][1], self.model.ResultNode[-1][2],
                                       self.model.ResultNode[-1][0] + 10 * self.camera_T[0][0],
                                       self.model.ResultNode[-1][1] + 10 * self.camera_T[1][0],
                                       self.model.ResultNode[-1][2] + 10 * self.camera_T[2][0], self.camera_T[0][2],
                                       self.camera_T[1][2], self.camera_T[2][2]])

        shared_draw['F'] = self.F
        shared_draw['constraint_triangle'] = self.constraint_triangle
        shared_draw['ResultNode'] = self.model.ResultNode
        shared_draw['l1'] = self.model.l1
        shared_draw['na'] = self.na
        shared_draw['la'] = self.la
        shared_draw['velocity_InsertionUnit'] = self.velocity_InsertionUnit
        shared_draw['camera_matrix'] = self.camera_matrix
        shared_draw['camera_T_start'] = self.camera_T_start'''

    def RunFEM(self, shared_mem, shared_fps, velocity_sheath, shared_draw, ctx, shared_udp, shared_location, shared_RL,
               shared_state):
        if True:  # self.rotatebool == 0:
            if True:  # self.camulatebool:
                # self.t += 1
                # 重新输入单元位置和速度到GPU
                if self.give_c:
                    if 0 <= self.run_time < 2.5:
                        self.beq[0] = 40 * self.delta_h_time * 1e-3  # 0.04 * self.run_time
                    if 2.5 <= self.run_time < 5:
                        self.beq[0] = -40 * self.delta_h_time * 1e-3  # 0.1 - 0.04 * (self.run_time - 2.5)
                    if 5 <= self.run_time < 7.5:
                        self.beq[0] = 40 * self.delta_h_time * 1e-3  # 0.04 * (self.run_time - 5)
                    if 7.5 <= self.run_time <= 10:
                        self.beq[0] = -40 * self.delta_h_time * 1e-3  # 0.1 - 0.04 * (self.run_time - 7.5)


                    '''# 球大小变化
                    # 定义半径随时间的改变逻辑（例如：利用正弦函数让半径在 20mm 到 40mm 之间平滑呼吸变化）
                    current_radius = 30.0 + 10.0 * np.sin(self.run_time * 2.0)
                    # 计算当前帧的缩放比例
                    self.scaled_factor = current_radius / 40.0
                    # 核心：更新网格的顶点位置
                    # 逻辑为：先将顶点平移到以球心为原点的位置 -> 应用缩放比例 -> 再平移回原来的球心
                    self.x = self.center + (self.x_copy - self.center) * self.scaled_factor
                    self.vNormal, self.distance0 = fa_vector(self.x, self.xtriangle, self.skeletonlist)
                    self.X.rehtod(self.x)
                    self.Distance0.rehtod(self.distance0)'''

                    self.run_time += self.delta_h_time
                self.tttt[0] = time.time()
                ttttt = time.time()
                self.REsultNode.rehtod(self.model.ResultNode * 1000)
                self.V.rehtod(self.model.v * 1000)
                self.M_A.rehtod(self.model.a * 1000)
                self.a0a0[0] = 0
                # 计算碰撞约束
                '''self.SolveModel(self.REsultNode._cptr, self.X._cptr, self.Xtriangle._cptr, self.REsultNode._cptr,
                                self.VNormal._cptr, self.Distance0._cptr,
                                self.Constraint_triangle._cptr, cuda.InOut(self.a0a0), self.AA._cptr, self.B._cptr,
                                self.AdjacentTriangle._cptr,
                                cuda.In(self.Adjx), cuda.In(np.array([self.delta_h_time])), block=(self.model.gNode.shape[0], 1, 1),
                                grid=(self.xtriangle.shape[0], 1, 1))'''
                # self.getConstraint以加速度为变量，推导根据newmark和HHT方法
                '''self.getConstraint(cuda.InOut(self.a0a0), self.REsultNode.cptr, self.V.cptr, self.M_A.cptr, self.X.cptr, self.Xtriangle.cptr,
                                self.VNormal.cptr, self.Distance0.cptr,
                                self.Constraint_triangle.cptr, self.AA.cptr, self.B.cptr,
                                cuda.In(self.Adjx), self.H.cptr, block=(self.model.gNode.shape[0], 1, 1),
                                grid=(self.xtriangle.shape[0], 1, 1))'''
                # self.getConstraint_x以位移变化为变量，推导根据newmark和HHT方法
                self.getConstraint_x(cuda.InOut(self.a0a0), self.REsultNode.cptr, self.V.cptr, self.M_A.cptr,
                                     self.X.cptr,
                                     self.Xtriangle.cptr,
                                     self.VNormal.cptr, self.Distance0.cptr,
                                     self.Constraint_triangle.cptr, self.AA.cptr, self.B.cptr,
                                     cuda.In(self.Adjx), self.H.cptr, block=(self.model.gNode.shape[0], 1, 1),
                                     grid=(self.xtriangle.shape[0], 1, 1))
                # 计算新的旋转矩阵
                # self.TransformMatrix(self.REsultNode.cptr, self.T.cptr, block=(12, 12, 1),
                #                      grid=(self.node_number - 1, 1, 1))  # cost 12ms，引入GPU后消除
                # T为局部坐标转全局坐标
                # self.tT = self.T.d2h()
                # 获取新的约束
                self.constraint_triangle = self.Constraint_triangle.d2h_row(self.a0a0[0])
                self.A = self.AA.d2h_row(self.a0a0[0])
                self.b = self.B.d2h_row(self.a0a0[0]) *1e-3
                # for m in range(self.A.shape[0]):
                #     for n in range(self.A.shape[1]):
                #         if self.A[m, n] != 0:
                #             print("A is not zero", n)
                #             print(self.A[m, n])
                #     print(self.b[m])

                self.tttt[1] = time.time()
                print('collision detect', time.time() - ttttt)

                # print(self.a0a0)
                # print(self.x.shape, self.xtriangle.shape)
                # print(self.x[self.xtriangle[3, 0]], self.x[self.xtriangle[3, 1]], self.x[self.xtriangle[3, 2]])
                # 重新计算全局刚度矩阵
                # k = resetgK(self.k0, self.model.l1)
                # k = self.model.k0
                # self.model.AssembleStiffnessMatrix(k, self.tT, self.node_number)
                # gk0 = self.model.gK.copy()
                # x0 = self.model.x.copy()
                self.model.AssembleStiffnessMatrix_corotate()
                self.tttt[2] = time.time()
                # self.F = np.dot(self.model.gK, self.model.x + self.model.v * self.delta_h_time)
                # 求解位移变化,每个节点六个维度，前三个为位移，后三个为轴角，轴代表旋转轴，模长代表旋转角度
                # print(self.model.M.toarray())
                #  - np.dot(self.model.M.toarray(), self.model.dv)
                # D = 0.009 *self.model.M.toarray() + 0.002 * self.model.gK
                # self.model.gK = (self.model.gK + self.model.gK.transpose())*0.5

                # print(np.linalg.matrix_rank(G))
                # P = G.copy()
                # T = np.transpose(G).copy()
                # for i in range(G.shape[0]):
                #     G[:i, i] = 0
                # print(G)
                # print('nambla', np.linalg.eigvals(G))
                # print(np.linalg.det(self.model.gK))
                # print(np.linalg.det(self.model.M))
                # print(self.model.M)

                # D = 0.1 * self.model.M + 0.01 * self.model.gK0  # 把这个Ｋ加进去以后，模型就出问题，阻尼矩阵似乎不能学瑞丽阻尼设置，但是实验时可以，莫名修复了
                D = 0.1 * self.model.M + 0.1 * self.model.gK
                # D = 0.0 * self.model.M + 0.0 * self.model.gK
                # 隐式积分方法
                # implicit_integration(self.model, self.model.M, D, self.model.gK, self.F, self.f1, self.gravity, self.A, self.b, self.Aeq, self.beq, self.delta_h_time)
                # self.F += self.delta_h_time * np.dot(self.model.gK, self.model.v) + np.dot(D, self.model.dv)
                # 隐式alpha积分方法
                # alpha = implicit_alpha_integration(self.model, self.model.M, D, self.model.gK, self.F, self.f1, self.gravity, self.A, self.b, self.Aeq, self.beq, self.delta_h_time)
                # self.F += (1+alpha)*self.delta_h_time * np.dot(self.model.gK, self.model.v) + np.dot(D, self.model.dv)
                # newmark-beta方法
                # newmark_beta_integration(self.model, self.model.M, D, self.model.gK, self.F, self.f1, self.gravity, self.A, self.b, self.Aeq, self.beq, self.delta_h_time)
                # self.F += np.dot(self.model.gK, self.model.v * self.delta_h_time + (self.model.a * 0.5 * self.delta_h_time ** 2))
                # newmark-beta(更新x)方法
                # beta = newmark_beta_integration_x(self.model, self.model.M, D, self.model.gK, self.F, self.f1, self.gravity,
                #                          self.A, self.b, self.Aeq, self.beq, self.delta_h_time)
                # self.f2 = np.dot(self.model.gK, self.model.x)
                # self.model.a = solve_qp(sp.csc_matrix(np.dot(np.transpose(self.model.M), self.model.M)),
                #                   np.dot(np.transpose(self.model.M), self.f2 - self.f1 - self.gravity),
                #                   sp.csc_matrix(self.A),
                #                   self.b.flatten(),
                #                   sp.csc_matrix(self.Aeq),
                #                   self.beq.flatten(), solver='osqp')
                # self.F = - np.dot(self.model.M, self.model.x /(self.delta_h_time**2 * beta) + self.model.v / (self.delta_h_time * beta) + self.model.a * (0.5 / beta - 1))
                # HHT-alpha方法
                # alpha = hht_alpha_integration(self.model, self.model.M, D, self.model.gK, self.F, self.f1, self.gravity, self.A, self.b, self.Aeq, self.beq, self.delta_h_time)
                # self.F += (1 + alpha) * np.dot(self.model.gK, self.model.v * self.delta_h_time + (self.model.a * 0.5 * self.delta_h_time ** 2))
                # HHT-alpha-x方法,用dx作为变量
                # self.Aeq = np.zeros((1, 600))
                # self.beq = np.array([0])
                '''alpha, beta, gamma, self.y, self.tttt = hht_alpha_integration_x(self.model, self.model.M, D, self.model.gK, self.F,
                                                             self.f1, self.gravity,
                                                             self.A, self.b, self.Aeq, self.beq, self.delta_h_time,
                                                             self.q, self.fc, self.tttt)
                self.q = np.dot(self.model.M, -1 / beta / self.delta_h_time * self.model.v - (
                            0.5 / beta - 1) * self.model.a) + np.dot(D,
                                                                     (1 - gamma / beta * (1 + alpha)) * self.model.v + (
                                                                                 1 + alpha) * (
                                                                                 1 - gamma / 2 / beta) * self.delta_h_time * self.model.a)
                self.F += np.dot(self.model.gK, self.model.dx)'''
                # self.fc = alpha / (1+alpha) * np.dot(self.A.transpose(), self.y)


                # nnqp-HHT-alpha-x方法,用dx作为变量
                alpha, beta, gamma, self.y, self.tttt = nnqp_hht(self.model, self.model.M, D,
                                                                 self.model.gK, self.F,
                                                                 self.f1, self.gravity,
                                                                 self.A, self.b, self.Aeq, self.beq,
                                                                 self.delta_h_time,
                                                                 self.q, self.fc, self.tttt, self.alpha)
                self.q = np.dot(self.model.M, -1 / beta / self.delta_h_time * self.model.v - (
                        0.5 / beta - 1) * self.model.a) + np.dot(D,
                                                                 (1 - gamma / beta * (1 + alpha)) * self.model.v + (
                                                                         1 + alpha) * (
                                                                         1 - gamma / 2 / beta) * self.delta_h_time * self.model.a)
                # self.F += np.dot(self.model.gK, self.model.dx)
                self.F = self.model.AssembleInternalForce_corotate()
                # self.fc = alpha / (1 + alpha) * np.dot(self.A.transpose(), self.y[:self.A.shape[0]])

                # ==========================================================
                # 本帧求解开始 (假设 self.q 已经基于 t_n 的 v 和 a 计算完毕)
                # ==========================================================
                '''beta = 0.25 * (1 - self.alpha) ** 2
                gamma = 0.5 * (1 - 2 * self.alpha)
                self.q = np.dot(self.model.M, -1 / beta / self.delta_h_time * self.model.v - (
                        0.5 / beta - 1) * self.model.a) + np.dot(D,
                                                                 (1 - gamma / beta * (1 + self.alpha)) * self.model.v + (
                                                                         1 + self.alpha) * (
                                                                         1 - gamma / 2 / beta) * self.delta_h_time * self.model.a)
                # 1. 【存档】 备份第 n 帧的干净状态
                self.model.backup_state()

                # 2. 【预测阶段 (Predictor)】
                self.F = self.model.AssembleInternalForce_corotate()
                gK_predictor = self.model.gK.copy()

                # 第一次试探求解 (求出碰撞表面的位置)
                # ！！！注意：这里传入的 fc 必须是 0 ！！！
                alpha, beta, gamma, self.y, self.tttt = nnqp_hht(self.model, self.model.M, D,
                                                                 gK_predictor, self.F,
                                                                 self.f1, self.gravity,
                                                                 self.A, self.b, self.Aeq, self.beq,
                                                                 self.delta_h_time,
                                                                 self.q, 0, self.tttt, self.alpha)

                # 提取预测的完整位移增量
                dx_predict = self.model.dx.copy()

                # 3. 【提取高度非线性的真实力学反馈】
                # 在被接触面截停的弯曲状态下，重新计算极其精确的总内力和切线刚度
                F_corrector = self.model.AssembleInternalForce_corotate()
                self.model.AssembleStiffnessMatrix_corotate()
                gK_corrector = self.model.gK.copy()

                # ---------------------------------------------------------
                # 核心数学修正：构造 Pseudo Force
                # 将非线性残差映射回 tn 时刻，欺骗求解器使其等价于牛顿迭代
                # ---------------------------------------------------------
                F_pseudo = F_corrector - np.dot(gK_corrector, dx_predict)

                # 4. 【读档 (时光倒流)】
                self.model.restore_state()

                # 5. 【校正阶段 (Corrector)】
                # 换上精准的 F_pseudo 和 gK_corrector，进行真正的求解
                alpha, beta, gamma, self.y, self.tttt = nnqp_hht(self.model, self.model.M, D,
                                                                 gK_corrector, F_pseudo,
                                                                 self.f1, self.gravity,
                                                                 self.A, self.b, self.Aeq, self.beq,
                                                                 self.delta_h_time,
                                                                 self.q, 0, self.tttt, self.alpha)'''
                # ==========================================================



                # 动态方法，python osqp直接迭代，dx作为变量
                '''alpha, beta, gamma, self.y = py_osqp_dynamic_x(self.model, self.model.M, D, self.model.gK, self.F,
                                                               self.f1, self.gravity,
                                                               self.A, self.b, self.Aeq, self.beq,
                                                               self.delta_h_time, self.q, self.fc, self.alpha)
                self.alpha = alpha
                self.q = np.dot(self.model.M, -1 / beta / self.delta_h_time * self.model.v - (
                        0.5 / beta - 1) * self.model.a) + np.dot(D,
                                                                 (1 - gamma / beta * (1 + alpha)) * self.model.v + (
                                                                         1 + alpha) * (
                                                                         1 - gamma / 2 / beta) * self.delta_h_time * self.model.a)
                self.F += np.dot(self.model.gK, self.model.dx)
                self.fc = alpha * (np.dot(self.A.transpose(), self.y[:self.A.shape[0]]) + np.dot(self.Aeq.transpose(),
                                                                                                 self.y[
                                                                                                 self.A.shape[0]:]))'''

                self.tttt[5] = time.time()
                # print('A.shape:', self.A.shape[0])
                # print('contact force y(inequality variable is in the first):', self.y)
                print('time_of_getCons', self.tttt[1] - self.tttt[0])
                print('time_of_solve', self.tttt[3] - self.tttt[2])
                self.compute_time.append(self.tttt[5] - self.tttt[0])
                self.Nb_ave_constraint.append(self.y.shape[0] - 6)
                self.time_of_getCons.append(self.tttt[1] - self.tttt[0])
                self.time_of_getK.append(self.tttt[2] - self.tttt[1])
                self.time_of_update.append(self.tttt[5] - self.tttt[3])
                self.time_of_solve.append(self.tttt[3] - self.tttt[2])

                # 隐式beta积分方法
                # beta = implicit_beta_integration(self.model, self.model.M, D, self.model.gK, self.F, self.f1, self.gravity, self.A, self.b, self.Aeq, self.beq, self.delta_h_time)
                # self.F += (1+beta) * np.dot(self.model.gK, self.model.x + self.delta_h_time * self.model.v) - beta * np.dot(gk0, x0) + np.dot(D, self.model.dv)

                # 求解可能无解， 作一极小刚体位移
                # if dv is None:
                #     dv = self.model.dv_copy.copy() + 0.000001
                #     print('warn:OSQP can not be solved!')
                # if np.linalg.norm(dv) > 155:
                #     print(np.linalg.norm(dv))
                #     print('The solved result is too large!')
                #     dv *= 0.001

                self.la = 1.0

                # self.beq[0:3] = 0.
                ct = linalg.expm(np.cross(np.eye(3), self.model.x[-3:]))
                self.camera_T = ct @ self.camera_T_start

                # [node_number, dummy] = np.shape(gNode)

                self.camera_matrix = np.array(
                    [self.model.ResultNode[-1][0], self.model.ResultNode[-1][1], self.model.ResultNode[-1][2],
                     self.model.ResultNode[-1][0] + 10 * self.camera_T[0][0],
                     self.model.ResultNode[-1][1] + 10 * self.camera_T[1][0],
                     self.model.ResultNode[-1][2] + 10 * self.camera_T[2][0], 10 * self.camera_T[0][1],
                     10 * self.camera_T[1][1], 10 * self.camera_T[2][1]])
                # print(self.camera_T)
                # self.camera_matrix = np.array([self.model.ResultNode[-1][0], self.model.ResultNode[-1][1], self.model.ResultNode[-1][2],
                #                                2 * self.model.ResultNode[-1][0] - self.model.ResultNode[-2][0],
                #                                2 * self.model.ResultNode[-1][1] - self.model.ResultNode[-2][1],
                #                                2 * self.model.ResultNode[-1][2] - self.model.ResultNode[-2][2], self.camera_T[0][2],
                #                                self.camera_T[1][2], self.camera_T[2][2]])
                self.shared_camera_matrix['camera_matrix'] = self.camera_matrix.copy()
                if self.is_save_data:
                    path_root = os.getcwd()
                    localtime = time.localtime()
                    time_now = time.strftime("/%Y-%m-%d/h%H/%H-%M-%S", localtime)
                    if not os.path.exists(path_root + "/data/run_data" + time_now):
                        os.makedirs(path_root + "/data/run_data" + time_now)
                        self.dirs_index = 0
                    self.data_time_stream = path_root + "/data/run_data" + time_now + "/%d.pickle" % self.dirs_index
                    with open(self.data_time_stream, "wb") as f:
                        pickle.dump(self.model.ResultNode, f)
                        pickle.dump(self.model.l1, f)
                        pickle.dump(self.camera_matrix, f)
                        pickle.dump(self.F, f)
                        pickle.dump(self.f1, f)
                        pickle.dump(self.model.x, f)
                        self.dirs_index += 1
                fps = 1.0 / (time.time() - ttttt)

                print('fps', fps)
                shared_draw['fps'] = fps

                if self.save_data_CMAME and self.run_time <= 5.0 and self.give_c:
                    min_length = 10000
                    for i in range(self.model.ResultNode.shape[0]):
                        length1 = np.linalg.norm(self.model.ResultNode[i]*1000 - self.center)
                        if length1 < min_length:
                            min_length = length1
                            nearest_node_index = i
                    self.CMAME_data_exp2.append([self.run_time, min_length])
                    self.baseline_exp2.append([self.run_time, self.scaled_factor*40.0])

                    path_root = os.getcwd()
                    path_CMAME_data = path_root + "/data/CMAME2_dyn"
                    if not os.path.exists(path_CMAME_data):
                        os.makedirs(path_CMAME_data)

                    data_time_stream = path_CMAME_data + "/N%d.pickle" % self.node_number
                    with open(data_time_stream, "wb") as f:
                        pickle.dump(self.CMAME_data_exp2, f)

                    # baseline_stream = path_CMAME_data + "/baseline.pickle"
                    # with open(baseline_stream, "wb") as f:
                    #     pickle.dump(self.baseline_exp2, f)



        self.U += 0.5 * np.dot(self.model.dx.transpose(), np.dot(self.model.gK, self.model.dx)) + np.dot(self.F,
                                                                                                         self.model.dx)
        dE = 0.5 * np.dot(self.model.v.transpose(), np.dot(self.model.M, self.model.v)) + self.U
        print('dE:', dE)
        shared_state['dE'] = [dE]
        shared_state['x'] = [self.model.x[-5]]
        shared_state['v'] = [self.model.v[-5]]
        shared_state['a'] = [self.model.a[-5]]
        shared_state['t'] = [self.run_time]

        self.rotatebool = 1
        self.camulatebool = False
        datafloat = shared_udp['datafloat']
        self.hGimbalAngle[0] = datafloat[0]  # 104 - 158
        self.hGimbalAngle[1] = datafloat[1]  # -79 - -132
        steermode = int(datafloat[3])  # 新版本里面该项由4变成了3
        sim_sheathinsert = datafloat[2]
        sim_tubeinsert = 0  # datafloat[3] # 新版本里面暂时将插入部的插入距离去掉了

        dx = (sim_tubeinsert - self.sim_tubeinsert_copy)
        dl = (sim_sheathinsert - self.sim_sheathinsert_copy)

        self.sim_tubeinsert_copy = sim_tubeinsert
        self.sim_sheathinsert_copy = sim_sheathinsert
        if abs(dx) > abs(dl):
            len_element = np.linalg.norm(self.model.ResultNode[1] - self.model.ResultNode[0])
            mbeq = dx / len_element * (self.model.ResultNode[1] - self.model.ResultNode[0])  # 此处参数-1 可调插入部抽出速度
            self.beq[0] = mbeq[0]
            self.beq[1] = mbeq[1]
            self.beq[2] = mbeq[2]

            # if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
            #     for i in range(self.Aeq.shape[1] - 6):
            #         self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
            #     self.Aeq_position -= 1
            #     self.Aeq[0:6, 0:6] = 0

            self.model.set_l1(self.model.l1 - dx / len_element)
            if self.model.l1 < 0:
                self.model.set_l1(0)
            if self.model.l1 > self.node_number - 1:
                self.model.set_l1(self.node_number - 1)

            # if self.model.l1 < self.model.gNode.shape[0] - 1 and dx < 0:
            #     self.na += 1
            # if self.model.l1 > self.model.gNode.shape[0] - 20 and dx > 0:
            #     self.na -= 1

            self.rotatebool = 0
            self.camulatebool = True

        elif abs(dx) < abs(dl):
            len_element = np.linalg.norm(self.model.ResultNode[1] - self.model.ResultNode[0])
            mbeq = dl / len_element * (self.model.ResultNode[1] - self.model.ResultNode[0])  # 参数-1调整体抽出速度
            self.beq[0] = mbeq[0]
            self.beq[1] = mbeq[1]
            self.beq[2] = mbeq[2]
            # if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
            #     for i in range(self.Aeq.shape[1] - 6):
            #         self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
            #     self.Aeq_position -= 1
            #     self.Aeq[0:6, 0:6] = 0
            self.rotatebool = 0
            self.camulatebool = True
        # print(sim_sheathinsert, sim_tubeinsert, hGimbalAngle, steermode)
        tube_bend = 0.1
        max_velocity = 1.0
        # if sim_tubeinsert > 10:
        #     tube_bend = 0.002
        # elif sim_tubeinsert > 20:
        #     tube_bend = 0.003

        delta_hGimbalAngle = self.hGimbalAngle - self.hGimbalAngle_copy

        if delta_hGimbalAngle[0] > max_velocity:
            delta_hGimbalAngle[0] = max_velocity
            self.hGimbalAngle_copy[0] += max_velocity
        elif delta_hGimbalAngle[0] < -max_velocity:
            delta_hGimbalAngle[0] = -max_velocity
            self.hGimbalAngle_copy[0] -= max_velocity
        else:
            self.hGimbalAngle_copy[0] = self.hGimbalAngle[0]

        if delta_hGimbalAngle[1] > max_velocity:
            delta_hGimbalAngle[1] = max_velocity
            self.hGimbalAngle_copy[1] += max_velocity
        elif delta_hGimbalAngle[1] < -max_velocity:
            delta_hGimbalAngle[1] = -max_velocity
            self.hGimbalAngle_copy[1] -= max_velocity
        else:
            self.hGimbalAngle_copy[1] = self.hGimbalAngle[1]

        if steermode == 1:
            # self.f1[-3:] -= 0.001 * np.transpose(self.camera_T[0:3, 1])  # 此处参数调插入部转向速度
            self.f1[-3:] += 1 * delta_hGimbalAngle[0] * tube_bend * self.camera_T_start[0:3, 1] + 1 * \
                            delta_hGimbalAngle[
                                1] * tube_bend * self.camera_T_start[0:3, 2]  # 此处参数调插入部转向速度

            self.rotatebool = 0
            self.camulatebool = True

        if steermode == 2:
            # self.f1[-3:] -= 0.001 * np.transpose(self.camera_T[0:3, 1])  # 此处参数调插入部转向速度
            self.f1[int(self.model.l1) * 6 + 3:int(self.model.l1) * 6 + 6] += 0.05 * delta_hGimbalAngle[
                0] * 0.0015 * self.camera_T[0:3,
                              1] + \
                                                                              0.05 * delta_hGimbalAngle[
                                                                                  1] * 0.0015 * self.camera_T[0:3,
                                                                                                2]  # 此处参数调鞘转向速度+

            self.rotatebool = 0
            self.camulatebool = True
        # ----------------------------------------------------  用末端点定位作为约束，达到跟随的效果
        # 注意 在四元数转轴角时使得轴角不能再超过360度，未来在转动角度超过360度时可能会出现bug
        # 注意 两个轴角之间的差不代表他们之间的旋转过程，有更复杂的表示，除非同轴才能直接相减
        # loc = shared_location['location']
        # cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * 3.1415926535 / 2))
        # camera_location = eulerAnglesToRotationMatrix(loc[3:6], 'zyx') @ cy
        # camera_location_delta = camera_location @ np.transpose(self.camera_T)  #(self.camera_T)  # (self.camera_location_copy)
        # Quat = rotateToQuaternion(camera_location_delta)
        # Angles = QuaternionToAxialAngle(Quat)
        # for ieq in range(6):
        #     self.Aeq[ieq + self.gBC1.shape[0], self.node_number*6 - 6 + ieq] = 1
        #     if ieq < 3:
        #         self.beq[ieq + self.gBC1.shape[0]] = loc[ieq] - self.model.ResultNode[-1][ieq]  # self.location_copy[ieq]  # self.model.ResultNode[-1][ieq]
        #     else:
        #         self.beq[ieq + self.gBC1.shape[0]] = Angles[ieq-3]
        # self.location_copy = loc.copy()
        # self.Angles_copy = Angles.copy()
        # self.camera_location_copy = camera_location.copy()
        # self.rotatebool = 0
        # self.camulatebool = True
        # -----------------------------------------------------------------------------------

        # --------------------------------------------------用末端点与定位点之间的差作为受力输入
        #         loc = shared_location['location']
        #         cy = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * 3.1415926535 / 2))
        #         camera_location = eulerAnglesToRotationMatrix(loc[3:6], 'zyx') @ cy
        #         camera_location_delta = camera_location @ np.transpose(self.camera_T)
        #         Quat = rotateToQuaternion(camera_location_delta)
        #         Angles = QuaternionToAxialAngle(Quat)
        #         position_locate = loc[0:3] - self.model.ResultNode[-1]
        #         if np.linalg.norm(position_locate) != 0 or np.linalg.norm(Angles) != 0:
        #             position_loc = position_locate @ self.camera_T
        #             len_element = np.linalg.norm(self.model.ResultNode[1] - self.model.ResultNode[0])
        #             mbeq = position_loc[0] / len_element * (self.model.ResultNode[1] - self.model.ResultNode[0])  # 参数-1调整体抽出速度
        #             self.beq[0] = mbeq[0]
        #             self.beq[1] = mbeq[1]
        #             self.beq[2] = mbeq[2]
        #             self.rotatebool = 0
        #             self.camulatebool = True
        #             self.f1[-12:] += 0.05 * np.dot(self.model.gK[-12:, -3:], Angles.T)
        #             self.f1[-3:] += 0.01 * position_loc[1] * self.camera_T[0:3, 2] - 0.01 * position_loc[2]* self.camera_T[0:3, 1]
        # ------------------------------------------------------------------------------------
        #         self.gravity = self.gravity_copy.copy()
        action = shared_RL['action']
        if shared_draw['event'][2] == 5 or action == 'q':  # e.type == pygame.QUIT:
            shared_state['x'] = None
            sys.exit()
        elif shared_draw['event'][0] == 1 or action:  # e.type == pygame.KEYDOWN:
            if shared_draw['event'][1] == 1 or action == 'q':  # e.key == pygame.K_ESCAPE:
                shared_state['x'] = None
                print('self.compute_time:', np.mean(self.compute_time))
                print('Nb_ave_constraint:', np.mean(self.Nb_ave_constraint))
                print('self.time_of_getCons:', np.mean(self.time_of_getCons))
                print('self.time_of_getK:', np.mean(self.time_of_getK))
                print('self.time_of_update:', np.mean(self.time_of_update))
                print('self.time_of_solve:', np.mean(self.time_of_solve))

                sys.exit()
            if shared_draw['event'][1] == 2 or action == '1':  # e.key == pygame.K_1:
                self.model.resetmodel()
                self.hGimbalAngle = np.zeros(2, dtype=np.float64)
                self.hGimbalAngle_copy = np.zeros(2, dtype=np.float64)
                self.ff = 0.
                # self.model.gNode = self.model.gNode_start.copy()
                self.beq = self.beq_start.copy()
                self.Aeq = self.Aeq_start.copy()
                # self.model.gK = self.model.gK_start.copy()
                # self.model.x = self.model.x_start.copy()
                self.f1 = np.zeros(self.node_number * 6, dtype=np.float64)
                self.F = np.zeros(self.node_number * 6, dtype=np.float64)
                self.sigma_F = np.zeros(self.node_number * 6, dtype=np.float64)
                self.U = 0
                self.rotatebool = 0
                self.camulatebool = True
                # self.model.l1 = self.node_number - 1
                self.na = 0
                # self.topnode = self.topnode_start.copy()
                self.forsum = 0
                self.alltime = 0
                self.Aeq_position = 0
                self.camera_T = self.camera_T_start.copy()

            elif not shared_draw['set_locate']:
                if shared_draw['event'][1] == 3:  # e.key == pygame.K_2:
                    shared_draw['set_locate'] = 1
                    self.set_locate = True
                    self.hGimbalAngle = np.zeros(2, dtype=np.float64)
                    self.hGimbalAngle_copy = np.zeros(2, dtype=np.float64)
                    self.ff = 0.
                    self.model.gNode = self.model.gNode_start.copy()
                    self.beq = self.beq_start.copy()
                    self.Aeq = self.Aeq_start.copy()
                    self.model.gK = self.model.gK_start.copy()
                    # self.model.x = self.model.x_start.copy()
                    self.f1 = np.zeros(self.node_number * 6, dtype=np.float64)
                    self.F = np.zeros(self.node_number * 6, dtype=np.float64)
                    self.sigma_F = np.zeros(self.node_number * 6, dtype=np.float64)
                    self.U = 0
                    self.rotatebool = 0
                    self.camulatebool = True
                    self.model.l1 = self.node_number - 1
                    self.na = 0

                    self.forsum = 0
                    self.alltime = 0
                    self.Aeq_position = 0
                    self.camera_T = self.camera_T_start.copy()

                elif shared_draw['event'][1] == 4 or action == 'w':  # e.key == pygame.K_w:
                    # 如果怎么调都有维度不对，上面有刚度可以调self.gMaterial
                    velocity_insert = 0.01
                    mbeq = -velocity_insert * (self.model.ResultNode[1] - self.model.ResultNode[0])  # 此处参数-1 可调插入部抽出速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
                        for i in range(self.Aeq.shape[1] - 6):
                            self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
                        self.Aeq_position -= 1
                        self.Aeq[0:6, 0:6] = 0
                    self.model.set_l1(self.model.l1 + velocity_insert)
                    if self.model.l1 < 0:
                        self.model.set_l1(0)
                    if self.model.l1 > self.node_number - 1:
                        self.model.set_l1(self.node_number - 1)
                    # self.f1[1] += 0.01 * self.node_number / 100
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 5 or action == 's':  # e.key == pygame.K_s:
                    velocity_insert = 0.01
                    mbeq = velocity_insert * (self.model.ResultNode[1] - self.model.ResultNode[0])  # 此处参数1 可调插入部插入速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    if self.Aeq[0, 0] == 0 and self.Aeq_position >= 1:
                        for i in range(self.Aeq.shape[1] - 6):
                            self.Aeq[0:6, i] = self.Aeq[0:6, i + 6]
                        self.Aeq_position -= 1
                        self.Aeq[0:6, -6:] = 0
                    self.model.set_l1(self.model.l1 - velocity_insert)
                    if self.model.l1 < 0:
                        self.model.set_l1(0)
                    if self.model.l1 > self.node_number - 1:
                        self.model.set_l1(self.node_number - 1)
                    # self.f1[1] -= 0.01 * self.node_number / 100
                    self.rotatebool = 0
                    self.t1 = True
                    self.camulatebool = True

                elif shared_draw['event'][1] == 6:  # e.key == pygame.K_z:
                    self.f1[-3:] -= 0.001 * np.transpose(self.camera_T[0:3, 0])  # 此处参数调插入部转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 7:  # e.key == pygame.K_x:
                    xx = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * 0.01))
                    self.camera_T = self.camera_T @ xx  # 此处参数调插入部转向速度
                    self.camera_T_start = self.camera_T.copy()
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 8 or action == 'a':  # e.key == pygame.K_a:
                    self.f1[-3:] -= 0.1 * self.node_number / 100 * np.transpose(
                        self.camera_T_start[0:3, 1])  # 此处参数调插入部转向速度

                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 9 or action == 'd':  # e.key == pygame.K_d:
                    # self.f1[-3:] += 0.2 * 0.001 * np.transpose(self.camera_T[0:3, 1])  # 此处参数调插入部转向速度
                    # self.f1[-3:] += 0.1 * np.transpose(self.camera_T_start[0:3, 1])  # 此处参数调插入部转向速度
                    self.f1[-6:-3] += 0.01 * self.node_number / 100 * np.transpose(self.camera_T_start[0:3, 2])
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 10:  # e.key == pygame.K_q:
                    self.f1[-3:] += 0.1 * np.transpose(self.camera_T_start[0:3, 2])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True
                elif shared_draw['event'][1] == 11:  # e.key == pygame.K_e:
                    self.f1[-3:] -= 0.1 * np.transpose(self.camera_T_start[0:3, 2])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 12 or action == 'UP':  # e.key == pygame.K_UP:
                    mbeq = -10 * self.node_number / 100 * (
                                self.model.ResultNode[1] - self.model.ResultNode[0])  # 参数-1调整体抽出速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    # if self.Aeq[-1, -1] == 0 and self.Aeq_position >= 1:
                    #     for i in range(self.Aeq.shape[1] - 6):
                    #         self.Aeq[0:6, -i - 1] = self.Aeq[0:6, -i - 7]
                    #     self.Aeq_position -= 1
                    #     self.Aeq[0:6, 0:6] = 0
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 13 or action == 'DOWN':  # e.key == pygame.K_DOWN:
                    mbeq = 10 * self.node_number / 100 * (
                                self.model.ResultNode[1] - self.model.ResultNode[0])  # 参数1调整体插入速度
                    self.beq[0] = mbeq[0]
                    self.beq[1] = mbeq[1]
                    self.beq[2] = mbeq[2]
                    # if self.Aeq[0, 0] == 0 and self.Aeq_position >= 1:
                    #     for i in range(self.Aeq.shape[1] - 6):
                    #         self.Aeq[0:6, i] = self.Aeq[0:6, i + 6]
                    #     self.Aeq_position -= 1
                    #     self.Aeq[0:6, -6:] = 0

                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 14 or action == 'LEFT':  # e.key == pygame.K_LEFT:

                    self.f1[int(self.model.l1) * 6 + 0:int(self.model.l1) * 6 + 3] -= 0.01 * np.transpose(
                        self.camera_T[0:3, 2])  # 此处参数调鞘转向速度

                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 15 or action == 'RIGHT':  # e.key == pygame.K_RIGHT:
                    self.f1[int(self.model.l1) * 6 + 0:int(self.model.l1) * 6 + 3] += 0.01 * np.transpose(
                        self.camera_T[0:3, 2])  # 此处参数调鞘
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 16 or action == 'o':  # e.key == pygame.K_o:
                    self.f1[int(self.model.l1) * 6 + 3:int(self.model.l1) * 6 + 6] += 0.003 * np.transpose(
                        self.camera_T[0:3, 2])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 17 or action == 'i':  # e.key == pygame.K_i:
                    self.f1[int(self.model.l1) * 6 + 3:int(self.model.l1) * 6 + 6] -= 0.003 * np.transpose(
                        self.camera_T[0:3, 2])  # 此处参数调鞘转向速度
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 18:  # e.key == pygame.K_SPACE:
                    # self.gravity = self.gravity_copy.copy()
                    # self.gravity = set_gravity(self.node_number, self.gMaterial[0, 3], self.gMaterial[0, 2], 9.8, 7,
                    #                            self.tT, self.gravity)
                    self.f1 = np.zeros(np.shape(self.f1))
                    # print(self.gravity)
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 32:  # e.key == pygame.K_r:
                    self.gravity = self.gravity_copy.copy()
                    self.rotatebool = 0
                    self.camulatebool = True

                elif shared_draw['event'][1] == 33:  # e.key == pygame.K_j:
                    # self.f1[-6:-3] = 0.05 * np.transpose(self.camera_T_start[0:3, 2])  # 此处参数调鞘转向速度
                    self.give_c = True
                    self.rotatebool = 0
                    self.camulatebool = True

            elif shared_draw['set_locate']:
                if shared_draw['event'][1] == 19:  # e.key == pygame.K_3:
                    shared_draw['set_locate'] = 0
                    self.set_locate = False
                    print(self.topnode)
                    self.init_tube(shared_draw)

                elif shared_draw['event'][1] == 20:  # e.key == pygame.K_w:
                    self.topnode[0, 0] += 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 21:  # e.key == pygame.K_s:
                    self.topnode[0, 0] -= 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 22:  # e.key == pygame.K_a:
                    self.topnode[0, 1] -= 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 23:  # e.key == pygame.K_d:
                    self.topnode[0, 1] += 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 24:  # e.key == pygame.K_q:
                    self.topnode[0, 2] -= 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 25:  # e.key == pygame.K_e:
                    self.topnode[0, 2] += 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 26:  # e.key == pygame.K_i:
                    self.topnode[1, 0] += 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 27:  # e.key == pygame.K_k:
                    self.topnode[1, 0] -= 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 28:  # e.key == pygame.K_j:
                    self.topnode[1, 1] -= 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 29:  # e.key == pygame.K_l:
                    self.topnode[1, 1] += 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 30:  # e.key == pygame.K_u:
                    self.topnode[1, 2] -= 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

                elif shared_draw['event'][1] == 31:  # e.key == pygame.K_o:
                    self.topnode[1, 2] += 1
                    self.model.gNode, self.gElement = getnode(self.topnode, 100, False)
                    self.rotatebool = 0
                    self.camulatebool = False

        shared_draw['event'] = np.array([0, 0, shared_draw['event'][2]])
        shared_draw['F'] = self.F
        shared_draw['constraint_triangle'] = self.constraint_triangle
        shared_draw['ResultNode'] = self.model.ResultNode
        shared_draw['l1'] = self.model.l1
        shared_draw['na'] = self.na
        shared_draw['la'] = self.la
        shared_draw['camera_matrix'] = self.camera_matrix
        shared_draw['scaled_factor'] = self.scaled_factor

        shared_RL['state'] = self.model.ResultNode
        shared_RL['F'] = self.F
        shared_RL['U'] = self.U
        # print(self.model.ResultNode[-1])
        return self.model.ResultNode


def fem(shared_mem, shared_fps, velocity_InsertionUnit, velocity_sheath, shared_camera_matrix, trans_matrix,
        tip_matrix, shared_draw, shared_udp, mode, shared_location, shared_RL, Is_warm_start, shared_state):
    cuda.init()
    ctx = cuda.Device(0).make_context()
    mod = get_cuda_SourceModule()
    cuda_mesh = get_cuda_mesh()
    try:
        femmodel = FEM(shared_mem, velocity_InsertionUnit, shared_camera_matrix, mod, trans_matrix, tip_matrix,
                       shared_draw, mode, shared_location, Is_warm_start, cuda_mesh)
        while 1:
            State_s = femmodel.RunFEM(shared_mem, shared_fps, velocity_sheath, shared_draw, ctx,
                                      shared_udp, shared_location, shared_RL, shared_state)

    except SystemExit:
        print('FEM正常退出')
    except (Exception, BaseException) as errorstring:

        print('FEM退出，原因是：%s' % str(errorstring))
        print(traceback.format_exc())
        path_root = os.getcwd()
        with open(path_root + "/data/run_data" + "/Is_warm_start.txt", "w") as f:
            f.write(femmodel.data_time_stream)
    finally:
        ctx.pop()
        shared_state['x'] = None
        event = np.array([0, 0, 5])
        shared_draw['event'] = event
