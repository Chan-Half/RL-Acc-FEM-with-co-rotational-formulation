"""
###########################################################################################
#  @filename:  fem_model_mesh.py
#  @brief:     fem model mesh class
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2026.02.08
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
import math
import os
import pickle
import time
import numpy as np
import scipy.sparse as sp
import trimesh
import scipy.linalg as linalg
from qpsolvers import solve_qp
from fem_code.fem_cuda_mesh import get_cuda_mesh
from fem_code.fem_cuda_MatrixStruct import MatrixStruct


class Model(object):
    def __init__(self, cuda_mesh, delta_h_time, material=None, tip_matrix=None, node_number=100, topnode=None,
                 path=None,
                 mtype='beam'):
        self.type = mtype
        self.gMaterial = material
        self.node_number = node_number
        self.delta_h_time = delta_h_time
        if path:
            try:
                obj = trimesh.load(path)
            except (Exception, BaseException) as er_str:
                print('there is no model, the parameter path is not found')

        if self.type == 'beam':
            try:
                self.topnode = np.zeros((2, 3), dtype=np.float64)
                self.topnode[0] = np.transpose(1000 * tip_matrix[0:3, 3])
                self.topnode[1] = self.topnode[0] + 300 * np.transpose(tip_matrix[0:3, 0])
                self.topnode = np.array(self.topnode, dtype=np.float64)
            except:
                print('tip_matrix is None', tip_matrix)
                # 注意topnode的上下两端不能有参数一样，不能平行于xyz轴
                # self.topnode = np.array([[-24., -25., 31.], [88., 87., 87.]])
                # 10倍是厘米变成mm
                self.topnode = topnode

            self.gNode, self.gElement = self.getnode(True)
            self.gNode = np.array(self.gNode, dtype=np.float64)
            self.gElement = np.array(self.gElement, dtype=np.float64)
            self.ResultNode = np.zeros(np.shape(self.gNode), dtype=np.float64)

            self.l1 = self.node_number - 1
            T0 = self.TransformMatrix()
            self.k0 = self.StiffnessMatrix()
            # self.gK = np.zeros((node_number * 6, node_number * 6), dtype=np.float32)
            self.gK = np.zeros((self.node_number * 6, self.node_number * 6), dtype=np.float64)
            self.gK0 = np.zeros((self.node_number * 6, self.node_number * 6), dtype=np.float64)
            self.F_int = np.zeros(self.node_number * 6, dtype=np.float64)
            self.U = 0
            # self.AssembleStiffnessMatrix(self.k0, T0, self.node_number)

            self.topnode_start = self.topnode.copy()
            self.gNode_start = self.gNode.copy()
            self.gK_start = self.gK.copy()
            self.ResultNode = self.gNode.copy()
            self.dv = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float64)
            self.v = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float64)
            self.dv_copy = np.zeros_like(self.v)
            self.v_start = self.v.copy()
            self.x = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float64)
            self.dx = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float64)
            self.a = np.zeros(np.shape(self.gNode)[0] * 6, dtype=np.float64)

            # corotational fourmulation
            self.I = np.eye(3, dtype=np.float32)
            # 旋转矩阵初始化
            self.R0 = np.zeros((self.node_number - 1, 3, 3), dtype=np.float64)
            self.Rn = self.R0.copy()
            self.R1 = self.R0.copy()
            self.R2 = self.R0.copy()
            # self.R1_local = self.R0.copy()
            # self.R2_local = self.R0.copy()

            # 单元参数计算初始化
            self.ln = np.zeros(self.node_number - 1, dtype=np.float64)
            self.q = np.zeros((self.node_number - 1, 3), dtype=np.float64)
            self.q1 = self.q.copy()
            self.q2 = self.q.copy()
            self.v1 = self.q.copy()

            # 转换矩阵初始化
            self.Gt = np.zeros((self.node_number - 1, 3, 12), dtype=np.float64)
            self.P = np.zeros((self.node_number - 1, 6, 12), dtype=np.float64)
            self.Tg = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)

            self.Bm_local = np.zeros((self.node_number - 1, 7, 7), dtype=np.float64)
            self.Bm = np.zeros((self.node_number - 1, 7, 12), dtype=np.float64)
            self.H = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)

            # 局部变量初始化
            self.dl = np.zeros((self.node_number - 1, 7), dtype=np.float64)
            self.ddl = self.dl.copy()
            self.fl = self.dl.copy()
            self.dfl = self.dl.copy()
            self.dm = np.zeros((self.node_number - 1, 7), dtype=np.float64)
            self.ddm = self.dl.copy()
            self.fm = self.dl.copy()
            self.dfm = self.dl.copy()
            self.dg = np.zeros((self.node_number - 1, 12), dtype=np.float64)
            self.ddg = self.dg.copy()
            self.fg = self.dg.copy()
            self.dfg = self.dg.copy()

            # 刚度矩阵初始化
            self.Kl = np.zeros((self.node_number - 1, 7, 7), dtype=np.float64)
            # Kl为局部刚度矩阵，但只有七维，注意要初始化它以及下面几个刚度矩阵
            self.Km = np.zeros((self.node_number - 1, 7, 7), dtype=np.float64)
            self.Kh = np.zeros((self.node_number - 1, 7, 7), dtype=np.float64)
            self.Ka = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
            self.Kg = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
            self.Kadd = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
            self.Kr = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
            self.Kh0 = np.zeros((self.node_number - 1, 7, 7), dtype=np.float64)

            # 其余中间参量初始化
            self.nambla = np.zeros(self.node_number - 1, dtype=np.float64)
            self.D = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
            self.Q = np.zeros((self.node_number - 1, 12, 3), dtype=np.float64)
            # get M
            self.l0 = self.ln.copy()
            self.set_l0()
            self.M = np.zeros((self.node_number * 6, self.node_number * 6), dtype=np.float64)
            self.M1 = np.zeros((self.node_number - 1, 12, 12), dtype=np.float64)
            self.M_zero = self.M.copy()
            self.get_M1(Is_M_spase_Matrix=True)
            self.ti = np.zeros(14, dtype=np.float64)
            # 初始化部分参数
            self.set_R0Rn()
            self.set_R1R2()
            self.set_Kl()
            # 传入参数到GPU与定义cuda函数
            self.Dl = MatrixStruct(self.dl)
            self.Dl.send_to_gpu()
            self.Fl = MatrixStruct(self.fl)
            self.Fl.send_to_gpu()
            self.KH = MatrixStruct(self.Kh)
            self.KH.send_to_gpu()
            self.II = MatrixStruct(self.I)
            self.II.send_to_gpu()
            self.update_Kh_c = cuda_mesh.get_function("update_Kh")

            self.Dx = MatrixStruct(self.dx)
            self.Dx.send_to_gpu()
            self.RR1 = MatrixStruct(self.R1)
            self.RR1.send_to_gpu()
            self.RR2 = MatrixStruct(self.R2)
            self.RR2.send_to_gpu()
            self.update_R1R2_c = cuda_mesh.get_function("update_R1R2")

            self.RR0 = MatrixStruct(self.R0)
            self.RR0.send_to_gpu()
            self.QQ = MatrixStruct(self.q)
            self.QQ.send_to_gpu()
            self.QQ1 = MatrixStruct(self.q1)
            self.QQ1.send_to_gpu()
            self.QQ2 = MatrixStruct(self.q2)
            self.QQ2.send_to_gpu()
            self.update_q_c = cuda_mesh.get_function("update_q")

            self.VV1 = MatrixStruct(self.v1)
            self.VV1.send_to_gpu()
            self.REsultNode = MatrixStruct(self.ResultNode)
            self.REsultNode.send_to_gpu()
            self.RRn = MatrixStruct(self.Rn)
            self.RRn.send_to_gpu()
            self.update_Rn_c = cuda_mesh.get_function("update_Rn")

            self.TG = MatrixStruct(self.Tg)
            self.TG.send_to_gpu()
            self.update_Tg_c = cuda_mesh.get_function("update_Tg")

            self.GT = MatrixStruct(self.Gt)
            self.GT.send_to_gpu()
            self.Nambla = MatrixStruct(self.nambla)
            self.Nambla.send_to_gpu()
            self.LN = MatrixStruct(self.ln)
            self.LN.send_to_gpu()
            self.update_Gt_c = cuda_mesh.get_function("update_Gt")

            self.PP = MatrixStruct(self.P)
            self.PP.send_to_gpu()
            self.update_P_c = cuda_mesh.get_function("update_P")

            self.BM = MatrixStruct(self.Bm)
            self.BM.send_to_gpu()
            self.BM_local = MatrixStruct(self.Bm_local)
            self.BM_local.send_to_gpu()
            self.update_B_c = cuda_mesh.get_function("update_B")
            # 更新参数
            self.update()
            self.AssembleStiffnessMatrix_corotate()
            #获取初始刚度矩阵
            self.AssembleStiffnessMatrix_corotate0()
            # print(self.Rn[0])
            # print(self.R1[0])
            # print(self.q[0])

    def AssembleStiffnessMatrix(self, k, T, node_number):
        # self.gK[:,:] = 0 #  = np.zeros((node_number * 6, node_number * 6), dtype = np.float32)
        self.gK = np.zeros((self.node_number * 6, self.node_number * 6), dtype=np.float64)
        for i in range(np.shape(k)[0]):
            self.gK[6 * i:6 * i + 12, 6 * i:6 * i + 12] += np.dot(np.dot(T[i], k[i]), np.transpose(T[i]))

    def getnode(self, is_cal_element):
        if self.type == 'beam':
            ele_len_x = (self.topnode[1][0] - self.topnode[0][0]) / (self.node_number - 1)
            ele_len_y = (self.topnode[1][1] - self.topnode[0][1]) / (self.node_number - 1)
            ele_len_z = (self.topnode[1][2] - self.topnode[0][2]) / (self.node_number - 1)
            a = np.zeros((self.node_number, 3), dtype=np.float64)
            b = np.zeros((self.node_number - 1, 3), dtype=np.float64)
            for i in range(self.node_number):
                a[i] = np.array(
                    [self.topnode[0][0] + i * ele_len_x, self.topnode[0][1] + i * ele_len_y,
                     self.topnode[0][2] + i * ele_len_z], dtype=np.float64)
                if is_cal_element and i < self.node_number - 1:
                    b[i] = np.array([i + 1, i + 2, 0], dtype=np.float64)
            return a, b

    def TransformMatrix(self):
        T = np.zeros((np.shape(self.gNode)[0] - 1, 12, 12), dtype=np.float64)
        for i in range(np.shape(self.gNode)[0] - 1):
            L = np.linalg.norm(self.gNode[i + 1] - self.gNode[i])
            lx = (self.gNode[i + 1, 0] - self.gNode[i, 0]) / L
            mx = (self.gNode[i + 1, 1] - self.gNode[i, 1]) / L
            nx = (self.gNode[i + 1, 2] - self.gNode[i, 2]) / L

            T[i] = np.array(
                [[lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0,
                  0, 0, 0],
                 [mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0,
                  0, 0, 0, 0],
                 [nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0,
                  0, 0, 0],
                 [0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0,
                  0, 0, 0, 0],
                 [0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                  0, 0, 0],
                 [0, 0, 0, 0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                  -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2),
                  mx / (lx ** 2 + mx ** 2) ** (1 / 2)],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                  -lx / (lx ** 2 + mx ** 2) ** (1 / 2)],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0]])
        return T

    def StiffnessMatrix(self):
        #  计算单元刚度矩阵
        E = self.gMaterial[0, 0]
        I = self.gMaterial[0, 1]  # 惯性矩
        Jk = 2 * I  # 极惯性矩
        G = E / 2
        A = self.gMaterial[0, 2]

        L1 = np.zeros(np.shape(self.gNode)[0] - 1, dtype=np.float64)
        k = np.zeros((np.shape(self.gNode)[0] - 1, 12, 12), dtype=np.float64)
        for i in range(np.shape(self.gNode)[0] - 1):
            L1[i] = np.linalg.norm(self.gNode[i + 1] - self.gNode[i])
            recalculate_k = -1
            for j in range(i):
                if L1[j] == L1[i]:
                    recalculate_k = j
            if recalculate_k == -1:
                L = L1[i]
                '''k[i, 0, 0] = A * E / L
                k[i, 0, 6] = -k[i, 0, 0]
                k[i, 1, 1] = 12 * E * I / L ** 3
                k[i, 1, 7] = -k[i, 1, 1]
                k[i, 1, 5] = 6 * E * I / L ** 2'''
                k[i] = np.array([[A * E / L, 0, 0, 0, 0, 0, - A * E / L, 0, 0, 0, 0, 0],
                                 [0, 12 * E * I / L ** 3, 0, 0, 0, 6 * E * I / L ** 2, 0, -12 * E * I / L ** 3, 0, 0, 0,
                                  6 * E * I / L ** 2],
                                 [0, 0, 12 * E * I / L ** 3, 0, - 6 * E * I / L ** 2, 0, 0, 0, -12 * E * I / L ** 3, 0,
                                  - 6 * E * I / L ** 2, 0],
                                 [0, 0, 0, Jk / L * G, 0, 0, 0, 0, 0, -Jk / L * G, 0, 0],
                                 [0, 0, - 6 * E * I / L ** 2, 0, 4 * E * I / L, 0, 0, 0, 6 * E * I / L ** 2, 0,
                                  2 * E * I / L,
                                  0],
                                 [0, 6 * E * I / L ** 2, 0, 0, 0, 4 * E * I / L, 0, -6 * E * I / L ** 2, 0, 0, 0,
                                  2 * E * I / L],
                                 [- A * E / L, 0, 0, 0, 0, 0, A * E / L, 0, 0, 0, 0, 0],
                                 [0, -12 * E * I / L ** 3, 0, 0, 0, -6 * E * I / L ** 2, 0, 12 * E * I / L ** 3, 0, 0,
                                  0,
                                  -6 * E * I / L ** 2],
                                 [0, 0, -12 * E * I / L ** 3, 0, 6 * E * I / L ** 2, 0, 0, 0, 12 * E * I / L ** 3, 0,
                                  6 * E * I / L ** 2, 0],
                                 [0, 0, 0, -Jk / L * G, 0, 0, 0, 0, 0, Jk / L * G, 0, 0],
                                 [0, 0, - 6 * E * I / L ** 2, 0, 2 * E * I / L, 0, 0, 0, 6 * E * I / L ** 2, 0,
                                  4 * E * I / L,
                                  0],
                                 [0, 6 * E * I / L ** 2, 0, 0, 0, 2 * E * I / L, 0, -6 * E * I / L ** 2, 0, 0, 0,
                                  4 * E * I / L]], dtype=np.float32)
            if recalculate_k > -1:
                k[i] = k[recalculate_k].copy()

        return k

    def warmstart(self):
        with open(os.getcwd() + "/data/run_data" + "/Is_warm_start.txt", "r") as f:
            for line in f:
                data_line = line.strip("\n").split()
        with open(data_line[0], "rb") as f:
            self.ResultNode = pickle.load(f)
            self.l1 = pickle.load(f)
            # self.camera_matrix = pickle.load(f)
            # self.F = pickle.load(f)
            # self.f1 = pickle.load(f)
            self.v = pickle.load(f)

    def set_dv(self, dv):
        self.dv = np.array(dv, dtype=np.float64)

    def set_v(self):
        self.v = self.v + self.dv

    def set_x(self):
        for i in range(self.node_number):
            self.x[6 * i:6 * i + 3] += self.delta_h_time * self.v[6 * i:6 * i + 3]
            self.x[6 * i + 3:6 * i + 6] += self.delta_h_time * self.v[6 * i + 3:6 * i + 6]
        # print('dv:', self.dv[-6:])
        # print('v:', self.v[-6:])
        # print('x:', self.x[-6:])

    def set_ResultNode(self):
        self.ResultNode = np.zeros(np.shape(self.gNode), dtype=np.float64)
        for i in range(self.gNode.shape[0]):
            self.ResultNode[i][0] = self.gNode[i][0] + self.x[6 * i + 0] + 0.0
            self.ResultNode[i][1] = self.gNode[i][1] + self.x[6 * i + 1] + 0.0
            self.ResultNode[i][2] = self.gNode[i][2] + self.x[6 * i + 2] + 0.0

    def set_l1(self, l1):
        self.l1 = l1

    def resetmodel(self):
        self.gNode = self.gNode_start.copy()
        self.gK = self.gK_start.copy()
        self.dv = self.v_start.copy()
        self.x = self.v_start.copy()
        self.v = self.v_start.copy()
        self.l1 = self.node_number - 1

    # corotational fourmulation
    def Ts(self, theta):
        # 求导
        theta_norm = np.linalg.norm(theta)
        # assert theta_norm < 3.1415926535 * 2
        if theta_norm == 0:
            return self.I
        e = theta / theta_norm
        td = np.sin(theta_norm / 2) / (theta_norm / 2)
        te = np.sin(theta_norm) / theta_norm
        return te * self.I + (1 - te) * np.outer(e, e) + 0.5 * td ** 2 * self.theta_hat(theta)

    def Ts_inv(self, theta):
        # 求导
        theta_norm = np.linalg.norm(theta)
        # assert theta_norm < 3.1415926535 * 2
        if theta_norm == 0:
            return self.I
        # nambla = (2 * np.sin(theta_norm) - theta_norm * (1 + np.cos(theta_norm))) / (2 * theta_norm ** 2 * np.sin(theta_norm / 2))
        # hat = self.theta_hat(theta)
        td = theta_norm / 2 / np.tan(theta_norm / 2)
        return td * self.I + (1 - td) / theta_norm ** 2 * np.outer(theta, theta) - 0.5 * self.theta_hat(theta)
        # return self.I + nambla*np.dot(hat, hat)- 0.5 * hat

    def theta_hat(self, theta):
        # 反对称矩阵
        return np.array([[0, -theta[2], theta[1]], [theta[2], 0, -theta[0]], [-theta[1], theta[0], 0]], dtype=np.float64)

    def AxialAngle2Rotation(self, theta):
        # return linalg.expm(np.cross(np.eye(3), theta))
        theta_norm = np.linalg.norm(theta)
        theta_hat = self.theta_hat(theta)
        if theta_norm == 0:
            return self.I
        return self.I + np.sin(theta_norm) / theta_norm * theta_hat + (
                    np.sin(theta_norm / 2) / theta_norm * 2) ** 2 * np.dot(theta_hat, theta_hat)

    def set_R0Rn(self):
        # 初始旋转，以后可以改成一致的情况

        for i in range(self.node_number - 1):
            L = np.linalg.norm(self.gNode[i + 1] - self.gNode[i])
            lx = (self.gNode[i + 1, 0] - self.gNode[i, 0]) / L
            mx = (self.gNode[i + 1, 1] - self.gNode[i, 1]) / L
            nx = (self.gNode[i + 1, 2] - self.gNode[i, 2]) / L
            self.R0[i] = np.array(
                [[lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2)],
                 [mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2)],
                 [nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0]], dtype=np.float64)
        self.Rn = self.R0.copy()

        # print('Rn0', self.Rn[0])
        # return R0, Rn

    def set_R1R2(self):
        for i in range(self.node_number - 1):
            self.R1[i] = np.eye(3, dtype=np.float64)
            self.R2[i] = np.eye(3, dtype=np.float64)
            # self.R1_local[i] = np.eye(3)
            # self.R2_local[i] = np.eye(3)

    def update_R1R2(self):
        # 更新于dm之后
        # 注意R1R2的更新方式可能会变
        for i in range(self.node_number - 1):
            # self.R1_local[i] = np.dot(self.AxialAngle2Rotation(self.ddm[i, 1:4]), self.R1_local[i])
            # self.R2_local[i] = np.dot(self.AxialAngle2Rotation(self.ddm[i, 4:7]), self.R2_local[i])
            self.R1[i] = np.dot(self.AxialAngle2Rotation(self.dx[i * 6 + 3:i * 6 + 6]), self.R1[i])
            self.R2[i] = np.dot(self.AxialAngle2Rotation(self.dx[i * 6 + 9:i * 6 + 12]), self.R2[i])

    def update_ln(self):
        for i in range(self.node_number - 1):
            self.ln[i] = np.linalg.norm(self.ResultNode[i + 1] - self.ResultNode[i])

    def update_q(self):
        for i in range(self.node_number - 1):
            # 注意这里的乘法,经过测试暂未发现行列的bug
            self.q1[i] = np.dot(self.R1[i], np.dot(self.R0[i], np.array([0, 1, 0]).transpose()))
            self.q2[i] = np.dot(self.R2[i], np.dot(self.R0[i], np.array([0, 1, 0]).transpose()))
            self.q[i] = 0.5 * (self.q1[i] + self.q2[i])
        # print(self.q1)
        # print(self.q2)

    def update_Rn(self):
        # 注意Rn的更新方式可能会变
        for i in range(self.node_number - 1):
            self.v1[i] = (self.ResultNode[i + 1] - self.ResultNode[i]) / np.linalg.norm(self.ResultNode[i + 1] - self.ResultNode[i])
            self.Rn[i, :, 0] = self.v1[i]
            cross_rn = np.cross(self.v1[i], self.q[i])
            cross_rn /= np.linalg.norm(cross_rn)
            self.Rn[i, :, 2] = cross_rn
            self.Rn[i, :, 1] = np.cross(cross_rn, self.v1[i])
            # print('Rn', self.Rn[0])

    def update_Gt(self):
        for i in range(self.node_number - 1):
            # 这里要不要转置,以及公式中到底是０和１下标还是１和２
            Rntq = np.dot(self.Rn[i].transpose(), self.q[i])
            Rntq1 = np.dot(self.Rn[i].transpose(), self.q1[i])
            Rntq2 = np.dot(self.Rn[i].transpose(), self.q2[i])

            self.nambla[i] = Rntq[0] / Rntq[1]

            self.Gt[i, 0, 2] = Rntq[0] / Rntq[1] / self.ln[i]
            self.Gt[i, 0, 3] = Rntq1[1] / Rntq[1] / 2.0
            self.Gt[i, 0, 4] = -Rntq1[0] / Rntq[1] / 2.0
            self.Gt[i, 0, 8] = -Rntq[0] / Rntq[1] / self.ln[i]
            self.Gt[i, 0, 9] = Rntq2[1] / Rntq[1] / 2.0
            self.Gt[i, 0, 10] = -Rntq2[0] / Rntq[1] / 2.0

            self.Gt[i, 1, 2] = 1.0 / self.ln[i]
            self.Gt[i, 1, 8] = -1.0 / self.ln[i]

            self.Gt[i, 2, 1] = -1.0 / self.ln[i]
            self.Gt[i, 2, 7] = 1.0 / self.ln[i]
            # print('Gt', self.Gt[0])

    def update_P(self):
        for i in range(self.node_number - 1):
            self.P[i, 0:3, :] = -self.Gt[i]
            self.P[i, 3:6, :] = -self.Gt[i]
            self.P[i, 0:3, 3:6] += self.I
            self.P[i, 3:6, 9:12] += self.I

    def update_Tg(self):
        # 更新于Ｒn后面
        for i in range(self.node_number - 1):
            for j in range(4):
                self.Tg[i, j * 3:j * 3 + 3, j * 3:j * 3 + 3] = self.Rn[i]

    def update_B(self):
        # 更新于dl, P, Tg之后
        for i in range(self.node_number - 1):
            self.Bm_local[i, 0, 0] = 1
            self.Bm_local[i, 1:4, 1:4] = self.Ts_inv(self.dl[i, 1:4])
            self.Bm_local[i, 4:7, 4:7] = self.Ts_inv(self.dl[i, 4:7])

            self.Bm[i, 0, 0:3] = -self.v1[i]
            self.Bm[i, 0, 6:9] = self.v1[i]
            self.Bm[i, 1:7, :] = np.dot(self.P[i], np.transpose(self.Tg[i]))
            # print('Bm_local', self.Bm_local)
            # print('Bm', self.Bm)

    def update_dm(self):
        for i in range(self.node_number - 1):
            self.ddm[i] = np.dot(self.Bm[i], self.dx[i * 6:i * 6 + 12])
            self.dm[i] += self.ddm[i]
        # print('v', self.v)
        # print('ddm', self.ddm)
        # print('dm', self.dm)




    def update_dg(self):
        for i in range(self.node_number - 1):
            self.ddg[i] = np.dot(self.H[i], self.dx[i * 6:i * 6 + 12])
            self.dg[i] += self.ddg[i]




    def update_dl(self):
        for i in range(self.node_number - 1):
            self.ddl[i] = np.dot(self.Bm_local[i], self.ddm[i])
            self.dl[i] += self.ddl[i]
        # print('ddl', self.ddl)
        # print('dl', self.dl)



    def update_fl(self):
        for i in range(self.node_number - 1):
            self.dfl[i] = np.dot(self.Kl[i], self.ddl[i])
            self.fl[i] += self.dfl[i]
        # print('dfl', self.dfl)
        # print('fl', self.fl)

    def update_fm(self):
        for i in range(self.node_number - 1):
            self.dfm[i] = np.dot(self.Km[i], self.ddm[i])
            self.fm[i] += self.dfm[i]
        # print('dfm', self.dfm)
        # print('fm', self.fm)

    def update_fg(self):
        for i in range(self.node_number - 1):
            self.dfg[i] = np.dot(self.Kg[i], self.ddg[i])
            self.fg[i] += self.dfg[i]

    def update_Kh1(self):
        for i in range(self.node_number - 1):
            for j in range(2):
                theta = self.dl[i, 1 + j * 3:4 + j * 3]
                m = self.fl[i, 1 + j * 3:4 + j * 3]
                a = np.linalg.norm(theta)
                # if i == 0 and j == 0:
                #     print(theta)
                #     print(m)

                # 零点求导不一定是单位矩阵
                if a == 0:
                    self.Kh[i, 1 + j * 3:4 + j * 3, 1 + j * 3:4 + j * 3] = self.I
                else:
                    nambla = (2. * math.sin(a) - a * (1. + math.cos(a))) / (2. * a ** 2. * math.sin(a / 2.))
                    # 注意这里加了一个８的系数，在不对称影响参考一文中有，但是共旋参考文献中没有这一点
                    miu = (a * (a + math.sin(a)) - 8. * math.sin(a / 2.) *math.sin(a / 2.)) / (4. * a*a*a*a * math.sin(a / 2.)*math.sin(a / 2.))
                    if i == 0 and j == 0:
                        print('Kha', a)
                        print('Khnambla', nambla)
                        print('Khmiu', miu)

                    k1 = nambla * (np.outer(theta, m) - 2 * np.outer(m, theta) + np.dot(theta, m) * self.I)
                    k2 = miu * np.dot(np.dot(self.theta_hat(theta), self.theta_hat(theta)), np.outer(m, theta))
                    k3 = 0.5 * self.theta_hat(m)
                    kh = np.dot(k1 + k2 - k3, self.Ts_inv(theta))
                    self.Kh[i, 1 + j * 3:4 + j * 3, 1 + j * 3:4 + j * 3] = kh

    def update_Km(self):
        for i in range(self.node_number - 1):
            self.Km[i] = np.dot(np.transpose(self.Bm_local[i]), np.dot(self.Kl[i], self.Bm_local[i])) + self.Kh[i]
            # 注意这里将Kg对称了
            # self.Km[i] = 0.5 * (self.Km[i] + np.transpose(self.Km[i]))

    def update_Ka(self):
        for i in range(self.node_number - 1):
            D3 = (self.I - np.outer(self.v1[i], self.v1[i])) / self.ln[i]
            self.D[i, 0:3, 0:3] = D3
            self.D[i, 6:9, 6:9] = D3
            self.D[i, 0:3, 6:9] = -D3
            self.D[i, 6:9, 0:3] = -D3
            fm = self.fm[i]
            Qi = np.dot(self.P[i].transpose(), fm[1:7])
            for j in range(4):
                self.Q[i, j * 3:j * 3 + 3, :] = self.theta_hat(Qi[j * 3:j * 3 + 3])

            a1 = self.nambla[i] * (fm[1] + fm[4]) - (fm[2] + fm[5])
            a2 = fm[3] + fm[6]
            a = np.array([0, a1, a2], dtype=np.float64) / self.ln[i]

            ka1 = self.D[i] * fm[0] - np.dot(self.Tg[i],
                                             np.dot(self.Q[i], np.dot(self.Gt[i], np.transpose(self.Tg[i]))))
            self.Ka[i] = ka1 + np.dot(self.Tg[i], np.outer(np.dot(self.Gt[i].transpose(), a), self.Bm[i, 0]))

    def update_Kg(self):
        for i in range(self.node_number - 1):
            self.Kg[i] = np.dot(np.transpose(self.Bm[i]), np.dot(self.Km[i], self.Bm[i])) + self.Ka[i]
            # 注意这里将Kg对称了
            # self.Kg[i] = 0.5*(self.Kg[i] + np.transpose(self.Kg[i]))

    def set_Kl(self):
        #  计算单元刚度矩阵
        E = self.gMaterial[0, 0]
        I = self.gMaterial[0, 1]  # 惯性矩
        Jk = 2 * I  # 极惯性矩
        G = E / 2
        A = self.gMaterial[0, 2]

        l0 = np.zeros(np.shape(self.gNode)[0] - 1, dtype=np.float64)

        for i in range(np.shape(self.gNode)[0] - 1):
            l0[i] = np.linalg.norm(self.gNode[i + 1] - self.gNode[i])
            recalculate_k = -1
            for j in range(i):
                if l0[j] == l0[i]:
                    recalculate_k = j
            if recalculate_k == -1:
                L = l0[i]
                self.Kl[i] = np.array([[A * E / L, 0, 0, 0, 0, 0, 0],
                                       [0, Jk / L * G, 0, 0, -Jk / L * G, 0, 0],
                                       [0, 0, 4 * E * I / L, 0, 0, 2 * E * I / L, 0],
                                       [0, 0, 0, 4 * E * I / L, 0, 0, 2 * E * I / L],
                                       [0, -Jk / L * G, 0, 0, Jk / L * G, 0, 0],
                                       [0, 0, 2 * E * I / L, 0, 0, 4 * E * I / L, 0],
                                       [0, 0, 0, 2 * E * I / L, 0, 0, 4 * E * I / L]], dtype=np.float64)
            if recalculate_k > -1:
                self.Kl[i] = self.Kl[recalculate_k].copy()

    def update(self):
        # self.ti[0] = time.time()
        self.update_ln()
        # self.update_dg()
        self.update_dm()
        self.update_dl()
        self.update_fl()
        self.update_fm()
        # self.ti[1] = time.time()
        # self.update_fg()
        self.Dx.rehtod(self.dx)
        self.REsultNode.rehtod(self.ResultNode)
        self.LN.rehtod(self.ln)
        self.Dl.rehtod(self.dl)
        self.Fl.rehtod(self.fl)
        # self.RR1.rehtod(self.R1)
        # self.RR2.rehtod(self.R2)
        self.update_R1R2_c(self.Dx.cptr, self.RR1.cptr, self.RR2.cptr, self.II.cptr, block=(1, 1, 1),
                           grid=(self.node_number - 1, 1, 1))
        self.R1 = self.RR1.d2h()
        self.R2 = self.RR2.d2h()
        # self.update_R1R2()

        self.ti[2] = time.time()
        # self.update_q()
        self.update_q_c(self.RR1.cptr, self.RR2.cptr, self.RR0.cptr, self.QQ.cptr, self.QQ1.cptr, self.QQ2.cptr, block=(1, 1, 1),
                        grid=(self.node_number - 1, 1, 1))
        # self.q = self.QQ.d2h()
        # self.q1 = self.QQ1.d2h()
        # self.q2 = self.QQ2.d2h()
        # self.ti[3] = time.time()
        self.update_Rn_c(self.VV1.cptr, self.LN.cptr, self.REsultNode.cptr, self.RRn.cptr, self.QQ.cptr,
                         block=(1, 1, 1),
                         grid=(self.node_number - 1, 1, 1))
        # self.update_Rn()
        self.v1 = self.VV1.d2h()
        # self.Rn = self.RRn.d2h()
        # self.ti[4] = time.time()
        # self.update_Tg()
        # self.RRn.rehtod(self.Rn)
        self.update_Tg_c(self.TG.cptr, self.RRn.cptr, block=(1, 1, 1), grid=(self.node_number - 1, 1, 1))
        self.Tg = self.TG.d2h()
        # self.ti[5] = time.time()
        # self.update_Gt()
        self.update_Gt_c(self.RRn.cptr, self.QQ.cptr, self.QQ1.cptr, self.QQ2.cptr, self.Nambla.cptr, self.LN.cptr, self.GT.cptr, block=(1, 1, 1), grid=(self.node_number - 1, 1, 1))
        self.Gt = self.GT.d2h()

        # self.ti[6] = time.time()
        # self.update_P()
        self.update_P_c(self.PP.cptr, self.GT.cptr, block=(1, 1, 1), grid=(self.node_number - 1, 1, 1))
        self.P = self.PP.d2h()

        # self.ti[7] = time.time()
        # self.update_B()
        self.update_B_c(self.BM.cptr, self.BM_local.cptr, self.VV1.cptr, self.PP.cptr, self.TG.cptr, self.Dl.cptr, self.II.cptr, block=(1, 1, 1), grid=(self.node_number - 1, 1, 1))
        self.Bm = self.BM.d2h()
        self.Bm_local = self.BM_local.d2h()
        # self.ti[8] = time.time()
        # self.update_H()

        self.update_Kh_c(self.Dl.cptr, self.Fl.cptr, self.KH.cptr, self.II.cptr, block=(2, 1, 1),
                         grid=(self.node_number - 1, 1, 1))
        self.Kh = self.KH.d2h()
        # self.update_Kh1()
        # self.ti[9] = time.time()
        self.update_Km()
        # self.ti[10] = time.time()
        self.update_Ka()
        # self.ti[11] = time.time()
        self.update_Kg()
        # self.ti[12] = time.time()
        # self.update_Kadd()
        # self.update_Kr()
        self.update_M()
        # self.ti[13] = time.time()
        # print('dx', self.ti[1] - self.ti[0])
        # print('R1R2', self.ti[2] - self.ti[1])
        # print('q', self.ti[3] - self.ti[2])
        # print('Rn', self.ti[4] - self.ti[3])
        # print('Tg', self.ti[5] - self.ti[4])
        # print('Gt', self.ti[6] - self.ti[5])
        # print('P', self.ti[7] - self.ti[6])
        # print('B', self.ti[8] - self.ti[7])
        # print('Kh', self.ti[9] - self.ti[8])
        # print('Km', self.ti[10] - self.ti[9])
        # print('Ka', self.ti[11] - self.ti[10])
        # print('Kg', self.ti[12] - self.ti[11])
        # print('M', self.ti[13] - self.ti[12])
        # print('update', self.ti[13]-self.ti[0])
        # print(self.M1)
        # print(self.M)
        # print('ln', self.ln)
        # print('l0', self.l0)
        # print('Bm_local', self.Bm_local[0])
        # print('Bm', self.Bm[-1])
        # print('Kl', self.Kl)
        # print('Kh', self.Kh[-1])
        # print('Km', self.Km[-1])
        # print('Ka', self.Ka[-1])
        # print('Kg', self.Kg)
        # print('K0', self.k0[-1])

    def AssembleStiffnessMatrix_corotate(self):
        # self.gK[:,:] = 0 #  = np.zeros((node_number * 6, node_number * 6), dtype = np.float32)
        # ti = time.time()
        self.gK = np.zeros((self.node_number * 6, self.node_number * 6), dtype=np.float64)
        for i in range(self.node_number - 1):
            self.gK[6 * i:6 * i + 12, 6 * i:6 * i + 12] += self.Kg[i]
        # print('gK', time.time()-ti)


    def AssembleStiffnessMatrix_corotate0(self):
        # self.gK[:,:] = 0 #  = np.zeros((node_number * 6, node_number * 6), dtype = np.float32)
        # ti = time.time()
        self.gK0 = np.zeros((self.node_number * 6, self.node_number * 6), dtype=np.float64)
        for i in range(self.node_number - 1):
            self.gK0[6 * i:6 * i + 12, 6 * i:6 * i + 12] += self.Kg[i]
        # print('gK', time.time()-ti)

    def AssembleInternalForce_corotate(self):
        """
        计算共旋法下的全局总内力 F_int (基于总变形映射，绝对禁止增量累加)
        Returns:
            F_int (np.ndarray): 长度为 node_number * 6 的一维数组，表示当前的全局总内力
        """

        F_int = np.zeros(self.node_number * 6, dtype=np.float64)
        for i in range(self.node_number - 1):
            # 1. 局部坐标系下计算总内力 f_l = K_l * d_l (这里是纯线性关系)
            # self.Kl 形状为 (7, 7), self.dl 形状为 (7,)
            fl_total = np.dot(self.Kl[i], self.dl[i])

            # 2. 从局部构型映射到中间构型 f_m = Bm_local^T * f_l
            # self.Bm_local 形状为 (7, 7)，转置后为 (7, 7)
            fm_total = np.dot(self.Bm_local[i].transpose(), fl_total)

            # 3. 从中间构型映射到全局构型 f_g = B_m^T * f_m
            # self.Bm 形状为 (7, 12)，转置后为 (12, 7)
            fg_total = np.dot(self.Bm[i].transpose(), fm_total)

            # 4. 组装到全局内力向量
            F_int[6 * i: 6 * i + 12] += fg_total

        return F_int


    def get_M1(self, Is_M_spase_Matrix=True):
        if self.type == 'beam':
            if Is_M_spase_Matrix:
                # get M
                node_ele_m = self.gMaterial[0, 2] * self.l0[0] * self.gMaterial[0, 3]
                for i in range(self.node_number - 1):
                    self.M1[i, 0, 0] = 0.5 * node_ele_m
                    self.M1[i, 1, 1] = 0.5 * node_ele_m
                    self.M1[i, 2, 2] = 0.5 * node_ele_m
                    self.M1[i, 3, 3] = self.gMaterial[0, 3] * self.gMaterial[0, 1] * 2
                    self.M1[i, 4, 4] = self.l0[0] ** 3 * self.gMaterial[0, 2] * self.gMaterial[0, 3] / 3
                    self.M1[i, 5, 5] = self.l0[0] ** 3 * self.gMaterial[0, 2] * self.gMaterial[0, 3] / 3
                    self.M1[i, 6, 6] = 0.5 * node_ele_m
                    self.M1[i, 7, 7] = 0.5 * node_ele_m
                    self.M1[i, 8, 8] = 0.5 * node_ele_m
                    self.M1[i, 9, 9] = self.gMaterial[0, 3] * self.gMaterial[0, 1] * 2
                    self.M1[i, 10, 10] = self.l0[0] ** 3 * self.gMaterial[0, 2] * self.gMaterial[0, 3] / 3
                    self.M1[i, 11, 11] = self.l0[0] ** 3 * self.gMaterial[0, 2] * self.gMaterial[0, 3] / 3

                    # self.M1[i, 0, 6] = 70
                    # self.M1[i, 1, 5] = 22*self.l0[i]
                    # self.M1[i, 2, 4] = -22*self.l0[i]
                    #
                    # self.M1[i, 1, 7] = 54
                    # self.M1[i, 2, 8] = 54
                    # self.M1[i, 3, 9] = 70
                    # self.M1[i, 4, 10] = -3 * self.l0[i] ** 2
                    # self.M1[i, 5, 11] = -3 * self.l0[i] ** 2
                    # self.M1[i, 2, 10] = 13 * self.l0[i]
                    # self.M1[i, 1, 11] = -13 * self.l0[i]
                    # self.M1[i, 4, 8] = -13 * self.l0[i]
                    # self.M1[i, 5, 7] = 13 * self.l0[i]
                    # self.M1[i, 7, 11] = -22 * self.l0[i]
                    # self.M1[i, 8, 10] = 22 * self.l0[i]
                    #
                    # self.M1[i, 1, 7] = 54
                    # self.M1[i, 2, 8] = 54
                    # self.M1[i, 3, 9] = 70
                    # self.M1[i, 4, 10] = -3 * self.l0[i] ** 2
                    # self.M1[i, 5, 11] = -3 * self.l0[i] ** 2
                    # self.M1[i, 2, 10] = 13 * self.l0[i]
                    # self.M1[i, 1, 11] = -13 * self.l0[i]
                    # self.M1[i, 4, 8] = -13 * self.l0[i]
                    # self.M1[i, 5, 7] = 13 * self.l0[i]
                    # self.M1[i, 7, 11] = -22 * self.l0[i]
                    # self.M1[i, 8, 10] = 22 * self.l0[i]
                    #
                    # self.M1[i] = node_ele_m / 420 * (self.M1[i]+self.M1[i].transpose())

    def update_M(self, Is_M_spase_Matrix=True):
        self.M = self.M_zero.copy()
        if Is_M_spase_Matrix:
            for i in range(self.node_number - 1):
                self.M[6 * i:6 * i + 12, 6 * i:6 * i + 12] += self.M1[i]
                # np.dot(np.dot(self.Tg[i], self.M1[i]), np.transpose(self.Tg[i]))
        else:
            self.M = self.gMaterial[0, 2] * np.linalg.norm(self.topnode[1] - self.topnode[0]) / (
                    self.node_number - 1) * self.gMaterial[0, 1]

    def set_l0(self):
        for i in range(self.node_number - 1):
            self.l0[i] = np.linalg.norm(self.gNode[i + 1] - self.gNode[i])

    def get_M(self, Is_M_spase_Matrix=True):
        if self.type == 'beam':
            if Is_M_spase_Matrix:
                # get M
                node_ele_m = self.gMaterial[0, 2] * np.linalg.norm(self.topnode[1] - self.topnode[0]) / (
                        self.node_number - 1) * self.gMaterial[0, 3]
                # self.M = node_ele_m*np.eye(self.node_number * 6, dtype=np.float32)
                M_mid = np.zeros(self.node_number * 6, dtype=np.float64)
                for i in range(self.node_number):
                    if i == 0 or i == self.node_number - 1:
                        M_mid[i * 6] = 0.5 * node_ele_m
                        M_mid[i * 6 + 1] = 0.5 * node_ele_m
                        M_mid[i * 6 + 2] = 0.5 * node_ele_m
                        # M_mid[i * 6 + 3] = 0.5 * node_ele_m
                        # M_mid[i * 6 + 4] = self.l0 / 420 * node_ele_m
                        # M_mid[i * 6 + 5] = self.l0 / 420 * node_ele_m
                        continue
                    M_mid[i * 6] = node_ele_m
                    M_mid[i * 6 + 1] = node_ele_m
                    M_mid[i * 6 + 2] = node_ele_m
                M_offset = np.zeros(1, dtype=np.float64)  # 偏移量
                self.M = sp.dia_matrix((M_mid, M_offset), shape=(self.node_number * 6, self.node_number * 6),
                                       dtype=np.float64)
            else:
                self.M = self.gMaterial[0, 2] * np.linalg.norm(self.topnode[1] - self.topnode[0]) / (
                        self.node_number - 1) * self.gMaterial[0, 1]

    def update_H(self):
        # 更新于dl, P, Tg之后
        for i in range(self.node_number - 1):
            self.H[i, 0:3, 0:3] = self.I
            self.H[i, 3:6, 3:6] = self.Ts(self.delta_h_time * self.v[6 * i + 3:6 * i + 6])
            self.H[i, 6:9, 6:9] = self.I
            self.H[i, 9:12, 9:12] = self.Ts(self.delta_h_time * self.v[6 * i + 9:6 * i + 12])

    def update_Kadd(self):
        for i in range(self.node_number - 1):
            for j in range(2):
                theta = self.delta_h_time * self.v[6 * i + 6 * j + 3:6 * i + 6 * j + 6]
                v = self.fg[i, 3 + j * 6:6 + j * 6]
                a = np.linalg.norm(theta)
                # 零点求导不一定是单位矩阵
                if a == 0:
                    self.Kadd[i, 3 + j * 6:6 + j * 6, 3 + j * 6:6 + j * 6] = self.I
                else:
                    e = theta / a
                    k1 = -(np.sin(a) / a - (np.sin(a / 2) / (a / 2)) ** 2) * np.outer(np.cross(e, v), e)
                    k2 = 0.5 * (np.sin(a / 2) / (a / 2)) ** 2 * self.theta_hat(v)
                    k3 = (np.cos(a) - np.sin(a) / a) / a * (np.outer(e, v) - np.dot(e, v) * np.outer(e, e))
                    k4 = (1 - np.sin(a) / a) / a * (
                                np.outer(e, v) - 2 * np.dot(e, v) * np.outer(e, e) + np.dot(e, v) * self.I)
                    self.Kh[i, 1 + j * 3:4 + j * 3, 1 + j * 3:4 + j * 3] = k1 + k2 + k3 + k4

    def update_Kr(self):
        for i in range(self.node_number - 1):
            self.Kr[i] = np.dot(np.transpose(self.H[i]), np.dot(self.Kg[i], self.H[i])) + self.Kadd[i]
            # 注意这里将Kr对称了
            # self.Kr[i] = 0.5 * (self.Kr[i] + np.transpose(self.Kr[i]))


    def update_state_implicit_integration(self, dv):
        self.dv = np.array(dv, dtype=np.float64)
        self.v = self.v + self.dv
        self.a = self.dv / self.delta_h_time
        self.dx = self.v * self.delta_h_time
        self.x += self.dx
        self.set_ResultNode()
        self.update()

    def update_state_newmark_beta_integration(self, a, gamma, beta):
        # self.a为前一时刻t的加速度，a为本时刻t+1的加速度
        self.dx = self.v * self.delta_h_time + ((0.5 - beta) * self.a + beta * a) * self.delta_h_time ** 2
        self.x += self.dx
        self.dv = ((1 - gamma) * self.a + gamma * a) * self.delta_h_time
        self.v += self.dv
        self.a = a
        self.set_ResultNode()
        self.update()

    def update_state_hht_alpha_integration(self, a, gamma, beta):
        # self.a为前一时刻t的加速度，a为本时刻t+1的加速度
        # if a is None:
        #     a = self.a.copy()
        self.dx = self.v * self.delta_h_time + ((0.5 - beta) * self.a + beta * a) * self.delta_h_time ** 2
        self.x += self.dx
        self.dv = ((1 - gamma) * self.a + gamma * a) * self.delta_h_time
        self.v += self.dv
        self.a = a
        self.set_ResultNode()
        self.update()

    def update_state_hht_alpha_integration_x(self, dx, gamma, beta):
        # self.a为前一时刻t的加速度，a为本时刻t+1的加速度
        # if a is None:
        #     a = self.a.copy()
        self.dx = dx
        self.x += self.dx
        a_old = self.a.copy()
        self.a = 1 / beta / self.delta_h_time**2 *self.dx - 1 / beta / self.delta_h_time * self.v - (1/2/beta-1) * a_old
        self.dv = gamma / beta / self.delta_h_time * self.dx - gamma / beta * self.v - (gamma/2/beta-1) * self.delta_h_time * a_old
        self.v += self.dv

        self.set_ResultNode()
        self.update()

    def update_state_newmark_beta_integration_x(self, x, gamma, beta):
        # self.a为前一时刻t的加速度，a为本时刻t+1的加速度
        self.dx = x - self.x
        self.dv = gamma / (beta * self.delta_h_time) * self.dx - gamma / beta * self.v - (gamma / 2 / beta - 1) * self.a * self.delta_h_time
        self.v += self.dv
        self.x = x
        self.set_ResultNode()
        self.update()



    def backup_state(self):
        """ 备份当前帧 (t_n) 的所有独立物理与几何状态 """
        # 1. 运动学状态
        self._bk_x = self.x.copy()
        self._bk_v = self.v.copy()
        self._bk_a = self.a.copy()

        # 2. 局部增量/内力状态 (共旋法核心历史变量)
        self._bk_dl = self.dl.copy()
        self._bk_dm = self.dm.copy()
        self._bk_dg = self.dg.copy()
        self._bk_fl = self.fl.copy()
        self._bk_fm = self.fm.copy()
        self._bk_fg = self.fg.copy()

        # 3. 旋转矩阵历史
        self._bk_R1 = self.R1.copy()
        self._bk_R2 = self.R2.copy()

    def restore_state(self):
        """ 完美回滚到 t_n 时刻，并清理不需要的增量 """
        # 1. 恢复运动学状态
        self.x = self._bk_x.copy()
        self.v = self._bk_v.copy()
        self.a = self._bk_a.copy()

        # 2. 恢复增量状态
        self.dl = self._bk_dl.copy()
        self.dm = self._bk_dm.copy()
        self.dg = self._bk_dg.copy()
        self.fl = self._bk_fl.copy()
        self.fm = self._bk_fm.copy()
        self.fg = self._bk_fg.copy()

        self.R1 = self._bk_R1.copy()
        self.R2 = self._bk_R2.copy()

        # 3. 极其关键：必须将恢复的旋转矩阵手动推送到 GPU！
        # 因为 update_R1R2_c 是在 GPU 内部读取 RR1 并进行累加的
        self.RR1.rehtod(self.R1)
        self.RR2.rehtod(self.R2)

        # 4. 清零步进增量 dx 和 dv
        # 这样调用 update() 时，ddm 和 ddl 就会算出来是 0，保证 dl 等变量不会被错误累加
        self.dx.fill(0.0)
        self.dv.fill(0.0)

        # 5. 基于干净的 t_n 状态，重新同步所有的从属矩阵 (Bm, Gt, Kg 等)
        self.set_ResultNode()
        self.update()
