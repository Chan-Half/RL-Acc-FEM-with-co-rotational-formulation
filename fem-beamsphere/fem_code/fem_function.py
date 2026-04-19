"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:  fem_function.py
#  @brief:     fem function
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2023.04.03
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""

import numpy as np


# @jit(nopython=True)
def getnode(topnode, node_num, is_cal_element):
    ele_len_x = (topnode[1][0] - topnode[0][0]) / (node_num - 1)
    ele_len_y = (topnode[1][1] - topnode[0][1]) / (node_num - 1)
    ele_len_z = (topnode[1][2] - topnode[0][2]) / (node_num - 1)
    a = np.zeros((node_num, 3), dtype=np.float32)
    b = np.zeros((node_num - 1, 3), dtype=np.float32)
    for i in range(node_num):
        a[i] = np.array([topnode[0][0] + i * ele_len_x, topnode[0][1] + i * ele_len_y, topnode[0][2] + i * ele_len_z])
        if is_cal_element and i < node_num - 1:
            b[i] = np.array([i + 1, i + 2, 0])
    return a, b


# @jit(nopython=True)
def AssembleStiffnessMatrix1(k, T, node_number):
    gK = np.zeros((node_number * 6, node_number * 6))
    for i in range(np.shape(k)[0]):
        gK[6 * i:6 * i + 12, 6 * i:6 * i + 12] += np.dot(np.dot(T[i], k[i]), np.transpose(T[i]))
    return gK


# @jit(nopython=True)
def TransformMatrix(RNode):
    #  计算单元的坐标转换矩阵( 局部坐标 -> 整体坐标 )
    #  输入参数
    #      ie  ----- 节点号
    #  返回值
    #      T ------- 从局部坐标到整体坐标的坐标转换矩阵
    # global gElement, gNode

    T = np.zeros((np.shape(RNode)[0] - 1, 12, 12))
    for i in range(np.shape(RNode)[0] - 1):
        L = np.linalg.norm(RNode[i + 1] - RNode[i])
        lx = (RNode[i + 1, 0] - RNode[i, 0]) / L
        mx = (RNode[i + 1, 1] - RNode[i, 1]) / L
        nx = (RNode[i + 1, 2] - RNode[i, 2]) / L

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
        '''尝试将对应角度的旋转矩阵变为单位矩阵，但是显然不行，转向时完全不作动作
                T[i] = np.array(
                    [[lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0,
                      0, 0, 0],
                     [mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0,
                      0, 0, 0, 0],
                     [nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                      0, 0, 0],
                     [0, 0, 0, 0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2),
                      -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])'''
    return T


# @jit(nopython=True)
# def AssembleStiffnessMatrix(k, T, gK):
#     # gK = np.zeros((node_number * 6, node_number * 6), dtype = np.float32)
#     for i in range(np.shape(k)[0]):
#         gK[6 * i:6 * i + 12, 6 * i:6 * i + 12] += np.dot(np.dot(T[i], k[i]), np.transpose(T[i]))
#
#     return gK


# @jit(nopython=True)
def StiffnessMatrix(RNode, gMaterial):
    #  计算单元刚度矩阵
    E = gMaterial[0, 0]
    I = gMaterial[0, 1]
    Jk = 2 * I
    G = E / 2
    A = gMaterial[0, 2]
    '''xi = gNode[gElement[ie][0] - 1][0]
    yi = gNode[gElement[ie][0] - 1][1]
    zi = gNode[gElement[ie][0] - 1][2]

    xj = gNode[gElement[ie][1] - 1][0]

    yj = gNode[gElement[ie][1] - 1][1]

    zj = gNode[gElement[ie][1] - 1][2]

    L = ((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2) ** (1 / 2)'''
    L1 = np.zeros(np.shape(RNode)[0] - 1, dtype=np.float32)
    k = np.zeros((np.shape(RNode)[0] - 1, 12, 12))
    for i in range(np.shape(RNode)[0] - 1):
        L1[i] = np.linalg.norm(RNode[i + 1] - RNode[i])
        recalculate_k = -1
        for j in range(i):
            if L1[j] == L1[i]:
                recalculate_k = j
        if recalculate_k == -1:
            L = L1[i]
            k[i] = np.array([[A * E / L, 0, 0, 0, 0, 0, - A * E / L, 0, 0, 0, 0, 0],
                             [0, 12 * E * I / L ** 3, 0, 0, 0, 6 * E * I / L ** 2, 0, -12 * E * I / L ** 3, 0, 0, 0,
                              6 * E * I / L ** 2],
                             [0, 0, 12 * E * I / L ** 3, 0, - 6 * E * I / L ** 2, 0, 0, 0, -12 * E * I / L ** 3, 0,
                              - 6 * E * I / L ** 2, 0],
                             [0, 0, 0, Jk / L * G, 0, 0, 0, 0, 0, -Jk / L * G, 0, 0],
                             [0, 0, - 6 * E * I / L ** 2, 0, 4 * E * I / L, 0, 0, 0, 6 * E * I / L ** 2, 0, 2 * E * I / L,
                              0],
                             [0, 6 * E * I / L ** 2, 0, 0, 0, 4 * E * I / L, 0, -6 * E * I / L ** 2, 0, 0, 0,
                              2 * E * I / L],
                             [- A * E / L, 0, 0, 0, 0, 0, A * E / L, 0, 0, 0, 0, 0],
                             [0, -12 * E * I / L ** 3, 0, 0, 0, -6 * E * I / L ** 2, 0, 12 * E * I / L ** 3, 0, 0, 0,
                              -6 * E * I / L ** 2],
                             [0, 0, -12 * E * I / L ** 3, 0, 6 * E * I / L ** 2, 0, 0, 0, 12 * E * I / L ** 3, 0,
                              6 * E * I / L ** 2, 0],
                             [0, 0, 0, -Jk / L * G, 0, 0, 0, 0, 0, Jk / L * G, 0, 0],
                             [0, 0, - 6 * E * I / L ** 2, 0, 2 * E * I / L, 0, 0, 0, 6 * E * I / L ** 2, 0, 4 * E * I / L,
                              0],
                             [0, 6 * E * I / L ** 2, 0, 0, 0, 2 * E * I / L, 0, -6 * E * I / L ** 2, 0, 0, 0,
                              4 * E * I / L]])
        if recalculate_k > -1:
            k[i] = k[recalculate_k].copy()

    return k


# def resetgK(k0, self_l1):
#     k = k0.copy()
#     l1 = int(self_l1)
#     if l1 < k0.shape[0] - 10:
#         k[0:l1 - 5, :, :] = k0[0:l1 - 5, :, :] * 500
#         k[l1 - 5:l1, :, :] = k0[l1 - 5:l1, :, :] * 20
#         k[-10:, :, :] = k0[-10:, :, :] *2
#     elif l1 < k0.shape[0]:
#         k[0:l1 - 5, :, :] = k0[0:l1 - 5, :, :] * 500
#         k[l1 - 5:l1, :, :] = k0[l1 - 5:l1, :, :] * 20
#         k[l1:, :, :] = k0[l1:, :, :] *2
#     return k

def resetgK(k0, self_l1):
    k = k0.copy()
    l1 = int(self_l1)
    k[0:l1 - 5, :, :] = k0[0:l1 - 5, :, :] * 5
    k[l1 - 5:l1, :, :] = k0[l1 - 5:l1, :, :] * 2
    k[l1:, :, :] = k0[l1:, :, :]
    return k

