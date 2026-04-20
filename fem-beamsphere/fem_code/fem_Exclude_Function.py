"""
###########################################################################################
#  @copyright: Institute of Automation, Chinese Academy of Sciences
#  @filename:  fem_cuda_mesh.py
#  @brief:     fem cuda function of corotate mesh
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2025.12.19
#  @Email:     chen.hao2020@ia.ac.cn
#  @Note:      Remember to free up the memory allocated by `new`, otherwise a memory leak will occur.
###########################################################################################
"""
import numpy as np
from numba import jit


def ldl_T(matrix):
    w = matrix.shape[0]
    L = np.zeros((w, w))
    for i in range(w):
        L[i, i] = 1
    D = np.zeros((w, w))
    for i in range(w):
        D[i, i] = matrix[i, i] - np.dot(np.dot(L[i, :i], D[:i, :i]), L[i, :i].T)
        for j in range(i+1, w):
            L[j, i] = (matrix[j, i] - np.dot(np.dot(L[j,:i],D[:i,:i]), L[i, :i].T))/D[i, i]
    return L, D


def AssembleStiffnessMatrix(gK, k, T):
    # self.gK[:,:] = 0 #  = np.zeros((node_number * 6, node_number * 6), dtype = np.float32)
    for i in range(np.shape(k)[0]):
        gK[6 * i:6 * i + 12, 6 * i:6 * i + 12] += np.dot(np.dot(T[i], k[i]), np.transpose(T[i]))


# @jit(nopython=True)
def pathnear(path0, xtriangle, x):
    bool_nearpath = np.zeros(0, dtype=np.int32)
    for j in range(np.shape(xtriangle)[0]):
        for i in path0:
            if np.linalg.norm(i - x[xtriangle[j][0]]) < 20:
                bool_nearpath = np.append(bool_nearpath, np.int32(j))
                break

    return bool_nearpath


def cutface(x, xtriangle):
    ff = np.shape(xtriangle)[0]
    j = 0
    while j < ff:
        if x[xtriangle[j][0]][2] <= 8 or x[xtriangle[j][1]][2] <= 8 or x[xtriangle[j][2]][2] <= 8:
            xtriangle = np.delete(xtriangle, j, axis=0)
            ff = np.shape(xtriangle)[0]
            continue
        j += 1
    return xtriangle


# 点，障碍点，目标点，斥力增益，引力增益，斥力距离阈值，引力距离阈值
# @jit(nopython=True)
def compute_f(point0, ref_all, goal0, k_ref, k_att, t_ref, t_att):
    # 点，障碍点，目标点，斥力增益，引力增益，斥力距离阈值，引力距离阈值
    # 加上引力
    lgoal = np.sqrt((point0[0] - goal0[0]) ** 2 + (point0[1] - goal0[1]) ** 2 + (point0[2] - goal0[2]) ** 2)
    f = np.zeros(3)  # 合力
    if lgoal <= t_att:
        f[0] += 0.5 * k_att * lgoal ** 2 * (goal0[0] - point0[0]) / lgoal
        f[1] += 0.5 * k_att * lgoal ** 2 * (goal0[1] - point0[1]) / lgoal
        f[2] += 0.5 * k_att * lgoal ** 2 * (goal0[2] - point0[2]) / lgoal
    else:
        f[0] += (t_att * k_att * lgoal - 0.5 * k_att * t_att ** 2) * (goal0[0] - point0[0]) / lgoal
        f[1] += (t_att * k_att * lgoal - 0.5 * k_att * t_att ** 2) * (goal0[1] - point0[1]) / lgoal
        f[2] += (t_att * k_att * lgoal - 0.5 * k_att * t_att ** 2) * (goal0[2] - point0[2]) / lgoal

    # 加上斥力
    for ref in ref_all:
        lref = np.sqrt((point0[0] - ref[0]) ** 2 + (point0[1] - ref[1]) ** 2 + (point0[2] - ref[2]) ** 2)
        if lref <= t_ref:
            f[0] += 0.5 * k_ref * (t_ref - lref) / (t_ref * lref ** 3) * (point0[0] - ref[0]) / lref
            f[1] += 0.5 * k_ref * (t_ref - lref) / (t_ref * lref ** 3) * (point0[1] - ref[1]) / lref
            f[2] += 0.5 * k_ref * (t_ref - lref) / (t_ref * lref ** 3) * (point0[2] - ref[2]) / lref
        else:
            f[0] += 0
            f[1] += 0
            f[2] += 0
    return f


def ar_potential_field(vertices):
    # 点，目标点，斥力增益，引力增益，斥力距离阈值，引力距离阈值
    # start0 = [36.5, 17.6, 2.0]
    # goal0 = [-35., -77., -140.]  # 该点可用
    # goal0 = [-35., -77., -160.]
    # goal0 = [37., -77., -150.]
    # goal0 = [82., 43., 107.]

    start0 = [166.5, 128.0, 7.0]
    goal0 = [134.8, 216.4, 249.8]

    k_ref = 1000
    k_att = 0.1
    t_ref = 4  # 该值应大于等于管半径
    t_att = 1
    path0 = []  # 路径
    step0 = 0.3  # 步长
    bool0 = False  # 返回是否到达目标附近
    thres_goal = 1  # 是否到目标附近的阈值
    path0.append(start0)
    p0 = [start0[0], start0[1], start0[2]]
    for i in range(400):
        # print(i)
        f_all = compute_f(np.array(p0), vertices, np.array(goal0), k_ref, k_att, t_ref, t_att)
        # print(f_all)
        p0[0] += step0 * f_all[0] / np.sqrt(f_all[0] ** 2 + f_all[1] ** 2 + f_all[2] ** 2)
        p0[1] += step0 * f_all[1] / np.sqrt(f_all[0] ** 2 + f_all[1] ** 2 + f_all[2] ** 2)
        p0[2] += step0 * f_all[2] / np.sqrt(f_all[0] ** 2 + f_all[1] ** 2 + f_all[2] ** 2)
        # print(p0)
        path0.append([p0[0], p0[1], p0[2]])

        # print(path0)
        if np.sqrt((p0[0] - goal0[0]) ** 2 + (p0[1] - goal0[1]) ** 2 + (p0[2] - goal0[2]) ** 2) <= thres_goal:
            bool0 = True
            path0.append(goal0)
            break
    '''if bool0 is False:
        path0.append(goal0)'''
    # path0.append(goal0)
    output = open('path.txt', 'w', encoding='gbk')
    for i in range(len(path0)):
        # output.write('@')
        for j in range(len(path0[i])):
            output.write(str(path0[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')  # 相当于Tab一下，换一个单元格
        output.write('\n')  # 写完一行立马换行
    output.close()

    return path0, bool0


# @jit(nopython=True)
def SolveModel_cpu(ResultNode, x, xtriangle, gNode, A1, b1, vNormal, distance0, bool_nearpath):
    #  求解约束模型
    #  输入参数:
    #     无
    #  返回值：
    #     无
    row_num = 600
    A = np.zeros((row_num, 6 * np.shape(gNode)[0]), dtype=np.float32)
    b = np.zeros((row_num, 1), dtype=np.float32)
    constraint_triangle = np.zeros((row_num, 1), dtype=np.int32)
    k = 0

    for j in bool_nearpath:
        if k > row_num - 20:
            row_num += 100
            A = np.row_stack((A, np.zeros((100, 6 * np.shape(gNode)[0]), dtype=np.float32)))
            b = np.row_stack((b, np.zeros((100, 1), dtype=np.float32)))
            constraint_triangle = np.row_stack((constraint_triangle, np.zeros((100, 1), dtype=np.int32)))

        pNode, distance1 = pPlane(ResultNode, vNormal[j], distance0[j][0])

        for i in range(np.shape(gNode)[0]):
            if distance1[i] < 1.5 and AreaMth(pNode[i], x, xtriangle, j):
                A[k, (6 * i):(6 * i + 3)] = vNormal[j]
                b[k] = distance0[j][0] - np.dot(vNormal[j], gNode[i])
                constraint_triangle[k] = j
                k += 1

    return A[0:k, :], b[0:k, :], A1[0:k, :], b1[0:k, :], constraint_triangle[0:k, :]


# @jit(nopython=True)
def AreaMth(P, x, xtriangle, j):
    # Compute vectors
    v0 = np.array([x[xtriangle[j][0]][0], x[xtriangle[j][0]][1], x[xtriangle[j][0]][2]]) - P
    v1 = np.array([x[xtriangle[j][1]][0], x[xtriangle[j][1]][1], x[xtriangle[j][1]][2]]) - P
    v2 = np.array([x[xtriangle[j][2]][0], x[xtriangle[j][2]][1], x[xtriangle[j][2]][2]]) - P

    theta1 = np.arccos(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    theta2 = np.arccos(np.dot(v2, v1) / (np.linalg.norm(v2) * np.linalg.norm(v1)))
    theta3 = np.arccos(np.dot(v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2)))

    if 2 * 3.1415926535 - (theta1 + theta2 + theta3) <= 3.5:
        return True
    else:
        return False


# @jit(nopython=True)
def pPlane(ResultNode, vNormal, distance0):
    # 求投影点及点到面的距离
    distance1 = np.abs(np.dot(ResultNode, vNormal) - distance0)
    vNormal = vNormal.reshape((1, len(vNormal)))
    pNode = ResultNode - np.dot((np.dot(vNormal, ResultNode.T) - distance0).T, vNormal)
    return pNode, distance1


'''
@jit(nopython=True)
def solve_qp_myself(G, H, A, b, Aeq, beq):
    delta1 = 1e5
    delta2 = 1e2
    x = np.zeros((np.shape(G)[0], 1))
    t = 1
    while 1:
        g = np.dot(A, x) - b
        delta_f = np.dot(G, x) + H + delta1 * np.dot(Aeq.T, (np.dot(Aeq, x) - beq))
        for i in range(np.shape(A)[0]):
            if g[i] > 0:
                # print(np.shape(delta_f))
                # print((np.dot(A[i], x) - b[i])[0])
                # print(np.shape(A[i].T))
                delta_f = delta_f + delta2 * (np.dot(A[i], x) - b[i])[0] * A[i].T.reshape((np.shape(A)[1], 1))
                # print(np.shape(delta_f))

        x = x - 0.001 * delta_f / t / (np.linalg.norm(delta_f) + 1)
        x[0:6, 0] = 0

        # / (np.linalg.norm(delta_f)+1)
        # print(np.linalg.norm(x))
        t += 1
        ff = np.linalg.norm(delta_f)
        # print(ff)

        if ff / t < 0.05 or t > 2000:
            print(np.linalg.norm(delta_f))
            break

    return x
'''


@jit(nopython=True)
def fa_vector(x, xtriangle, skeletonlist):
    lenx = np.shape(xtriangle)[0]
    # lensk = np.shape(skeletonlist)[0]
    vNormal = np.zeros((lenx, 3), dtype=np.float64)
    distance0 = np.zeros((lenx, 1), dtype=np.float64)
    # distancesk = np.zeros((lenx, 1), dtype=np.float32)
    # aaaa = np.zeros((lenx, 3), dtype=np.float32)
    for j in range(lenx):
        # sknum = 10000000
        c = np.cross(x[xtriangle[j][2]] - x[xtriangle[j][0]], x[xtriangle[j][1]] - x[xtriangle[j][0]])
        # 归一化
        norm = np.linalg.norm(c)
        vNormal[j] = c.copy() / norm
        distance0[j][0] = np.dot(vNormal[j], x[xtriangle[j][0]])
    '''    for i in range(lensk):
            laa = np.linalg.norm(skeletonlist[i] - (x[xtriangle[j][0]]+x[xtriangle[j][1]]+x[xtriangle[j][2]])/3)
            if laa < sknum:
                sknum = laa
                distancesk[j][0] = i
    sk = np.dot(vNormal, skeletonlist.T)
    for j in range(lenx):
        if sk[j][np.int(distancesk[j][0])] > distance0[j][0]:
            vNormal[j] = -vNormal[j]
            distance0[j] = -distance0[j]'''
    '''for j in range(lenx):
        for i in range(Adjx[j]):
            aaaa[j] += vNormal[int(adjTri[j, i])]
        if np.dot(aaaa[j], vNormal[j]) < 0:
            vNormal[j] = -vNormal[j]
            distance0[j] = -distance0[j]'''
    return vNormal, distance0


