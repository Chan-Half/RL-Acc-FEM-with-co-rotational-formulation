"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:  fem_numerical_integration_scheme.py
#  @brief:     fem numerical integration scheme
#  @author:    Hao Chen
#  @version:   1.3
#  @date:      2023.07.03
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
import math
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as linalg
from qpsolvers import solve_qp
# from py_osqp import solve_py_osqp
import rlqp
import osqp

def implicit_integration(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time):
    """
    # osqp implicit_integration
    (M + h*D + K*h^2)*dv(t+1) = h(f1 + gravity - (h*K*v+D*dv))
    Returns: None
    """
    G = M + delta_h_time ** 2 * gK + delta_h_time * D
    dv = solve_qp(sp.csc_matrix(np.dot(np.transpose(G), G)),
                  np.dot(np.transpose(G), delta_h_time * (F - f1 - gravity)), sp.csc_matrix(A),
                  b.flatten(),
                  sp.csc_matrix(Aeq),
                  beq.flatten(), solver='osqp')  # sym_proj=True这个参数是自动把Hessian矩阵对称
    model.update_state_implicit_integration(dv)
    # osqp
    '''D = 0.1 * self.model.M  # + 0.01 * self.model.gK  # 把这个Ｋ加进去以后，模型就出问题，阻尼矩阵似乎不能学瑞丽阻尼设置
    G = self.model.M + self.delta_h_time ** 2 * self.model.gK + self.delta_h_time * D
    dv = solve_qp(sp.csc_matrix(np.dot(np.transpose(G), G)), np.dot(np.transpose(G), self.delta_h_time*(self.F -self.f1 - self.gravity)), sp.csc_matrix(self.A),
                             self.b.flatten(),
                             sp.csc_matrix(self.Aeq),
                             self.beq.flatten(), solver='osqp')   # sym_proj=True这个参数是自动把Hessian矩阵对称
    self.model.update_state_implicit_integration(dv)'''
    '''dv = solve_qp(sp.csc_matrix(G),
                                  np.array(self.delta_h_time * (self.F - self.f1 - self.gravity)),
                                  sp.csc_matrix(self.A),
                                  self.b.flatten(),
                                  sp.csc_matrix(self.Aeq),
                                  self.beq.flatten(), solver='osqp')'''  # sym_proj=True最后这个参数是自动把Hessian矩阵对称
    # cvxopt
    '''P = matrix(np.array(self.model.M + self.delta_h_time**2 * self.model.gK, dtype=np.float64))
    q = matrix(np.array(self.delta_h_time*(self.F - self.f1 - self.gravity), dtype=np.float64))
    # , kktsolver='ldl', options={'kktreg':1e-9}
    dv = solvers.qp(P, q, matrix(np.float64(self.A)), matrix(np.float64(self.b)), matrix(np.float64(self.Aeq)), matrix(np.float64(self.beq)))['x']
    dv = np.array(dv, dtype=np.float32).transpose()[0]'''
    # np.linalg.solve
    '''P = np.vstack((G, self.Aeq))
    f = self.delta_h_time*(self.F - self.f1 - self.gravity)
    q = np.hstack((f, self.beq.transpose()[0]))
    print(P.shape)
    print(q.shape)
    dv = np.linalg.solve(P, q)'''


def newmark_beta_integration(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time):
    """
    # osqp newmark_beta_integration
    Returns: None
    """
    alpha = -0.05  # 这个alpha和HHT中的概念相同
    beta = 0.25 * (1 - alpha) ** 2
    gamma = 0.5 * (1 - 2 * alpha)
    P = M + gK * beta * delta_h_time ** 2
    a = solve_qp(sp.csc_matrix(np.dot(np.transpose(P), P)),
                 np.dot(np.transpose(P), F - f1 - gravity),
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp')

    model.update_state_newmark_beta_integration(a, gamma, beta)


def hht_alpha_integration(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time):
    """
    # osqp hht_alpha_integration
    使用a作为未知数的HHT方法
    Returns: None
    """
    # 注意这里的alpha和newmark-beta中的alpha不同
    t1 = time.time()
    alpha = -0.3
    beta = 0.25 * (1 - alpha)**2
    gamma = 0.5 * (1 - 2*alpha)
    P = M + D * (1+alpha) * gamma * delta_h_time + gK * (1+alpha) * beta * delta_h_time ** 2

    a = solve_qp(sp.csc_matrix(np.dot(np.transpose(P), P)),
                 np.dot(np.transpose(P), F - f1 - gravity),
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp', verbose=True)
    if a is None:
        np.save('./data/osqp_data/P.npy', P)
        np.save('./data/osqp_data/f.npy', F - f1 - gravity)
        np.save('./data/osqp_data/A.npy', A)
        np.save('./data/osqp_data/b.npy', b)
        np.save('./data/osqp_data/Aeq.npy', Aeq)
        np.save('./data/osqp_data/beq.npy', beq)
        print('osqp 求解失败')

    t2 = time.time()
    model.update_state_hht_alpha_integration(a, gamma, beta)
    print('solve', t2-t1)
    print('corotate-update', time.time()-t2)
    return alpha


def hht_alpha_integration_x(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time, q, fc, tttt):
    """
    # osqp hht_alpha_integration_x
    使用dx作为未知数的HHT方法
    Returns: None
    """
    # 注意这里的alpha和newmark-beta中的alpha不同
    t1 = time.time()
    alpha = -0.3
    beta = 0.25 * (1 - alpha)**2
    gamma = 0.5 * (1 - 2*alpha)
    P = M / (beta * delta_h_time**2) + (1+alpha) * gamma / (beta * delta_h_time) * D + gK * (1+alpha)
    # 注意y是对偶变量，不等式约束对应对偶变量排在前面，等式约束对应对偶变量排在后面
    dx, y = solve_qp(0.5*(P+P.T),
                 q + F - f1 - gravity - fc,
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp', verbose=False)

    tttt[3] = time.time()
    if dx is None:
        np.save('./data/osqp_data/P.npy', P)
        np.save('./data/osqp_data/f.npy', q + F - f1 - gravity - fc)
        np.save('./data/osqp_data/A.npy', A)
        np.save('./data/osqp_data/b.npy', b)
        np.save('./data/osqp_data/Aeq.npy', Aeq)
        np.save('./data/osqp_data/beq.npy', beq)
        print('osqp 求解失败')

    t2 = time.time()
    model.update_state_hht_alpha_integration_x(dx, gamma, beta)
    tttt[4] = time.time()
    print('solve', t2-t1)
    print('corotate-update', time.time()-t2)
    return alpha, beta, gamma, y, tttt


def implicit_alpha_integration(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time):
    """
    # osqp implicit_integration
    (M + h*D + K*h^2)*dv(t+1) = h(f1 + gravity - (h*K*v+D*dv))
    Returns: None
    """
    alpha = -0.05
    G = M + (1+alpha) * delta_h_time ** 2 * gK + (1+alpha) * delta_h_time * D
    dv = solve_qp(sp.csc_matrix(np.dot(np.transpose(G), G)),
                  np.dot(np.transpose(G), delta_h_time * (F - f1 - gravity)), sp.csc_matrix(A),
                  b.flatten(),
                  sp.csc_matrix(Aeq),
                  beq.flatten(), solver='osqp')  # sym_proj=True这个参数是自动把Hessian矩阵对称
    model.update_state_implicit_integration(dv)
    return alpha


def implicit_beta_integration(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time):
    """
    # osqp implicit_integration
    (M + h*D + K*h^2)*dv(t+1) = h(f1 + gravity - (h*K*v+D*dv))
    Returns: None
    """
    beta = -0.2
    G = M + (1+beta) * delta_h_time ** 2 * gK + (1+beta) * delta_h_time * D
    dv = solve_qp(sp.csc_matrix(np.dot(np.transpose(G), G)),
                  np.dot(np.transpose(G), delta_h_time * (F - f1 - gravity)), sp.csc_matrix(A),
                  b.flatten(),
                  sp.csc_matrix(Aeq),
                  beq.flatten(), solver='osqp')  # sym_proj=True这个参数是自动把Hessian矩阵对称
    model.update_state_implicit_integration(dv)
    return beta


def newmark_beta_integration_x(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time):
    """
    # osqp newmark_beta_integration_x
    Returns: None
    """

    beta = 0.25
    gamma = 0.5
    P = M / beta / delta_h_time ** 2 + gK
    x = solve_qp(sp.csc_matrix(np.dot(np.transpose(P), P)),
                 np.dot(np.transpose(P), F - f1 - gravity),
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp')

    model.update_state_newmark_beta_integration_x(x, gamma, beta)
    return beta


def py_osqp_dynamic_x(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time, q, fc, alpha):
    """
    # osqp hht_alpha_integration_x
    使用dx作为未知数的HHT方法
    Returns: None
    """

    # 注意这里的alpha和newmark-beta中的alpha不同
    t1 = time.time()
    # alpha = 0.0
    beta = 0.25 * (1 - alpha)**2
    gamma = 0.5 * (1 - 2*alpha)
    P = M / (beta * delta_h_time**2) + (1+alpha) * gamma / (beta * delta_h_time) * D + gK * (1+alpha)

    # 转制后乘以自己会使得算出的结果出问题
    dx, y = solve_py_osqp(sp.csc_matrix(P/(1+alpha)), (q + F - f1 - gravity - fc)/(1+alpha), sp.csc_matrix(A), b.flatten(), sp.csc_matrix(Aeq), beq.flatten(), verbose=False)
    # r1 = np.dot(A, dx)-b
    # r2 = np.dot(Aeq, dx) - beq
    # r3 = np.dot(P/(1+alpha), dx) + (q + F - f1 - gravity - fc)/(1+alpha) + np.dot(A.transpose(), y[:A.shape[0]]) + np.dot(Aeq.transpose(), y[A.shape[0]:])
    # for i in r1:
    #     for j in i:
    #         if j > 1e-3:
    #             print('inequality unfeasible!')
    #             print(r1)
    #             print(j)
    #             break
    # for i in r2:
    #     for j in i:
    #         if j > 1e-3 or j < -1e-3:
    #             print('equality unfeasible!')
    #             print(r2)
    #             print(j)
    #             break
    # for j in r3:
    #     if j > 1e-3 or j < -1e-3:
    #         print('force residual is too large!')
    #         print(r3)
    #         print(j)
    #         break

    t2 = time.time()
    model.update_state_hht_alpha_integration_x(dx, gamma, beta)
    print('solve', t2-t1)
    print('corotate-update', time.time()-t2)
    return alpha, beta, gamma, y



# model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time, q, fc, tttt, alpha
def nnqp_hht(model, M, D, gK, F, f1, gravity, A, b, Aeq, beq, delta_h_time, q, fc, tttt, alpha):
    """
    # nnqp_static
    使用dx作为未知数的HHT方法
    python osqp直接迭代
    准静态方法
    Returns: None
    """

    # t1 = time.time()
    # P = sp.csc_matrix(gK)

    if A is not None:
        m = A.shape[0]
    else:
        m = 0

    beta = 0.25 * (1 - alpha) ** 2
    gamma = 0.5 * (1 - 2 * alpha)
    P = M / (beta * delta_h_time ** 2) + (1 + alpha) * gamma / (beta * delta_h_time) * D + gK * (1 + alpha)
    P = sp.csc_matrix(P)


    # 设置我们的求解器
    opts = {
        'verbose': True,
        'eps_abs': 1e-4,
        'eps_rel': 1e-4,
        'max_iter': 10000,
        'rho': 0.1,
        # 'sigma': 0.01,
        # 'linsys_solver': 'qdldl',
        # 'linsys_solver':'mkl pardiso',
        'linsys_solver': 'spslu',
        # 'adaptive_sigma': 'vector_policy',
        # 'adaptive_sigma_policy': '/home/cairch/Downloads/rlfem_train/benchmarks_full_777_traced.pt',
        # 'adaptive_sigma_policy':'/home/cairch/chenhao/project/FEM/FEM2025/exp_RAL_2025/data/trained_model/single_sigma_update.pt',
        'adaptive_rho': 'standard',
        'adaptive_sigma': 'disable',
        'polish': False,
        'check_termination': 1,
        'warm_start': False,
    }
    # solver = osqp.OSQP()
    solver = rlqp.RLQP()

    A = sp.csc_matrix(A)
    # b= b.flatten
    Aeq = sp.csc_matrix(Aeq)
    # beq = beq.flatten



    A_nnqp = None
    l_nnqp = None
    u_nnqp = None
    if A.shape[0] == 0 or b.shape == 0:
        A = None
        b = None
    if Aeq.shape[0] == 0 or beq.shape == 0:
        Aeq = None
        beq = None
    if A is not None and b is not None:
        A_nnqp = A
        l_nnqp = np.full(b.shape, -np.inf)
        u_nnqp = b


    if Aeq is not None and beq is not None:
        A_nnqp = Aeq if A_nnqp is None else sp.vstack([A_nnqp, Aeq], format="csc")
        l_nnqp = beq if l_nnqp is None else np.vstack([l_nnqp, beq])
        u_nnqp = beq if u_nnqp is None else np.vstack([u_nnqp, beq])
    t1 = time.time()




    try:
        solver.setup(P=P, q=q + F - f1 - gravity - fc, A=A_nnqp, l=l_nnqp, u=u_nnqp, **opts)
        t1 = time.time()
        res = solver.solve()
        dx = res.x
        y = res.y
    except:

        print('nnqp solve failed')
        dx = np.zeros_like(model.dx)
        y_shape = 0
        if A is not None:
            y_shape += A.shape[0]
        if Aeq is not None:
            y_shape += Aeq.shape[0]
        y = np.zeros(y_shape)



    # dx, y = solve_qp(gK.tocsc(), F - f1 - gravity, sp.csc_matrix(A), b.flatten(), sp.csc_matrix(Aeq),
    #                       beq.flatten(), verbose=True, solver='osqp')

    t2 = time.time()
    # if dx is None:
    #     print('the primal problem is infeasible')
    #     dx = np.zeros_like(model.dx)  # + 0.00000000001

    # # ------------------------------------------------------------ #
    # # !!!!!!!!!!!!!注意这里做了一个缩放，来自以前的方案，但这样受力就不再平衡了，随着计算再逐渐平衡!!!!!!!!!!!!
    # l_dx = np.linalg.norm(dx)
    # if l_dx > 1e-3:
    #     dx /= l_dx * 1e3
    #     y /= l_dx * 1e3



    model.update_state_hht_alpha_integration_x(dx, gamma, beta)


    # ==========================================================
    # 🌟 智能速度/加速度投影 (终极消除接触抖动)
    # ==========================================================
    # if m > 0:
    #     if A_nnqp is not None and y is not None:
    #         # 1. 提取质量矩阵的对角线 (极其关键：用于计算真实的物理冲量)
    #         if sp.issparse(model.M):
    #             M_diag = model.M.diagonal()
    #         else:
    #             M_diag = np.diag(model.M)
    #
    #         # 安全求逆，防止除零 (M_inv_diag 相当于每个自由度的 1/m)
    #         M_inv_diag = 1.0 / np.where(np.abs(M_diag) > 1e-12, M_diag, 1e-12)
    #
    #         # 2. 动态自适应阈值，过滤求解器微小噪声
    #         max_y = np.max(np.abs(y[:m]))
    #         active_tol = max(1e-3 * max_y, 1e-4)
    #         active_indices = np.where(np.abs(y[:m]) > active_tol)[0]
    #
    #         for idx in active_indices:
    #             # 提取单条接触约束的法向梯度 (不再做暴力的 norm 归一化)
    #             A_i = A_nnqp[idx, :].toarray().flatten()
    #
    #             # 计算接触面法向上的相对物理速度
    #             v_rel = np.dot(A_i, model.v)
    #
    #             # v_rel < 0 代表由于 HHT 算法的反噬，节点正试图“向外假弹”
    #             if v_rel < 0:
    #                 # 计算广义有效质量分母 (A * M^-1 * A^T)
    #                 denom = np.dot(A_i, A_i * M_inv_diag)
    #                 if denom > 1e-12:
    #                     # 计算完全非弹性碰撞的物理冲量 (Lagrange Multiplier for velocity)
    #                     impulse = -v_rel / denom
    #
    #                     # 将真实冲量作用于系统 (绝对不会污染旋转自由度！)
    #                     model.v += impulse * (A_i * M_inv_diag)
    #
    #             # 3. 对加速度做同等动量抹平，彻底掐断下一帧的震荡源
    #             a_rel = np.dot(A_i, model.a)
    #             if a_rel < 0:
    #                 denom = np.dot(A_i, A_i * M_inv_diag)
    #                 if denom > 1e-12:
    #                     impulse_a = -a_rel / denom
    #                     model.a += impulse_a * (A_i * M_inv_diag)




    print('solve', t2-t1)
    print('corotate-update', time.time()-t2)
    return alpha, beta, gamma, y, tttt

