import numpy as np
import os
import sys
import time
import scipy.sparse as sp
import scipy.linalg as linalg
from qpsolvers import solve_qp
import cvxopt
from cvxopt import matrix, solvers
from py_osqp import solve_py_osqp

if __name__ == '__main__':
    P = np.load('../data/osqp_data/P.npy')
    f = np.load('../data/osqp_data/f.npy')
    A = np.load('../data/osqp_data/A.npy')
    b = np.load('../data/osqp_data/b.npy')
    Aeq = np.load('../data/osqp_data/Aeq.npy')
    beq = np.load('../data/osqp_data/beq.npy')
    print('P:', P)
    print('f:', f)
    print('A:', A)
    print('b:', b)
    print('Aeq:', Aeq)
    print('beq:', beq)
    a = solve_qp(sp.csc_matrix(np.dot(np.transpose(P), P)),
                 np.dot(np.transpose(P), f),
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp', verbose=True)  # 可用算法：osqp, cvxopt
    print('a=',a)


    d = solve_qp(sp.csc_matrix(0.5*(P+P.T)),
                 f,
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp', verbose=True)  # 可用算法：osqp, cvxopt
    print('d=',d)

    e = solve_py_osqp(sp.csc_matrix(P),
                 f,
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp', verbose=True)  # 可用算法：osqp, cvxopt
    print('e=', e)

    c = solve_qp(sp.csc_matrix(P),
                 f,
                 sp.csc_matrix(A),
                 b.flatten(),
                 sp.csc_matrix(Aeq),
                 beq.flatten(), solver='osqp', verbose=True)  # 可用算法：osqp, cvxopt
    print('c=',c)




