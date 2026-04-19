from typing import Optional
import numpy as np
from numpy import eye, hstack, ones, ndarray, vstack, zeros
import osqppurepy
from osqppurepy import OSQP
from scipy import sparse
from warnings import warn


def solve_py_osqp(P, q, G, h, A, b, sym_proj=False, verbose=False, eps_abs=1e-4, eps_rel=1e-4, polish=True, **kwargs):
    if sym_proj:
        P = 0.5 * (P + P.transpose())
    if isinstance(A, ndarray) and A.ndim == 1:
        A = A.reshape((1, A.shape[0]))
    if isinstance(G, ndarray) and G.ndim == 1:
        G = G.reshape((1, G.shape[0]))
    kwargs["verbose"] = verbose

    if isinstance(P, ndarray):
        P = sparse.csc_matrix(P)
    solver = OSQP()
    kwargs.update({
        "eps_abs": eps_abs,
        "eps_rel": eps_rel,
        "polish": polish,
        "verbose": verbose,
    })
    if A is not None and b is not None:
        if isinstance(A, ndarray):
            A = sparse.csc_matrix(A)
        if G is not None and h is not None:
            l_inf = -np.inf * ones(len(h))
            qp_A = sparse.vstack([G, A], format="csc")
            qp_l = hstack([l_inf, b])
            qp_u = hstack([h, b])
            solver.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, **kwargs)
        else:  # no inequality constraint
            solver.setup(P=P, q=q, A=A, l=b, u=b, **kwargs)
    elif G is not None and h is not None:
        if isinstance(G, ndarray):
            G = sparse.csc_matrix(G)
        l_inf = -np.inf * ones(len(h))
        solver.setup(P=P, q=q, A=G, l=l_inf, u=h, **kwargs)
    else:  # no inequality nor equality constraint
        solver.setup(P=P, q=q, **kwargs)
    # if initvals is not None:
    #     solver.warm_start(x=initvals)
    res = solver.solve()
    if hasattr(solver, "constant"):
        success_status = solver.constant("OSQP_SOLVED")
    else:  # more recent versions of OSQP
        success_status = solver.constant("OSQP_SOLVED")
    if res.info.status_val != success_status:
        warn(f"OSQP exited with status '{res.info.status}'")
        return None
    return res.x, res.y
