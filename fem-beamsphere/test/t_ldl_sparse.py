import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg, minres, LinearOperator
from scipy.sparse.linalg import splu, factorized
import time


def solve_with_cholesky(A, b, regularization=1e-12):
    """
    使用Cholesky分解求解 Ax = b
    通过添加小正则化项处理半正定性
    """
    n = A.shape[0]
    # 添加正则化项确保正定性
    A_reg = A + regularization * sp.eye(n, format=A.format)

    # 进行Cholesky分解并求解
    solve = factorized(A_reg.tocsc())
    return solve(b)


def solve_with_ldlt(A, b, regularization=1e-12):
    """
    使用LDLT分解求解 Ax = b
    更适合处理半正定矩阵
    """
    n = A.shape[0]
    # 添加正则化项
    A_reg = A# + regularization * sp.eye(n, format=A.format)

    # 使用稀疏LU分解（设置对称模式近似LDLT）
    lu = splu(A_reg.tocsc(), permc_spec='NATURAL',
              diag_pivot_thresh=0, options=dict(SymmetricMode=True))
    return lu.solve(b)


def solve_with_cg(A, b, max_iter=1000, tol=1e-8, preconditioner=None):
    """
    使用共轭梯度法求解 Ax = b
    适合大规模稀疏问题
    """
    if preconditioner is None:
        # 使用对角预处理
        M = sp.diags(1.0 / A.diagonal()) if A.format == 'dia' else \
            sp.diags(1.0 / A.diagonal())
    else:
        M = preconditioner

    x, info = cg(A, b, M=M,rtol=tol, maxiter=max_iter)
    if info != 0:
        print(f"CG did not converge: info = {info}")
    return x


def solve_with_minres(A, b, max_iter=1000, tol=1e-8):
    """
    使用MINRES方法求解 Ax = b
    适合对称不定或半正定矩阵
    """
    x, info = minres(A, b, rtol=tol, maxiter=max_iter)
    if info != 0:
        print(f"MINRES did not converge: info = {info}")
    return x


def create_test_problem(n=1000, density=0.01, rank_deficit=0.1):
    """
    创建测试问题：稀疏对称半正定矩阵和右端向量
    rank_deficit: 秩亏比例，控制零特征值的数量
    """
    # 生成随机稀疏矩阵
    A = sp.random(n, n, density=density, format='csc')
    A = A @ A.T  # 使其对称正定

    # 引入秩亏（使矩阵半正定）
    num_zero_eigenvalues = int(n * rank_deficit)
    if num_zero_eigenvalues > 0:
        # 通过减去部分特征值使矩阵半正定
        eigvals, eigvecs = np.linalg.eigh(A.toarray())
        # 将最小的num_zero_eigenvalues个特征值设为0
        eigvals[:num_zero_eigenvalues] = 0
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T
        A = sp.csc_matrix(A)

    # 生成解向量和右端项
    x_true = np.random.randn(n)
    b = A @ x_true

    return A, b, x_true


def compare_solvers(A, b, x_true=None):
    """
    比较不同求解器的性能和精度
    """
    methods = {
        'Cholesky': solve_with_cholesky,
        'LDLT': solve_with_ldlt,
        'CG': solve_with_cg,
        'MINRES': solve_with_minres
    }

    results = {}

    for name, solver in methods.items():
        try:
            start_time = time.time()
            x = solver(A, b)
            solve_time = time.time() - start_time

            # 计算残差和误差
            residual = np.linalg.norm(A @ x - b)

            if x_true is not None:
                error = np.linalg.norm(x - x_true)
            else:
                error = None

            results[name] = {
                'time': solve_time,
                'residual': residual,
                'error': error,
                'solution': x
            }

            print(f"{name}: time={solve_time:.4f}s, residual={residual:.2e}" +
                  (f", error={error:.2e}" if error is not None else ""))

        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = None

    return results


# 示例使用
if __name__ == "__main__":
    # 创建测试问题
    n = 2000  # 矩阵大小
    density = 0.0005  # 稀疏度
    rank_deficit = 0.05  # 5%的零特征值

    print("创建测试问题...")
    A, b, x_true = create_test_problem(n, density, rank_deficit)

    print(f"矩阵维度: {A.shape}")
    print(f"非零元素数量: {A.nnz}")
    print(f"矩阵密度: {A.nnz / (n * n):.4f}")

    # 比较不同求解器
    print("\n比较求解器性能:")
    results = compare_solvers(A, b, x_true)

    # 选择最佳方法（基于残差）
    best_method = min(
        [(name, result['residual']) for name, result in results.items() if result is not None],
        key=lambda x: x[1]
    )[0]

    print(f"\n最佳方法: {best_method} (残差: {results[best_method]['residual']:.2e})")