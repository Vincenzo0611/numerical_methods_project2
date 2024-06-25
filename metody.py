import numpy as np

def jacobi(A, b, x0, tol=1e-9, max_iter=50):
    n = len(b)
    x = x0.copy()
    x_prev = x0.copy()
    iterations = 0
    residuum_norm_all = []

    while True:
        iterations += 1
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x_prev[j]
            x[i] = (b[i] - sigma) / A[i][i]
        residuum_norm = residuum(A, x, b)
        residuum_norm_all.append(residuum_norm)
        if residuum_norm < tol or iterations >= max_iter:
            break

        x_prev = x.copy()

    return x, residuum_norm_all


def gauss_seidel(A, b, x0, tol=1e-9, max_iter=50):
    n = len(b)
    x = x0.copy()
    iterations = 0
    residuum_norm_all = []

    while True:
        iterations += 1
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        residuum_norm = residuum(A, x, b)
        residuum_norm_all.append(residuum_norm)
        if residuum_norm < tol or iterations >= max_iter:
            break

    return x, residuum_norm_all

def residuum(A, x, b):
    return np.linalg.norm(np.dot(A, x) - b)

def lu_factorization(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for j in range(n):
        L[j][j] = 1.0

        for i in range(j + 1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - s1

        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (A[i][j] - s2) / U[j][j]

    return L, U


def lu_solve(A, b):
    n = len(b)
    y = [0.0] * n
    x = [0.0] * n
    L, U = lu_factorization(A)

    # Solve Ly = b
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

    # Solve Ux = y
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

    return x
