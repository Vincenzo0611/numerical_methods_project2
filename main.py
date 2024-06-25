from metody import *
import matplotlib.pyplot as plt
import time
indeks = 193218

N = 900 + int(str(indeks)[4]) * 10 + int(str(indeks)[5])
a1 = 5 + int(str(indeks)[3])
a2 = -1
a3 = -1

A = np.zeros((N, N))

np.fill_diagonal(A, a1)

np.fill_diagonal(A[1:], a2)
np.fill_diagonal(A[:, 1:], a2)

np.fill_diagonal(A[2:], a3)
np.fill_diagonal(A[:, 2:], a3)

f = int(str(indeks)[2])
b = np.array([np.sin(n * (f + 1)) for n in range(N)])

x0 = np.zeros(N)

start_time = time.time()
_, residuum_norm_j = jacobi(A, b, x0)
jacobi_time = time.time() - start_time

start_time = time.time()
_, residuum_norm_gs = gauss_seidel(A, b, x0)
gauss_seidel_time = time.time() - start_time

print("Czas trwania metody Jacobiego: {:.4f} sekund".format(jacobi_time))
print("Czas trwania metody Gaussa-Seidla: {:.4f} sekund".format(gauss_seidel_time))

plt.figure(figsize=(10, 6))
plt.semilogy(residuum_norm_j, label='Metoda Jacobiego')
plt.semilogy(residuum_norm_gs, label='Metoda Gaussa-Seidla')
plt.xlabel('Numer iteracji')
plt.ylabel('Norma residuum')
plt.title('Zmiany normy residuum w kolejnych iteracjach')
plt.legend()
plt.grid(True)
plt.show()

del residuum_norm_j
del residuum_norm_gs

a1 = 3
a2 = -1
a3 = -1

A = np.zeros((N, N))

np.fill_diagonal(A, a1)

np.fill_diagonal(A[1:], a2)
np.fill_diagonal(A[:, 1:], a2)

np.fill_diagonal(A[2:], a3)
np.fill_diagonal(A[:, 2:], a3)

x0 = np.zeros(N)

_, residuum_norm_j = jacobi(A, b, x0)
_, residuum_norm_gs = gauss_seidel(A, b, x0)

plt.figure(figsize=(10, 6))
plt.semilogy(residuum_norm_j, label='Metoda Jacobiego')
plt.semilogy(residuum_norm_gs, label='Metoda Gaussa-Seidla')
plt.xlabel('Numer iteracji')
plt.ylabel('Norma residuum')
plt.title('Zmiany normy residuum w kolejnych iteracjach')
plt.legend()
plt.grid(True)
plt.show()

del residuum_norm_j
del residuum_norm_gs


x_LU = lu_solve(A, b)
residuum_norm_LU = residuum(A, x_LU, b)

print(residuum_norm_LU)

del x_LU
del residuum_norm_LU

Ns = [100, 500, 1000, 2000, 3000]

times_j = []
times_gs = []
times_lu = []

a1 = 5 + int(str(indeks)[3])
a2 = -1
a3 = -1
f = int(str(indeks)[2])
for N in Ns:
    A = np.zeros((N, N))
    np.fill_diagonal(A, a1)

    np.fill_diagonal(A[1:], a2)
    np.fill_diagonal(A[:, 1:], a2)

    np.fill_diagonal(A[2:], a3)
    np.fill_diagonal(A[:, 2:], a3)
    b = np.array([np.sin(n * (f + 1)) for n in range(N)])

    x0 = np.zeros(N)

    start_time = time.time()
    _, _ = jacobi(A, x0, b)
    j_time = time.time() - start_time
    times_j.append(j_time)

    start_time = time.time()
    _, _ = gauss_seidel(A, x0, b)
    gs_time = time.time() - start_time
    times_gs.append(gs_time)

    start_time = time.time()
    _ = lu_solve(A, b)
    lu_time = time.time() - start_time
    times_lu.append(lu_time)

plt.figure(figsize=(10, 6))
plt.plot(Ns, times_j, label='Metoda Jacobiego')
plt.plot(Ns, times_gs, label='Metoda Gaussa-Seidla')
plt.plot(Ns, times_lu, label='Faktoryzacja LU')
plt.xlabel('Liczba niewiadomych (N)')
plt.ylabel('Czas [s]')
plt.title('Czas wyznaczania rozwiązania dla różnych metod')
plt.legend()
plt.grid(True)
plt.show()
