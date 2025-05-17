import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_implicit(h=0.1, points=11, T=1):
    tau = h ** 2 / 2
    time_steps = int(T / tau)
    x = np.linspace(0, 1, points)
    t = np.linspace(0, T, time_steps + 1)

    print(f"\nШаг h={h}, точек={points}\n"
          f"Шаг по времени tau={tau:.4f}, шагов={time_steps}\n")

    X, T_mesh = np.meshgrid(x, t)
    U = np.zeros_like(X)

    U[0, :] = X[0, :] ** 2  # u(x,0) = x²
    U[:, 0] = T_mesh[:, 0]  # u(0,t) = t
    U[:, -1] = T_mesh[:, -1] + 1  # u(1,t) = t + 1

    for n in range(time_steps):
        # Коэффициенты для матрицы
        alpha = -tau * t[n + 1] / (2 * h ** 2)
        beta = 1 + tau * t[n + 1] / h ** 2
        gamma = -tau * t[n + 1] / (2 * h ** 2)
        delta = tau * x[1:-1] / (4 * h)

        main_diag = beta * np.ones(points - 2)
        lower_diag = (alpha - delta[:-1]) * np.ones(points - 3)
        upper_diag = (gamma + delta[1:]) * np.ones(points - 3)

        diagonals = [lower_diag, main_diag, upper_diag]
        A = diags(diagonals, [-1, 0, 1], format='csc')

        rhs = np.zeros(points - 2)
        for i in range(1, points - 1):
            d2u_old = (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1]) / h ** 2
            du_old = (U[n, i + 1] - U[n, i - 1]) / (2 * h)
            explicit_part = U[n, i] + tau / 2 * (t[n] * d2u_old + x[i] * du_old + x[i] * t[n])

            d2u_new = (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1]) / h ** 2  # начальное приближение
            du_new = (U[n, i + 1] - U[n, i - 1]) / (2 * h)
            implicit_part = tau / 2 * (t[n + 1] * d2u_new + x[i] * du_new + x[i] * t[n + 1])

            rhs[i - 1] = explicit_part + implicit_part

        # Учет граничных условий
        rhs[0] -= (alpha - delta[0]) * U[n + 1, 0]
        rhs[-1] -= (gamma + delta[-1]) * U[n + 1, -1]

        U[n + 1, 1:-1] = spsolve(A, rhs)

    return X, T_mesh, U


def plot_solution(X, T, U):
    plt.figure(figsize=(10, 6))

    im = plt.imshow(U, cmap="viridis", aspect='auto', origin='lower',
                    extent=[X.min(), X.max(), T.min(), T.max()])

    contours = plt.contour(X, T, U, colors='white', linewidths=0.5, levels=10)
    plt.clabel(contours, inline=True, fontsize=8)

    plt.colorbar(im, label='u(x,t)')
    plt.xlabel("x"), plt.ylabel("t")
    plt.title(f"Неявная схема ({X.shape[1]} узлов)")
    plt.show()


for h, points in [(0.1, 11), (0.05, 51), (0.01, 101)]:
    X, T, U = solve_implicit(h=h, points=points)
    plot_solution(X, T, U)
