import numpy as np
import matplotlib.pyplot as plt


def medhod_progonki(a, b, c, d):
    n = len(d)
    alpha = np.zeros(n - 1)
    beta = np.zeros(n)

    # Прямой ход
    alpha[0] = c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * alpha[i - 1]
        alpha[i] = c[i] / denom
        beta[i] = (d[i] - a[i - 1] * beta[i - 1]) / denom

    beta[n - 1] = (d[n - 1] - a[n - 2] * beta[n - 2]) / (b[n - 1] - a[n - 2] * alpha[n - 2])

    # Обратный ход
    x = np.zeros(n)
    x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = beta[i] - alpha[i] * x[i + 1]

    return x


def solve_implicit(h=0.1, points=11, T=1):
    tau = h ** 2 / 2
    time_steps = int(T / tau)
    x = np.linspace(0, 1, points)
    t = np.linspace(0, T, time_steps + 1)

    print(f"\nШаг h={h}, точек={points}\n"
          f"Шаг по времени tau={tau:.4f}, шагов={time_steps}\n")

    X, T_grid = np.meshgrid(x, t)
    U = np.zeros_like(X)

    U[0, :] = X[0, :] ** 2                      # u(x,0) = x²
    U[:, 0] = T_grid[:, 0]                      # u(0,t) = t
    U[:, -1] = T_grid[:, -1] + 1                # u(1,t) = t + 1

    for n in range(time_steps):
        alpha = -tau * t[n + 1] / (2 * h ** 2)
        beta = 1 + tau * t[n + 1] / h ** 2
        gamma = -tau * t[n + 1] / (2 * h ** 2)
        delta = tau * x[1:-1] / (4 * h)

        m = points - 2                              # Количество внутренних точек

        A = (alpha - delta[:-1]) * np.ones(m - 1)   # Нижняя
        B = beta * np.ones(m)                       # Главная
        C = (gamma + delta[1:]) * np.ones(m - 1)    # Верхняя
        D = np.zeros(m)                             # Правая часть

        for i in range(1, points - 1):
            d2u_old = (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1]) / h ** 2
            du_old = (U[n, i + 1] - U[n, i - 1]) / (2 * h)
            explicit_part = U[n, i] + tau / 2 * (t[n] * d2u_old + x[i] * du_old + x[i] * t[n])

            d2u_new = (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1]) / h ** 2  # начальное приближение
            du_new = (U[n, i + 1] - U[n, i - 1]) / (2 * h)
            implicit_part = tau / 2 * (t[n + 1] * d2u_new + x[i] * du_new + x[i] * t[n + 1])

            D[i - 1] = explicit_part + implicit_part

        D[0] -= (alpha - delta[0]) * U[n + 1, 0]
        D[-1] -= (gamma + delta[-1]) * U[n + 1, -1]

        U[n + 1, 1:-1] = medhod_progonki(A, B, C, D)

    return X, T_grid, U


def ploting(X, T, U):
    plt.figure(figsize=(10, 6))

    im = plt.imshow(U, cmap="viridis", aspect='auto', origin='lower',
                    extent=[X.min(), X.max(), T.min(), T.max()])

    contours = plt.contour(X, T, U, colors='white', linewidths=0.5, levels=10)
    plt.clabel(contours, inline=True, fontsize=8)

    plt.colorbar(im, label='u(x,t)')
    plt.xlabel("x"), plt.ylabel("t")
    plt.title(f"Неявная схема с методом прогонки ({X.shape[1]} узлов)")
    plt.show()


for h, points in [(0.1, 11), (0.05, 21), (0.01, 101)]:
    X, T, U = solve_implicit(h=h, points=points)
    ploting(X, T, U)
