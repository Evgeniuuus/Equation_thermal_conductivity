import numpy as np
import matplotlib.pyplot as plt


def solve(h=0.1, points=11, T=1):
    tau = h ** 2 / 2
    time_steps = int(T / tau)
    x = np.linspace(0, 1, points)
    t = np.linspace(0, T, time_steps + 1)

    print(f"\nШаг h={h}, точек={points}\n" f"Шаг по времени tau={tau:.4f}, шагов={time_steps}\n")

    X, T_mesh = np.meshgrid(x, t)
    U = np.zeros_like(X)

    U[0, :] = X[0, :] ** 2  # Начальное условие u(x,0) = x²
    U[:, 0] = T_mesh[:, 0]  # Левая граница u(0,t) = t
    U[:, -1] = T_mesh[:, -1] + 1  # Правая граница u(1,t) = t + 1

    for n in range(time_steps):
        for i in range(1, len(x) - 1):
            d2u = (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1]) / h ** 2
            du = (U[n, i + 1] - U[n, i - 1]) / (2 * h)
            U[n + 1, i] = U[n, i] + tau * (t[n] * d2u + x[i] * du + x[i] * t[n])

    return X, T_mesh, U


def plot_solution(X, T, U):
    plt.figure(figsize=(10, 6))

    im = plt.imshow(U, cmap="viridis", aspect='auto', origin='lower',
                    extent=[X.min(), X.max(), T.min(), T.max()])

    contours = plt.contour(X, T, U, colors='white', linewidths=0.5, levels=10)
    plt.clabel(contours, inline=True, fontsize=8)

    plt.colorbar(im, label='u(x,t)')
    plt.xlabel("x"), plt.ylabel("t")
    plt.title(f"Численное решение ({X.shape[1]} узлов)")
    plt.show()


for h, points in [(0.1, 11), (0.05, 21), (0.01, 101)]:
    X, T, U = solve(h=h, points=points)
    plot_solution(X, T, U)
