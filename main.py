import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, h=0.1, start=0, points=51, T=1):
        self.h = h
        self.tau = self.h ** 2 / 2
        print(f"tau={self.tau}")
        self.vmin = 0
        self.vmax = None
        self.start = start
        self.end = 1
        self.T = T
        self.time_steps = int(self.T / self.tau)
        print(f"time_steps= {self.time_steps}")
        self.x = np.linspace(self.start, self.end, points)
        self.t = np.linspace(0, self.T, self.time_steps + 1)
        print(f"x = {len(self.x)}")
        print(f"t = {len(self.t)}")

        self.X, self.T = np.meshgrid(self.x, self.t)
        self.Z = np.zeros_like(self.X)
        self.Z[0, :] = self.X[0, :] ** 2
        self.Z[:, 0] = self.T[:, 0]

    def __iterate(self):
        for n in range(self.time_steps):
            for i in range(1, len(self.x) - 1):
                # du/dt = t*d²u/dx² + x*du/dx + x*t
                d2u_dx2 = (self.Z[n, i + 1] - 2 * self.Z[n, i] + self.Z[n, i - 1]) / self.h ** 2
                du_dx = (self.Z[n, i + 1] - self.Z[n, i - 1]) / (2 * self.h)
                self.Z[n + 1, i] = self.Z[n, i] + self.tau * (
                        self.t[n] * d2u_dx2 +
                        self.x[i] * du_dx +
                        self.x[i] * self.t[n]
                )

                # Граничные условия
                self.Z[n + 1, 0] = self.t[n + 1]  # левая граница
                self.Z[n + 1, -1] = self.t[n + 1] + 1  # правая граница

    def __iterate_2(self):
        for n in range(self.time_steps):
            # Коэффициент для разностной схемы
            sigma = self.t[n] * self.tau / self.h ** 2

            # Прямой ход метода прогонки
            alpha = np.zeros(len(self.x))
            beta = np.zeros(len(self.x))

            # Левое граничное условие (уже установлено)
            alpha[0] = 0
            beta[0] = self.t[n + 1]  # U(0,t) = t

            # Вычисление прогоночных коэффициентов
            for i in range(1, len(self.x) - 1):
                # Правая часть уравнения: x*t + U_prev/dt
                phi = self.x[i] * self.t[n] + self.Z[n, i] / self.tau

                a_i = sigma
                b_i = -(1 + 2 * sigma + self.x[i] * self.h / (2 * self.t[n]) if self.t[n] != 0 else 0)
                c_i = sigma
                d_i = -phi - (self.x[i] / (2 * self.h)) * (self.Z[n, i + 1] - self.Z[n, i - 1]) if self.t[n] != 0 else 0

                denominator = b_i - a_i * alpha[i - 1]
                alpha[i] = c_i / denominator
                beta[i] = (a_i * beta[i - 1] - d_i) / denominator

                # Обратный ход метода прогонки
                # Правое граничное условие уже установлено (U(1,t) = 1 + t)
                for i in range(len(self.x) - 2, 0, -1):
                    self.Z[n + 1, i] = alpha[i] * self.Z[n + 1, i + 1] + beta[i]

    def draw(self):
        self.__iterate()

        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(self.Z,
                       vmax=np.max(self.Z),
                       vmin=0,
                       cmap="viridis",
                       aspect='auto',
                       origin='lower',
                       extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()])

        contours = ax.contour(self.X, self.T, self.Z,
                              colors='white', linewidths=0.5, levels=10)

        cbar = plt.colorbar(im)

        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

        plt.xlabel("X")
        plt.ylabel("T")

        plt.show()

    def draw_2(self):
        self.__iterate_2()

        # Создаем фигуру и оси
        fig, ax = plt.subplots(figsize=(10, 6))

        # Отображаем тепловую карту
        im = ax.imshow(self.Z,
                       vmax=np.max(self.Z),
                       vmin=0,
                       cmap="magma",
                       aspect='auto',
                       origin='lower',
                       extent=[self.x.min(), self.x.max(), self.t.min(), self.t.max()])

        contours = ax.contour(self.X, self.T, self.Z,
                              colors='white', linewidths=0.5, levels=10)

        cbar = plt.colorbar(im)

        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

        plt.xlabel("X")
        plt.ylabel("T")

        plt.show()



# Задание 1

model = Model(h=0.1, points=11)
model.draw()


model = Model(h=0.05, points=51)
model.draw()


model = Model(h=0.01, points=101)
model.draw()