import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np


def create_result_animation(q, interval, n, scale=10):
    q = np.concatenate([q, q[::-1]], axis=0)
    fig, ax = plt.subplots()
    l1, = plt.plot([], [], '.')
    l2, = plt.plot([], [], '.')

    def update(num, data, l1, l2):
        l1.set_data(data[num, :n, 0], data[num, :n, 1])
        l2.set_data(data[num, n:, 0], data[num, n:, 1])
        return l1, l2

    def init():
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        return l1, l2

    return ani.FuncAnimation(fig, update, frames=q.shape[0], fargs=(q, l1, l2), init_func=init, interval=interval, blit=True)
