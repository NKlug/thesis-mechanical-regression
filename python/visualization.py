import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def create_result_slider(q, interval, n):
    global is_manual
    fig, ax = plt.subplots()
    l1, = plt.plot([], [], '.')
    l2, = plt.plot([], [], '.')
    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    # Slider
    samp = Slider(axamp, 'Step', valmin=0, valmax=q.shape[0] - 1, valstep=1, valinit=0)

    # Animation controls
    is_manual = False  # True if user has taken control of the animation

    def init_animation():
        a = 1
        x_max, x_min = np.max(q[:, :, 0]), np.min(q[:, :, 0])
        y_max, y_min = np.max(q[:, :, 1]), np.min(q[:, :, 1])
        ax.set_xlim(x_min - a, x_max + a)
        ax.set_ylim(y_min - a, y_max + a)
        return l1, l2

    def update_slider(val):
        global is_manual
        is_manual = True
        update(val)

    def update(val):
        # update curve
        l1.set_data(q[val, :n, 0], q[val, :n, 1])
        l2.set_data(q[val, n:, 0], q[val, n:, 1])
        # redraw canvas while idle
        fig.canvas.draw_idle()

    def update_plot(num):
        global is_manual
        if is_manual:
            return l1, l2  # don't change

        samp.set_val(num % q.shape[0])
        is_manual = False  # the above line called update_slider, so we need to reset this
        return l1, l2

    def on_click(event):
        # Check where the click happened
        (xm, ym), (xM, yM) = samp.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            global is_manual
            is_manual = False

    # call update function on slider value change
    samp.on_changed(update_slider)

    fig.canvas.mpl_connect('button_press_event', on_click)

    return ani.FuncAnimation(fig, update_plot, interval=interval, init_func=init_animation, frames=q.shape[0])


def create_result_animation(q, interval, n):
    q = np.concatenate([q, q[::-1]], axis=0)
    fig, ax = plt.subplots()
    l1, = plt.plot([], [], '.')
    l2, = plt.plot([], [], '.')

    def update_animation(num, data, l1, l2):
        l1.set_data(data[num, :n, 0], data[num, :n, 1])
        l2.set_data(data[num, n:, 0], data[num, n:, 1])
        return l1, l2

    def init_animation():
        a = 1
        x_max, x_min = np.max(q[:, :, 0]), np.min(q[:, :, 0])
        y_max, y_min = np.max(q[:, :, 1]), np.min(q[:, :, 1])
        ax.set_xlim(x_min - a, x_max + a)
        ax.set_ylim(y_min - a, y_max + a)
        return l1, l2

    return ani.FuncAnimation(fig, update_animation, frames=q.shape[0], fargs=(q, l1, l2), init_func=init_animation,
                             interval=interval, blit=True)
