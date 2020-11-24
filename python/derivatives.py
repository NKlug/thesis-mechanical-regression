import numpy as np


def symmetric_difference_quotient(f, x, h=None):
    """
    Calculates the symmetric difference quotient of f at point x
    :param h: parameter for symmetric difference quotient
    :param f: the function
    :param x: the point
    :return:
    """
    # TODO ensure we use float64 everywhere
    if h is None:
        h = np.cbrt(np.finfo(np.float32).eps)  # https://en.wikipedia.org/wiki/Numerical_differentiation#Step_size
    h = np.tile(h, reps=(x.shape[:-1], 1))
    return (f(x - h) + f(x + h)) / (2 * h)