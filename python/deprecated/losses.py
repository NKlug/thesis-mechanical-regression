import numpy as np


def optimal_recovery_loss(x, y, k):
    """
    Optimal recovery loss as in (2.15) in [Owhadi2020]
    :param k: kernel
    :param x: training data
    :param y: training labels
    :return:
    """
    return y.T @ np.linalg.inv(k(x, x)) @ y
