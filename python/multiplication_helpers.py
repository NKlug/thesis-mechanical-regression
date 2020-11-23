import numpy as np


def scalar_vector_matrices_mult(a, b):
    """

    :param a: 1d vector
    :param b: vector containing matrices of same length as a
    :return: element-wise scalar multiplication of vector elements with matrices
    """
    return b * a[:, None, None]


def vector_matrices_mult(a, b):
    """
    Calculates a[i] @ b[i] for all i.
    :param a: 2d vector
    :param b: vector containing matrices of same length as a
    :return:
    """
    return np.einsum('ij,ilj->il', a, b)


def vector_vector_mult(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    return np.einsum('ij,ij->i', a, b)
