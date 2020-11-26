import numpy as np
from libc.math cimport sqrt, exp

cdef double euclidean_distance(double x1, double x2, double y1, double y2):
    cdef double d= (x1 - y1)**2 + (x2 - y2)**2
    return sqrt(d)


def K_block(double[:] u, double[:] v):
    """
    K: X x X -> L(Y, Y), which means R^2 x R^2 -> R, which means R^(nx2) x R^(nx2) -> R^n for n values.
    However, in (2.15) a "block operator matrix with entries K(u_i, v_j)" is required.
    Hence this function returns exactly this (nxn) matrix
    :param u:
    :param v:
    :return:
    """
    cdef Py_ssize_t n = u.shape[0]//2
    cdef double[:] u_view = u
    cdef double[:] v_view = v

    np_result = np.empty(shape=(n, n))
    cdef double[:, :] result = np_result
    cdef double d

    cdef unsigned int i, j
    for i in range(n):
        for j in range(i+1):
            d = euclidean_distance(u[2*i], u[2*i+1], v[2*j], v[2*j+1])
            d = exp(- (d ** 2) / 25) + 0.1
            result[i, j] = d
            result[j, i] = d
    return np_result


def gamma(u, v):
    """
    X x X -> L(X, X), which means R^2 xv R^2 -> R^(2x2), which means R^(nx2) x R^(nx2) -> R^(2n x 2n) (block matrix)
    Alternatively, element wise vector matrix multiplication. <-
    :param u:
    :param v:
    :return:
    """
    cdef Py_ssize_t n = u.shape[0] // 2

    cdef double[:, :] gaussian = K_block(u, v)
    np_result = np.zeros((2*n, 2*n))
    cdef double[:, :] result = np_result
    cdef unsigned int i, j
    for i in range(n):
        for j in range(n):
            result[2*i, 2*j] = gaussian[i, j]
            result[2*i + 1, 2*j +1] = gaussian[i, j]

    return np_result
