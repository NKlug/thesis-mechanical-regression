import numpy as np

from derivatives import symmetric_difference_quotient
from kernels import gamma
from multiplication_helpers import vector_matrices_mult


def dq_h(q, p):
    def partial_f(direction):
        """

        :param direction: index of direction in which to calculate the partial derivative
        :return:
        """
        dir_vector = np.zeros(q.shape[0])
        dir_vector[direction] = 1
        f = lambda d: 0.5 * p.T @ gamma(q + dir_vector * d, q + dir_vector * d) @ p
        return symmetric_difference_quotient(f, 0)

    return np.vectorize(partial_f)(np.arange(q.shape[0]))


def dp_h(q, p):
    return p @ gamma(q, q)
