import numpy as np

from derivatives import symmetric_difference_quotient
from kernels import K_block, gamma
from leapfrog import explicit_leapfrog
from losses import optimal_recovery_loss
from multiplication_helpers import vector_vector_mult, vector_matrices_mult
import hamiltonians as ham


def V(p0, X, Y, mu=0.01):
    """
    Function \mathfrak{V} in Equation (3.17) in [Owhadi2020]
    :param p0: Initial momentum
    :param q1: Final locations
    :param X: Training data
    :param Y: Training labels
    :param mu: balance factor
    :return: Value of V in (3.17)
    """
    q, _ = explicit_leapfrog(ham.dq_h, ham.dp_h, X, p0, step=0.2)
    return mu / 2 * p0.T @ gamma(X, X) @ p0 + optimal_recovery_loss(q[-1], Y, K_block)


def grad_V(p0, X, Y):
    """
    Gradient of V at point p0
    :param p0: point at which to calculate the gradient
    :param X: Training data
    :param Y: Training labels
    :return: The gradient of V at point p0
    """

    def partial_V(direction):
        """

        :param direction: index of direction in which to calculate the partial derivative
        :return:
        """
        dir_vector = np.zeros(p0.shape[0])
        dir_vector[direction] = 1
        f = lambda x: V(p0 + dir_vector * x, X, Y)
        return symmetric_difference_quotient(f, 0)

    return np.vectorize(partial_V)(np.arange(p0.shape[0]))
