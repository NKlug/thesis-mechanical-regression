import numpy as np

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
    q, _ = explicit_leapfrog(ham.dq_h, ham.dp_h, X, p0, step=0.001)
    return mu / 2 * vector_vector_mult(vector_matrices_mult(p0, gamma(X, X)), p0) + \
           optimal_recovery_loss(q[-1], Y, K_block)


def derivative_V(p0, direction, X, Y, h=0.001):
    """
    Calculates the symmetric difference quotient of V in the given direction
    :param h: parameter for the symmetric difference quotient
    :param p0: point at which to calculate the derivative
    :param direction: derivation direction
    :param X: Training data
    :param Y: Training labels
    :return: The derivative of V in the given direction
    """
    direction = np.asarray(direction)
    delta_h = np.tile(h * direction, reps=(p0.shape[0], 1))
    return (V(p0 - delta_h, X, Y) - V(p0 + delta_h, X, Y)) / (2 * h)


def grad_V(p0, X, Y):
    """
    Gradient of V at point p0
    :param p0: point at which to calculate the gradient
    :param X: Training data
    :param Y: Training labels
    :return: The gradient of V at point p0
    """
    x = derivative_V(p0, [1, 0], X, Y)
    y = derivative_V(p0, [0, 1], X, Y)
    return np.stack([x, y], axis=-1)
