import tensorflow as tf

import gpu_ready.hamiltonians as ham
from gpu_ready.kernels import gamma, K_block
from gpu_ready.leapfrog import explicit_leapfrog
from gpu_ready.losses import optimal_recovery_loss


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
    v = tf.tensordot(p0, tf.linalg.matvec(gamma(X, X), p0), axes=1)
    return mu / 2 * v + optimal_recovery_loss(q, Y, K_block)

# def grad_V(p0, X, Y):
#     """
#     Gradient of V at point p0
#     :param p0: point at which to calculate the gradient
#     :param X: Training data
#     :param Y: Training labels
#     :return: The gradient of V at point p0
#     """
#     p_var = tf.Variable(p0, dtype=tf.float32)
#
#
#
#     return np.vectorize(partial_V)(np.arange(p0.shape[0]))
