import tensorflow as tf

import hamiltonians as ham
from python import gamma, K_block
from leapfrog import explicit_leapfrog
from python import optimal_recovery_loss


def V(p0, X, Y, mu=0.01, is_training=False):
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
    loss = mu / 2 * v + optimal_recovery_loss(q, Y, K_block)
    if is_training:
        tf.summary.scalar('loss', loss, step=tf.summary.experimental.get_step())
    return loss

