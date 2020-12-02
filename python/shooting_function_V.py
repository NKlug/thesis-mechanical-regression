import tensorflow as tf

import hamiltonians as ham
from kernels import Gamma, K
from leapfrog import explicit_leapfrog
from losses import optimal_recovery_loss

import pickle as pkl


def V(p0, X, Y, mu=0.01, is_training=False, global_step=None):
    """
    Function \mathfrak{V} in Equation (3.17) in [Owhadi2020]
    :param global_step: global step for logging the loss. Only relevant if is_training is True.
    :param is_training: if True, the loss will be logged to Tensorboard
    :param p0: Initial momentum
    :param q1: Final locations
    :param X: Training data
    :param Y: Training labels
    :param mu: balance factor
    :return: Value of V in (3.17)
    """
    q, _ = explicit_leapfrog(ham.dq_h, ham.dp_h, X, p0, step=0.2)
    v = tf.tensordot(p0, tf.linalg.matvec(Gamma(X, X), p0), axes=1)
    loss = mu / 2 * v + optimal_recovery_loss(q, Y, K)
    if is_training:
        if global_step is None:
            raise Exception('Global step must not be None when training!')
        tf.summary.scalar('loss', loss, step=global_step)
    return loss
