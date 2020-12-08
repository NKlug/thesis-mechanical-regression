import tensorflow as tf

from geodesic_shooting.leapfrog import explicit_leapfrog


def V(p0, X, Y, mu, leapfrog_step, Gamma, K, dq_h, dp_h, recovery_loss, is_training=False, global_step=None):
    """
    Function \mathfrak{V} in Equation (3.17) in [Owhadi2020]
    :param global_step: global step for logging the loss. Only relevant if is_training is True.
    :param is_training: if True, the loss will be logged to Tensorboard
    :param p0: Initial momentum
    :param X: Training data
    :param Y: Training labels
    :param mu: balance factor
    :return: Value of V in (3.17)
    """
    q, _ = explicit_leapfrog(dq_h, dp_h, X, p0, step=leapfrog_step)
    v = tf.tensordot(p0, tf.linalg.matvec(Gamma(X, X), p0), axes=1)
    deformation_loss = mu / 2 * v
    recovery_loss = recovery_loss(q, Y, K)
    loss = deformation_loss + recovery_loss
    if is_training:
        if global_step is None:
            raise Exception('Global step must not be None when training!')
        tf.summary.scalar('loss/recovery', recovery_loss, step=global_step.numpy())
        tf.summary.scalar('loss/deformation', deformation_loss, step=global_step.numpy())
        tf.summary.scalar('loss/total', loss, step=global_step.numpy())
    return loss
