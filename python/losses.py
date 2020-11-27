import tensorflow as tf


def optimal_recovery_loss(x, y, k):
    """
    Optimal recovery loss as in (2.15) in [Owhadi2020]
    :param k: kernel
    :param x: training data
    :param y: training labels
    :return: the loss
    """
    v = tf.linalg.matvec(k(x, x), y)
    return tf.linalg.tensordot(y, v, axes=1)
