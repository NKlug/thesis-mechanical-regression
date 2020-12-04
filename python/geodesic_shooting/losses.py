import tensorflow as tf


def optimal_recovery_loss(x, y, k, regularizer):
    """
    Optimal recovery loss as in (2.15) in [Owhadi2020]
    :param k: kernel
    :param x: training data
    :param y: training labels
    :return: the loss
    """
    # solve d = k(x, x) @ y rather than computing the inverse because of numerical stability
    kernel_mat = k(x, x)
    d = tf.linalg.solve(kernel_mat + regularizer * tf.eye(kernel_mat.shape[0], dtype=tf.float64), y[:, None])[:, 0]
    return tf.tensordot(y, d, axes=1)
