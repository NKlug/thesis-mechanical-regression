import tensorflow as tf


def dq_h(q, p, Gamma):
    """
    Partial derivative of the Hamiltonian in (1.13) in [Owhadi2020] w.r.t q
    :param q: coordinates
    :param p: momenta
    :return: the partial derivative vector
    """
    with tf.GradientTape() as t:
        t.watch(q)
        v = tf.linalg.matvec(Gamma(q, q), p)
        h = 0.5 * tf.tensordot(p, v, axes=1)
    return t.gradient(h, q)


def dp_h(q, p, Gamma):
    """
    Partial derivative of the Hamiltonian in (1.13) in [Owhadi2020] w.r.t q
    :param q: coordinates
    :param p: momenta
    :return: the partial derivative vector
    """
    return tf.linalg.matvec(Gamma(q, q), p)
