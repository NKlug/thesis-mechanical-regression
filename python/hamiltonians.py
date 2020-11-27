import tensorflow as tf
from kernels import gamma


def dq_h(q, p):
    with tf.GradientTape() as t:
        t.watch(q)
        v = tf.linalg.matvec(gamma(q, q), p)
        h = 0.5 * tf.tensordot(p, v, axes=1)
    return t.gradient(h, q)


def dp_h(q, p):
    return tf.linalg.matvec(gamma(q, q), p)
