import tensorflow as tf
from gpu_ready.kernels import gamma


def dq_h(q, p):
    q_var = tf.Variable(q, dtype=tf.float32)
    with tf.GradientTape() as t:
        v = tf.linalg.matvec(gamma(q_var, q_var), p)
        h = 0.5 * tf.tensordot(p, v, axes=1)
    return t.gradient(h, q_var)


def dp_h(q, p):
    return tf.linalg.matvec(gamma(q, q), p)
