from gpu_ready.gradient_descent import find_optimal_p0
from gpu_ready.kernels import gamma
from gpu_ready.leapfrog import explicit_leapfrog
from gpu_ready.swiss_roll_dataset import generate_swiss_roll_dataset
import gpu_ready.hamiltonians as ham
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X = tf.concat(generate_swiss_roll_dataset(1, jitter=0.1, coils=0.65), axis=0)
    X = tf.reshape(X, (-1))
    Y = tf.concat([tf.ones(100), -tf.ones(100)], axis=0)
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    a = gamma(X, X)

    # q, p = explicit_leapfrog(ham.dq_h, ham.dp_h, X, tf.random.normal(shape=X.shape), step=0.2, t_stop=1)

    find_optimal_p0(X, Y, steps=1000)
