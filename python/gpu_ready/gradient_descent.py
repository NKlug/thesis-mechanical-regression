import tensorflow as tf
from tqdm import tqdm

from gpu_ready.geodesic_shooting import V


def gradient_descent(grad_f, start, learning_rate, steps):
    solutions = tf.zeros(steps)
    solutions[0] = start
    for i in range(1, steps):
        solutions[i] = learning_rate * grad_f(solutions[i - 1])
    return solutions


def find_optimal_p0(X, Y, steps):
    optimizer = tf.optimizers.Adam(beta_1=0.5)
    p0 = tf.Variable(tf.random.normal(shape=X.shape))
    for _ in tqdm(range(steps)):
        loss = lambda: V(p0, X, Y)
        optimizer.minimize(loss, var_list=[p0])

    pass
