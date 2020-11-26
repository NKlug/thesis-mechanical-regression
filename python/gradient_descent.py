import numpy as np
from tqdm import tqdm


def gradient_descent(grad_f, start, learning_rate, steps):
    solutions = np.zeros(steps)
    solutions[0] = start
    for i in range(1, steps):
        solutions[i] = learning_rate * grad_f(solutions[i - 1])
    return solutions


def adam(grad_f, start, X, Y, steps, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, callback=None):
    solutions = np.zeros((steps, *start.shape))
    solutions[0] = start
    m = 0
    v = 0
    for t in tqdm(range(1, steps)):
        g = grad_f(solutions[t - 1], X, Y)
        m = beta_1 * m + (1 - beta_1) * g
        v = beta_2 * v + (1 - beta_2) * (g ** 2)
        m = m / (1 - np.power(beta_1, t))
        v = v / (1 - np.power(beta_2, t))
        solutions[t] = solutions[t - 1] - alpha * m / (np.sqrt(v) + epsilon)
        if callback:
            callback(solutions[t], X, Y)
    return solutions
