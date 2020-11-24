from kernels import gamma
from multiplication_helpers import vector_matrices_mult


def dq_h(q, p):
    return 0


def dp_h(q, p):
    return vector_matrices_mult(p, gamma(q, q))