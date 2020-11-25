from derivatives import symmetric_difference_quotient
from kernels import gamma
from multiplication_helpers import vector_matrices_mult


def dq_h(q, p):
    f = lambda q1: 0.5 * p.T @ gamma(q1, q1) @ p
    return symmetric_difference_quotient(f, q)


def dp_h(q, p):
    return p @ gamma(q, q)
