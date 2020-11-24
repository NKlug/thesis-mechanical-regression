import numpy as np
from multiplication_helpers import scalar_vector_matrices_mult


def K_block(u, v):
    """
    K: X x X -> L(Y, Y), which means R^2 x R^2 -> R, which means R^(nx2) x R^(nx2) -> R^n for n values.
    However, in (2.15) a "block operator matrix with entries K(u_i, v_j)" is required.
    Hence this function returns exactly this (nxn) matrix
    :param u:
    :param v:
    :return:
    """
    assert u.shape == v.shape
    # Calculate pairwise distances first
    x = np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1)
    return np.exp(- (x ** 2) / 25) + 0.1


def gamma(u, v):
    """
    X x X -> L(X, X), which means R^2 xv R^2 -> R^(2x2), which means R^(nx2) x R^(nx2) -> R^(2n x 2n) (block matrix)
    Alternatively, element wise vector matrix multiplication. <-
    :param u:
    :param v:
    :return:
    """
    assert u.shape == v.shape
    gaussian = np.exp(-np.linalg.norm(u - v, axis=-1) ** 2 / 25) + 0.1
    identity_matrices = np.tile(np.identity(u.shape[-1]), reps=(u.shape[0], 1, 1))
    return scalar_vector_matrices_mult(gaussian, identity_matrices)
