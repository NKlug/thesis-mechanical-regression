import tensorflow as tf


def K(u, v, s, r):
    """
    K: X x X -> L(Y, Y), which means R^2 x R^2 -> R.
    In this implementation this is R^(nx2) x R^(nx2) -> R^(nxn), as in (2.15) a
    "block operator matrix with entries K(u_i, v_j)" is required.
    This function returns exactly this (nxn) matrix.
    :param u:
    :param v:
    :return:
    """
    # Reshape to (n, 2) from (2n)
    u = tf.reshape(u, (u.shape[0] // 2, 2))
    v = tf.reshape(v, (v.shape[0] // 2, 2))
    # Calculate pairwise differences first
    pw_difference = u[:, None, :] - v[None, :, :]
    # Use scalar product instead of squared norm due to numerical instability when differentiating the square root
    x = tf.einsum('ijk,ijk->ij', pw_difference, pw_difference)  # Elementwise scalar product
    return tf.exp(- x / (s**2)) + r


def Gamma(u, v, s, r):
    """
    X x X -> L(X, X), which means R^2 x R^2 -> R^(2x2). In this implementation, this is
    R^(2n) x R^(2n) -> R^(2n x 2n) (block operator matrix).
    The blocks are: K(u_i, v_j) * I, where I is the 2x2 unit matrix.
    :param u:
    :param v:
    :return:
    """
    gaussian = K(u, v, s, r)
    identity_op = tf.linalg.LinearOperatorIdentity(num_rows=2, dtype=tf.float64)
    gaussian_op = tf.linalg.LinearOperatorFullMatrix(gaussian)
    return tf.linalg.LinearOperatorKronecker([gaussian_op, identity_op]).to_dense()
