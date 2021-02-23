import numpy as np
import tensorflow as tf


def sample_range(bottom_left_corner, top_right_corner, n):
    """
    Samples the given box uniformly
    :param n: number of sample points
    :param bottom_left_corner: bottom left corner of the box
    :param top_right_corner: top right corner of the box
    :return: sampled points
    """
    samples_x = tf.random.uniform((n, ), minval=bottom_left_corner[0], maxval=top_right_corner[0], dtype=tf.float64)
    samples_y = tf.random.uniform((n, ), minval=bottom_left_corner[1], maxval=top_right_corner[1], dtype=tf.float64)
    return tf.stack([samples_x, samples_y], axis=-1)


def generate_quadrants_dataset(seed=578, n=100):
    """
    Create a dataset made of two classes in four quadrants.
    :param n: number of points per spiral
    :param phi: angular velocity
    :param jitter: standard deviation of gaussian noise
    :param coils: parameter to adjust number of coils
    :param seed: seed for random sampling. The default of 578 was determined to maximize distances between nodes
    :return: Training data and labels
    """
    tf.random.set_seed(seed)
    top_left_quadrant = sample_range(np.asarray([-9.5, 0.5]), np.asarray([-0.5, 9.5]), n // 2)
    bottom_right_quadrant = sample_range(np.asarray([0.5, -9.5]), np.asarray([9.5, -0.5]), n // 2)
    top_right_quadrant = sample_range(np.asarray([0.5, 0.5]), np.asarray([9.5, 9.5]), n // 2)
    bottom_left_quadrant = sample_range(np.asarray([-9.5, -9.5]), np.asarray([-0.5, -0.5]), n // 2)
    X = tf.concat([top_left_quadrant, bottom_right_quadrant, top_right_quadrant, bottom_left_quadrant], axis=0)
    X = tf.reshape(X, (-1))
    Y = tf.concat([tf.ones(n), -tf.ones(n)], axis=0)
    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    return X, Y
