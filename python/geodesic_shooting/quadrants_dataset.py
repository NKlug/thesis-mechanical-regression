import tensorflow as tf


def sample_range(bottom_left_corner, top_right_corner, n):
    """
    Samples the given box uniformly
    :param n: number of sample points
    :param bottom_left_corner: bottom left corner of the box
    :param top_right_corner: top right corner of the box
    :return: sampled points
    """
    samples = tf.random.uniform((n, n), minval=bottom_left_corner, maxval=top_right_corner, dtype=tf.float64)
    return samples


def generate_quadrants_dataset(seed=0, n=100):
    """
    Create a dataset made of two classes in four quadrants.
    :param n: number of points per spiral
    :param phi: angular velocity
    :param jitter: standard deviation of gaussian noise
    :param coils: parameter to adjust number of coils
    :param seed: seed for random sampling
    :return: Training data and labels
    """
    tf.random.set_seed(seed)
    top_left_quadrant = sample_range([-9.5, 0.5], [-0.5, 9.5], n // 2)
    bottom_right_quadrant = sample_range([0.5, -9.5], [9.5, -0.5], n // 2)
    top_right_quadrant = sample_range([0.5, 0.5], [9.5, 9.5], n // 2)
    bottom_left_quadrant = sample_range([-9.5, -9.5], [-0.5, -0.5], n // 2)
    X = tf.concat([top_left_quadrant, bottom_right_quadrant, top_right_quadrant, bottom_left_quadrant])
    X = tf.reshape(X, (-1))
    Y = tf.concat([tf.ones(n), -tf.ones(n)], axis=0)
    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    return X, Y
