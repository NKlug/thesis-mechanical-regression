import tensorflow as tf
import numpy as np


@tf.function
def create_spiral(phi, jitter, coils, rotation, n=100):
    """
    Samples equidistant points from a 2D spiral with gaussian noise
    :param n: number of sample points
    :param coils: parameter to adjust number of coils
    :param phi: angular velocity
    :param jitter: standard deviation of gaussian noise
    :param rotation: rotation of the spiral
    :return: the generated spiral
    """
    # adjust start sample for arcsin (which is defined on [-1, 1])
    roll = tf.range(coils ** 2, n + coils ** 2, dtype=tf.float32)
    roll = tf.sqrt(roll)
    angles = tf.cumsum(tf.asin(coils / roll)) + rotation
    roll = tf.stack([-tf.cos(phi * angles) * roll + 0.5, tf.sin(phi * angles) * roll + 0.5], axis=-1)
    roll += tf.random.normal(shape=roll.shape, mean=0, stddev=jitter)

    return roll


def generate_swiss_roll_dataset(phi, jitter=0, coils=3, seed=0):
    """
    Generates the swissroll dataset in Section 3.13.2 in [Owhadi2020]
    :param phi: angular velocity
    :param jitter: standard deviation of gaussian noise
    :param coils: parameter to adjust number of coils
    :param seed: seed for random sampling
    :return: the two spirals of the swissroll dataset
    """
    tf.random.set_seed(seed)
    roll_1 = create_spiral(phi, jitter, coils, 0)
    roll_2 = create_spiral(phi, jitter, coils, np.pi)

    return roll_1, roll_2
