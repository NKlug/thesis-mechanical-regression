import numpy as np


def create_spiral(phi, jitter, coils, rotation, n=100):
    """

    :param n: number of sample points
    :param coils: parameter to adjust number of coils
    :param phi: angular velocity
    :param jitter: standard deviation of gaussian noise
    :param rotation: rotation of the spiral
    :return: the generated spiral
    """
    # adjust start sample for arcsin (which is defined on [-1, 1])
    roll = np.arange(coils**2, n + coils**2 + 1, dtype=np.float32)
    roll = np.sqrt(roll)
    angles = np.cumsum(np.arcsin(coils / roll)) + rotation
    roll = np.stack([-np.cos(phi * angles) * roll + 0.5, np.sin(phi * angles) * roll + 0.5], axis=-1)
    roll += np.random.normal(loc=0, scale=jitter, size=roll.shape)

    return roll


def generate_swiss_roll_dataset(phi, jitter=0, coils=3, seed=0):
    np.random.seed(seed)
    roll_1 = create_spiral(phi, jitter, coils, 0)
    roll_2 = create_spiral(phi, jitter, coils, np.pi)

    return roll_1, roll_2
