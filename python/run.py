from gradient_descent import find_optimal_p0
from swiss_roll_dataset import generate_swiss_roll_dataset
import tensorflow as tf

if __name__ == '__main__':
    # limit gpu memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    # Create Dataset: X is (2N x 1) and Y is (N x 1)
    X, Y = generate_swiss_roll_dataset(1, jitter=0.1, coils=0.65, n=40)

    # approximate the optimal initial momentum
    find_optimal_p0(X, Y, steps=100000, checkpoint_its=100)
