import tensorflow as tf

from geodesic_shooting.model import Model
from geodesic_shooting.swiss_roll_dataset import generate_swiss_roll_dataset

if __name__ == '__main__':
    # limit gpu memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    # Create Dataset: X is (2N x 1) and Y is (N x 1)
    X, Y = generate_swiss_roll_dataset(1, jitter=0.1, coils=0.65, n=40)

    model = Model(X=X, Y=Y, checkpoint_interval=10, log_dir='../training/logs',
                  checkpoint_dir='../training/checkpoints')
    # approximate the optimal initial momentum
    model.train(steps=1000000)
