import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from os import path

from geodesic_shooting import V


def find_optimal_p0(X, Y, steps, checkpoint_its=1000, experiment=None):
    """
    Approximates the optimal initial momentum by minimizing (3.17) in [Owhadi2020]
    w.r.t. p(0).
    :param X: training data
    :param Y: training labels
    :param steps: number of steps to run on the optimizer
    :param checkpoint_its: number of iterations after which to save the parameters
    :param experiment: name of the experiment
    """
    p0 = tf.Variable(tf.random.normal(shape=X.shape, dtype=tf.float64), trainable=True, dtype=tf.float64, name='p0')
    trainable_variables = [p0]
    loss = lambda: V(p0, X, Y, is_training=True)
    optimizer = tf.optimizers.Adam()

    if experiment is None:
        experiment = datetime.now().strftime('%Y_%m_%d_%H:%Mh')
    log_dir = path.join('..', 'training', 'logs', experiment)
    checkpoint_dir = path.join('..', 'training', 'checkpoints', experiment)

    summary_writer = tf.summary.create_file_writer(log_dir)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer)
    for i in tqdm(range(steps)):
        with summary_writer.as_default():
            tf.summary.experimental.set_step(i)
            optimizer.minimize(loss, var_list=trainable_variables)
            if i % checkpoint_its == 0:
                checkpoint.save(path.join(checkpoint_dir, 'checkpoint_' + str(i)))
