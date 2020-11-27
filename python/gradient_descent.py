import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from os import path

from geodesic_shooting import V


def find_optimal_p0(X, Y, steps, checkpoint_its=1000, experiment=None):
    p0 = tf.Variable(tf.random.normal(shape=X.shape), trainable=True, dtype=tf.float32)
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
                checkpoint.save(checkpoint_dir)
    return p0
