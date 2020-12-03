from datetime import datetime
from os import path

import tensorflow as tf
from tqdm import tqdm

from geodesic_shooting.shooting_function_V import V


class Model(object):
    def __init__(self, X, Y, checkpoint_interval, log_dir, checkpoint_dir, hyper_params, experiment=None):
        """
        A model of the mechanical regression problem proposed in [Owhadi2020].
        :param X: training data
        :param Y: training labels
        :param checkpoint_interval: number of iterations after which to save the parameters
        :param log_dir: logging directory
        :param checkpoint_dir: checkpoint directory
        :param experiment: optional experiment name. If None defaults to current date and time.
        """
        self.X = X
        self.Y = Y
        self.checkpoint_interval = checkpoint_interval
        if experiment is None:
            self.experiment = datetime.now().strftime('%Y_%m_%d_%H:%Mh')
        else:
            self.experiment = datetime.now().strftime('%Y_%m_%d_') + str(experiment)
        self.log_dir = path.join(log_dir, self.experiment)
        self.checkpoint_dir = path.join(checkpoint_dir, self.experiment)

        self.global_step = tf.Variable(0, name='global_step')
        initial_p0 = tf.random.normal(shape=X.shape, dtype=tf.float64)
        self.p0 = tf.Variable(initial_p0, trainable=True, dtype=tf.float64, name='p0')
        self.trainable_variables = [self.p0]

        self.loss = lambda: V(self.p0, self.X, self.Y, is_training=True, global_step=self.global_step)
        self.optimizer = tf.optimizers.Adam()

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              initial_momentum=self.p0,
                                              global_step=self.global_step)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir,
                                                             checkpoint_interval=self.checkpoint_interval,
                                                             step_counter=self.global_step,
                                                             max_to_keep=None)

    def train(self, steps):
        """
        Approximates the optimal initial momentum by minimizing (3.17) in [Owhadi2020]
        w.r.t. p(0).
        :param steps: number of steps to run on the optimizer
        """
        for i in tqdm(range(steps)):
            with self.summary_writer.as_default():
                self.global_step = i
                self.optimizer.minimize(self.loss, var_list=self.trainable_variables)
                self.checkpoint_manager.save()
