import tensorflow as tf
from tqdm import tqdm

import geodesic_shooting.hamiltonians as ham
from geodesic_shooting import kernels
from geodesic_shooting.shooting_function_V import V


class Model(object):
    def __init__(self, checkpoint_interval, model_params):
        """
        A model of the mechanical regression problem proposed in [Owhadi2020].
        :param checkpoint_interval: number of iterations after which to save the parameters
        :param model_params: TrainingParameters object containing additional parameters
        """
        self.checkpoint_interval = checkpoint_interval
        self.X, self.Y = model_params.dataset
        self.log_dir = model_params.log_dir
        self.checkpoint_dir = model_params.checkpoint_dir

        # Initialize variables
        self.global_step = tf.Variable(0, name='global_step')
        initial_p0 = tf.random.normal(shape=self.X.shape, dtype=tf.float64)
        self.p0 = tf.Variable(initial_p0, trainable=True, dtype=tf.float64, name='p0')
        self.trainable_variables = [self.p0]

        # Define functions with respective hyper-parameters
        self.K = lambda u, v: kernels.K(u, v, s=model_params.s, r=model_params.r)
        self.Gamma = lambda u, v: kernels.Gamma(u, v, s=model_params.s, r=model_params.r)
        self.dq_h = lambda q, p: ham.dq_h(q, p, self.Gamma)
        self.dp_h = lambda q, p: ham.dp_h(q, p, self.Gamma)
        self.loss = lambda: V(self.p0, self.X, self.Y, mu=model_params.mu,
                              leapfrog_step=model_params.h,
                              Gamma=self.Gamma,
                              K=self.K,
                              dq_h=self.dq_h,
                              dp_h=self.dp_h,
                              is_training=True,
                              global_step=self.global_step)
        self.optimizer = tf.optimizers.Adam()

        # Take care of logging and saving
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
