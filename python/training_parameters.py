from os import path

from geodesic_shooting.losses import optimal_recovery_loss, ridge_regression_loss
from geodesic_shooting.swiss_roll_dataset import generate_swiss_roll_dataset

losses = {
    'optimal_recovery': optimal_recovery_loss,
    'ridge_regression': ridge_regression_loss
}


class TrainingParameters(object):
    def __init__(self, dataset, mu, h, s, r, ls_regularizer, checkpoint_base_dir, log_base_dir, name=None,
                 user_dir_override=None, loss=None,
                 *args, **kwargs):
        """
        Hyper parameters for the model.
        :param dataset: name of dataset
        :param mu: balancing factor in the shooting function V
        :param h: step width for the leapfrog integrator
        :param s: standard deviation in gaussian kernel
        :param r: nugget for gaussian kernel
        """
        _datasets = {
            'default_swiss_roll': generate_swiss_roll_dataset(1, jitter=0.1, coils=0.65, n=100)
        }

        self.dataset = _datasets[dataset]
        self.mu = mu
        self.h = h
        self.s = s
        self.r = r
        self.ls_regularizer = ls_regularizer

        if loss is None:
            loss = 'optimal_recovery'
        self.loss = losses[loss]

        if name is None:
            raise Exception('Experiment name must not be None!')
        self.experiment = name
        if user_dir_override is not None:
            checkpoint_base_dir = checkpoint_base_dir.replace('~', user_dir_override)
            log_base_dir = log_base_dir.replace('~', user_dir_override)
        self.checkpoint_dir = path.realpath(path.expanduser(path.join(checkpoint_base_dir, self.experiment)))
        self.log_dir = path.realpath(path.expanduser(path.join(log_base_dir, self.experiment)))
