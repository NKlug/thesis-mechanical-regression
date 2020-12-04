from datetime import datetime
from os import path

from geodesic_shooting.swiss_roll_dataset import generate_swiss_roll_dataset


class TrainingParameters(object):
    def __init__(self, dataset, mu, h, s, r, ls_regularizer, checkpoint_base_dir, log_base_dir, experiment=None):
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
        if experiment is None:
            self.experiment = datetime.now().strftime('%Y_%m_%d_%H:%Mh')
        else:
            self.experiment = datetime.now().strftime('%Y_%m_%d_') + str(experiment)
        self.checkpoint_dir = path.realpath(path.expanduser(path.join(checkpoint_base_dir, self.experiment)))
        self.log_dir = path.realpath(path.expanduser(path.join(log_base_dir, self.experiment)))
