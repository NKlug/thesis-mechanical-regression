import re
from os import path, listdir

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from geodesic_shooting.leapfrog import simulate_flow


class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def all_checkpoints(self):
        """
        Creates a list of the paths of all checkpoints available for the model
        :return: A list of the paths of all checkpoints
        """
        files = [f for f in listdir(self.model.checkpoint_dir) if path.isfile(path.join(self.model.checkpoint_dir, f))]
        all_checkpoints = [re.findall(r'ckpt-\d+', f) for f in files if 'ckpt' in f]
        all_checkpoints = [path.join(self.model.checkpoint_dir, c[0]) for c in all_checkpoints if len(c) > 0]
        return sorted(set(all_checkpoints), key=lambda s: int(re.findall(r'\d+|$', path.basename(s))[0]))

    def flows_over_epochs(self, every_nth=1):
        """
        Calculates the flows q for each checkpoint available for the model
        :param every_nth: only evaluate every n-th checkpoint
        :return: dict of the flows and initial momenta for every n-th checkpoint
        """
        all_flows = {}
        for checkpoint in self.all_checkpoints()[::every_nth]:
            self.model.restore(checkpoint)

            q, _ = simulate_flow(self.model.dq_h, self.model.dp_h, self.model.X, self.model.p0,
                                 step=self.model.model_params.h, t_stop=1)
            q = q.reshape((-1, self.model.X.shape[0] // 2, 2))
            all_flows[checkpoint] = {'flow': q, 'p0': self.model.p0}

        return all_flows

    def find_best_checkpoint(self, all_flows):
        """
        Finds the checkpoint with the smallest overall loss
        :param all_flows: flows for the checkpoints that should be checked
        :return: path of the best checkpoint, value of smallest loss
        """
        best_checkpoint = None
        best_loss = np.infty
        for checkpoint, data in tqdm(all_flows.items()):
            flow = data['flow']
            p0 = data['p0']
            v = tf.tensordot(p0, tf.linalg.matvec(self.model.Gamma(self.model.X, self.model.X), p0),
                             axes=1)
            deformation_loss = self.model.model_params.mu / 2 * v
            loss = self.model.regression_loss(flow[-1].reshape(-1), self.model.Y, self.model.K) + deformation_loss
            loss = loss.numpy()
            if loss < best_loss:
                best_loss = loss
                best_checkpoint = checkpoint
        return best_checkpoint, best_loss
