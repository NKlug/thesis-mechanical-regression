import re
from os import path, listdir

import numpy as np

from geodesic_shooting.leapfrog import simulate_flow


class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def all_chekpoints(self):
        files = [f for f in listdir(self.model.checkpoint_dir) if path.isfile(path.join(self.model.checkpoint_dir, f))]
        all_checkpoints = [re.findall(r'ckpt-\d+', f) for f in files if 'ckpt' in f]
        all_checkpoints = [path.join(self.model.checkpoint_dir, c[0]) for c in all_checkpoints if len(c) > 0]
        return sorted(set(all_checkpoints), key=lambda s: int(re.findall(r'\d+|$', path.basename(s))[0]))

    def flows_over_epochs(self, every_nth=1):
        all_flows = []
        for checkpoint in self.all_chekpoints()[::every_nth]:
            self.model.restore(checkpoint)

            q, _ = simulate_flow(self.model.dq_h, self.model.dp_h, self.model.X, self.model.p0,
                                 step=self.model.model_params.h, t_stop=1)
            q = q.reshape((-1, self.model.X.shape[0]//2, 2))
            all_flows.append(q)

        return np.asarray(all_flows)
