import pickle as pkl
from glob import glob
from os import path

from config.parse_config_json import get_params_from_config
from evaluation.evaluator import Evaluator
from geodesic_shooting.model import Model


def evaluate_multiple_experiments(name_pattern, config_base_dir, user_dir_override):
    for config_path in glob(path.join(config_base_dir, name_pattern)):
        print('Evaluating experiment ', path.basename(config_path).replace('.json', ''))
        params = get_params_from_config(config_path, user_dir_override)
        model = Model(checkpoint_interval=10, model_params=params)
        evaluator = Evaluator(model)
        all_flows = evaluator.flows_over_epochs(every_nth=10)

        result_path = path.join(user_dir_override, 'training', 'results',
                                path.basename(config_path).replace('.json', '.pkl'))
        with open(result_path, 'wb') as f:
            pkl.dump(all_flows, f)


if __name__ == '__main__':
    name_pattern = '21_01_14/2021_01_14_s_5_r_0.1_h_0.2_mu_*'
    config_base_dir = '/home/nikolas/Projects/thesis-mechanical-regression/training/configs'
    user_dir_override = '/'

    evaluate_multiple_experiments(name_pattern, config_base_dir, user_dir_override)
