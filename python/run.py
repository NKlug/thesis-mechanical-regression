import argparse
from os import path, listdir
from pprint import pprint

import tensorflow as tf

from config.parse_config_json import get_params_from_config, from_json
from geodesic_shooting.model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', help=' location of the config for the current training.')
    args = parser.parse_args()

    # limit gpu memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Creating model from the following config:")
    config_path = path.realpath(path.expanduser(args.config))
    pprint(from_json(config_path))
    params = get_params_from_config(config_path)

    # check if there already exists an experiment with the same name
    if path.exists(params.checkpoint_dir) and len(listdir(params.checkpoint_dir)) != 0:
        print("There is already an experiment '{}' in '{}'! Aborting...".format(params['experiment'],
                                                                                params.checkpoint_dir))
        exit(1)

    model = Model(checkpoint_interval=10, model_params=params)
    # approximate the optimal initial momentum
    model.train(steps=150000)
