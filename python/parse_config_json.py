import json

from training_parameters import TrainingParameters


def _get_or_raise(data, key):
    result = data.get(key)
    if result is None:
        raise Exception("Key '{}' missing in data!".format(key))
    return result


def _get_or_default(data, key, default):
    return data.get(key, default)


def from_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def get_params_from_config(file_path, user_dir_override=None):
    data = from_json(file_path)

    return TrainingParameters(**data, user_dir_override=user_dir_override)
