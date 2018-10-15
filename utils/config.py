import shutil
import hashlib
import json
import os

from  generic.utils.logger import create_logger

def load_config(config_file, exp_dir, args=None):
    with open(config_file, 'rb') as f_config:
        config_str = f_config.read()
        exp_identifier = hashlib.md5(config_str).hexdigest()
        config = json.loads(config_str.decode('utf-8'))

    save_path = '{}/{{}}'.format(os.path.join(exp_dir, exp_identifier))
    if not os.path.isdir(save_path.format('')):
        os.makedirs(save_path.format(''))

    # create logger
    logger = create_logger(save_path.format('train.log'))
    logger.info("Config Hash {}".format(exp_identifier))
    logger.info("Config name : {}".format(config["name"]))
    logger.info(config)

    if args is not None:
        for key, val in vars(args).items():
            logger.info("{} : {}".format(key, val))

    # set seed
    set_seed(config)

    # copy config file
    shutil.copy(config_file, save_path.format('config.json'))

    return config, exp_identifier, save_path

def get_config_from_xp(exp_dir, identifier):
    config_path = os.path.join(exp_dir, identifier, 'config.json')
    if not os.path.exists(config_path):
        raise RuntimeError("Couldn't find config")

    with open(config_path, 'r') as f:
        return json.load(f)


def get_recursively(search_dict, field, no_field_recursive=False):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:

            if no_field_recursive \
                    and (isinstance(value, dict) or isinstance(key, list)):
                continue

            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field, no_field_recursive)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field, no_field_recursive)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found



def set_seed(config):
    import numpy as np
    import tensorflow as tf
    seed = config["seed"]
    if seed > -1:
        np.random.seed(seed)
        tf.set_random_seed(seed)