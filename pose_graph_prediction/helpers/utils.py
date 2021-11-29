from json import load as load_json_file

from os import environ
from os.path import exists

import random

import torch

import numpy as np

from pose_graph_prediction.model.pose_graph_prediction_net import PoseGraphPredictionNet

from typing import Any


def make_deterministic_as_possible():
    """
    It's good practice to fix randomness as much as possible during the development of neural networks to be able to
    compare changes to the network with previous states. This isn't entirely possible because some parallel operations
    on GPUs are inherently nondeterministic, but the other causes of randomness are fixed in this function.
    """
    environ['PYTHONHASHSEED'] = str(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def deterministic_init_function(worker_id: Any):
    np.random.seed(int(0))


def get_model(model_config: dict) -> PoseGraphPredictionNet:
    return PoseGraphPredictionNet(model_config)


def load_config_file(path_to_config_file: str) -> dict:
    if exists(path_to_config_file):
        with open(path_to_config_file) as json_file:
            return load_json_file(json_file)
    else:
        print("Config file does not exist. Exiting. Provided file path is ", path_to_config_file)
        exit(-1)


def load_model_weights(model: Any,
                       path: str):
    """
    Checks whether the network was trained in parallel on several GPUs or not and loads the parameters accordingly.
    This is a workaround due to training with DataParallel and visualizing the training results without it.
    DataParallel adds a "module." prefix to every layer name in the dict.
    To use the network on a CPU or GPU without DataParallel this prefix has to be removed.

    :param model: The model of the network.
    :param path: Path to the trained and stored model parameters.
    :return: Model with trained parameters
    """
    # Load stored network parameters that were trained using DataParallel
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    # Create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    model_was_probably_trained_with_data_parallel = False
    for k, v in state_dict.items():
        if str(k).startswith("module."):
            # Remove `module.` prefix
            name = k[7:]
            new_state_dict[name] = v
            model_was_probably_trained_with_data_parallel = True
        else:
            new_state_dict = state_dict
            model_was_probably_trained_with_data_parallel = False
            break

    if model_was_probably_trained_with_data_parallel:
        print("All names of the model started with \"module.\". It's assumed the model was trained using DataParallel.")
    else:
        print("Model doesn't seem to be trained with DataParallel. Loading parameters the usual way.")

    model.load_state_dict(new_state_dict)
