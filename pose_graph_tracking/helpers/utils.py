from json import load as load_json_file

from os import environ
from os.path import exists

import random

import torch

import numpy as np

from pose_graph_tracking.model.pose_graph_tracking_net import PoseGraphTrackingNet

from typing import Any


def makeDeterministicAsPossible():
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


def deterministicInitFn(worker_id: Any):
    np.random.seed(int(0))


def getModel(model_config: dict) -> PoseGraphTrackingNet:
    return PoseGraphTrackingNet(model_config)


def load_config_file(path_to_config_file: str) -> dict:
    if exists(path_to_config_file):
        with open(path_to_config_file) as json_file:
            return load_json_file(json_file)
    else:
        print("Config file does not exist. Exiting. Provided file path is ", path_to_config_file)
        exit(-1)
