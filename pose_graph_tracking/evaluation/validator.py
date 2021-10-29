from numpy import mean

import torch
from torch.nn import Module, MSELoss

from torch_geometric.data import Data, DataListLoader, DataLoader, Dataset
from torch_geometric.nn import DataParallel

from pose_graph_tracking.helpers.utils import deterministic_init_function


class Validator(object):
    def __init__(self,
                 model: Module,
                 validation_data: Dataset,
                 training_config: dict):
        """
        Runs a model on a validation dataset and returns the loss.

        :param model: Network model to be validated.
        :param validation_data: Data to validate on.
        :param training_config: Config dict providing the hyperparameters used for training the model.
        """
        # Default parameters
        self.batch_size = 50
        self.try_to_train_on_gpu = False
        self._load_parameters_from_config(training_config)

        self.model = model

        # Use GPU if available and user does not explicitly wants to train on the CPU
        if self.try_to_train_on_gpu:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        # Check if model can be trained in parallel on several GPUs
        self.number_of_gpus = torch.cuda.device_count()
        # If device is explicitly set to cpu although cuda is available, do not parallelize training to several devices
        if self.device.type == "cpu":
            self.number_of_gpus = 0

        if self.number_of_gpus > 1:
            self.model = DataParallel(self.model)

        self.model = self.model.to(self.device)

        self._init_data_loader(validation_data, self.batch_size)

    def _load_parameters_from_config(self,
                                     config: dict):
        self.batch_size = config.get("batch_size", self.batch_size)
        self.try_to_train_on_gpu = config.get("try_to_train_on_gpu", self.try_to_train_on_gpu)

    def _init_data_loader(self,
                          data: Dataset,
                          batch_size: int):
        """
        Depending on the number of GPUs available the data has to be loaded using a DataListLoader or a DataLoader.

        :param data: Dataset.
        :param batch_size: Batch size.
        """
        if self.number_of_gpus > 1:
            self.data_loader = DataListLoader(data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              worker_init_fn=deterministic_init_function)
        else:
            self.data_loader = DataLoader(data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          worker_init_fn=deterministic_init_function)

    def compute_validation_loss(self) -> float:
        """
        Runs the whole validation process.

        :return: Mean loss.
        """
        self.model.eval()

        losses = []
        for data in self.data_loader:
            loss = self._process_data(data)
            losses.append(loss.item())
        return mean(losses).item()

    def _process_data(self, data) -> MSELoss:
        """
        Process the data using the model and compute the loss.

        :param data: Data sample provided to the model.
        :return: Loss.
        """
        # If model is validated on one device, data is provided by a DataLoader and we need to make sure it's on the
        # correct device - CPU or GPU
        if self.number_of_gpus <= 1:
            data = data.to(self.device)

        # Use network to process the data
        model_result = self.model(data)

        # If model is validated on several GPUs in parallel, data is provided by a DataListLoader already on a GPU
        # -> extract all ground truth features into one tensor to compare to results of model in the next step
        if self.number_of_gpus > 1:
            ground_truth = torch.cat([single_graphs_data.ground_truth for single_graphs_data in data]
                                     ).to(model_result.device)
        else:
            ground_truth = data.ground_truth

        # Compute loss as mean squared error between model result and ground truth
        loss = MSELoss()(model_result, ground_truth)
        return loss
