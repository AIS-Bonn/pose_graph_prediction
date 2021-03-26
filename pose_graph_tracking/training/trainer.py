from datetime import datetime

from numpy import mean, ndarray

from os import makedirs
from os.path import exists

from pickle import dump

import torch
from torch.nn import Module, MSELoss
from torch.optim import Adam

from torch_geometric.data import Data, DataListLoader, DataLoader, Dataset
from torch_geometric.nn import DataParallel

from pose_graph_tracking.helpers.utils import deterministic_init_function
from pose_graph_tracking.helpers.defaults import MODEL_DIRECTORY, MODEL_NAME_PREFIX

from typing import Callable, Union


class Trainer(object):
    def __init__(self,
                 trainee: Module,
                 training_data: Dataset,
                 test_data: Dataset,
                 training_config: dict,
                 visualization_function: Union[Callable, None] = None):
        """
        Trains a trainee on the training_data and evaluates its performance on the test_data.

        :param trainee: Network model to be trained.
        :param training_data: Data to train on.
        :param test_data: Data to evaluate the performance.
        :param training_config: Config dict providing the hyperparameters for training.
        :param visualization_function: Optionally a visualization function can be provided to visualize the training.
        """
        # Default parameters
        self.batch_size = 50
        self.learning_rate = 0.001
        self.number_of_epochs = 2000000
        self.try_to_train_on_gpu = False
        self._load_parameters_from_config(training_config)

        self.trainee = trainee

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
            self.trainee = DataParallel(self.trainee)

        self.trainee = self.trainee.to(self.device)

        self.optimizer = Adam(self.trainee.parameters(), lr=self.learning_rate)

        self._init_training_data_loader(training_data, self.batch_size)
        self._init_test_data_loader(test_data, self.batch_size)

        self.visualization_function = visualization_function

        # TODO: extract from Trainer
        if exists(MODEL_DIRECTORY):
            print("\nSAVING TRAINED MODELS TO EXISTING MODEL DIRECTORY! STOP TRAINING IF OVERWRITE NOT INTENDED!\n")
        else:
            makedirs(MODEL_DIRECTORY)

    def _load_parameters_from_config(self,
                                     config: dict):
        self.batch_size = config.get("batch_size", self.batch_size)
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self.number_of_epochs = config.get("number_of_epochs", self.number_of_epochs)
        self.try_to_train_on_gpu = config.get("try_to_train_on_gpu", self.try_to_train_on_gpu)

    def _init_training_data_loader(self,
                                   training_data: Dataset,
                                   batch_size: int):
        """
        Depending on the number of GPUs available the data has to be loaded using a DataListLoader or a DataLoader.

        :param training_data: Training dataset.
        :param batch_size: Batch size for training.
        """
        if self.number_of_gpus > 1:
            self.training_data_loader = DataListLoader(training_data,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       worker_init_fn=deterministic_init_function)
        else:
            self.training_data_loader = DataLoader(training_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   worker_init_fn=deterministic_init_function)

    def _init_test_data_loader(self,
                               test_data: Dataset,
                               batch_size: int):
        """
        Depending on the number of GPUs available the data has to be loaded using a DataListLoader or a DataLoader.

        :param test_data: Test dataset.
        :param batch_size: Batch size for training.
        """
        if self.number_of_gpus > 1:
            self.test_data_loader = DataListLoader(test_data,
                                                   batch_size=batch_size,
                                                   worker_init_fn=deterministic_init_function)
        else:
            self.test_data_loader = DataLoader(test_data,
                                               batch_size=batch_size,
                                               worker_init_fn=deterministic_init_function)

    def run_training_session(self):
        """
        Runs the whole training process for the requested number of epochs.
        """
        results = []
        for epoch in range(self.number_of_epochs):
            # Train model on whole training set and return mean loss for this epoch
            train_loss = self.train()
            # Compute loss on test set after this epoch of training
            test_loss = self.test()

            results.append([epoch, train_loss, test_loss])
            self._save_results_to_file(results)

            # Save model and print results
            if epoch % 1 == 0:
                self.save_model(str(epoch))
                print(datetime.now(),
                      ' epoch ', epoch,
                      ' train_loss: ', train_loss,
                      ' test_loss: ', test_loss)

    def train(self) -> float:
        """
        Runs training for one epoch.

        :return: Mean loss.
        """
        self.trainee.train()

        losses = []
        for data in self.training_data_loader:
            self.optimizer.zero_grad()

            loss = self._process_data(data)
            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())
        return mean(losses).item()

    def test(self) -> float:
        """
        Run evaluation on test data for one epoch.

        :return: Mean loss.
        """
        self.trainee.eval()

        losses = []
        for data in self.test_data_loader:
            loss = self._process_data(data)
            losses.append(loss.item())
        return mean(losses).item()

    def _process_data(self, data) -> MSELoss:
        """
        Process the data by the trainee model and compute the loss.
        Optionally visualizes the results of the model, if a visualization function was provided to the constructor.

        :param data: Data sample provided to the model.
        :return: Loss.
        """
        # If model is trained on one device, data is provided by a DataLoader and we need to make sure it's on the
        # correct device - CPU or GPU
        if self.number_of_gpus <= 1:
            data = data.to(self.device)

        # Use network to process the data
        model_result = self.trainee(data)

        # If model is trained on several GPUs in parallel, data is provided by a DataListLoader already on a GPU
        # -> extract all ground truth features into one tensor to compare to results of model in the next step
        if self.number_of_gpus > 1:
            ground_truth = torch.cat([single_graphs_data.ground_truth for single_graphs_data in data]
                                     ).to(model_result.device)
        else:
            ground_truth = data.ground_truth

        # Compute loss as mean squared error between model result and ground truth
        loss = MSELoss()(model_result, ground_truth)

        if self.visualization_function is not None:
            self.visualize_model_output(model_result.detach().numpy(),
                                        ground_truth.detach().numpy(),
                                        data)

        return loss

    def _save_results_to_file(self,
                              results: List[List[float]]):
        f = open(MODEL_DIRECTORY + "results", 'wb')
        dump(results, f)
        f.close()

    def save_model(self,
                   model_id: str):
        torch.save(self.trainee.state_dict(), MODEL_DIRECTORY + MODEL_NAME_PREFIX + str(model_id) + ".model")

    def visualize_model_output(self,
                               model_result: ndarray,
                               ground_truth: ndarray,
                               data: Data):
        """
        # TODO: implement
        Visualizes the results of the model.

        :param model_result: Result of the model.
        :param ground_truth: Ground truth.
        :param data: Original data sample provided to the model.
        """
        raise NotImplementedError
        # self.visualization_function(model_result,
        #                             ground_truth,
        #                             data)
