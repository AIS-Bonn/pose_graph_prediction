import torch

from torch.nn import Module

from torch_geometric.data import Data

from pose_graph_tracking.model.pose_graph_prediction_layer import PoseGraphPredictionLayer
from pose_graph_tracking.model.utils import generate_encoder, generate_decoder


class PoseGraphPredictionNet(Module):
    def __init__(self,
                 model_config: dict):
        """
        Generates network layers and subnetworks necessary for the Pose Graph Prediction network.

        The network consists of one encoder for the edge features and one encoder for the node features, followed by
        a PoseGraphPredictionLayer to predict the next states of the joints, and a decoder for the joint positions.

        :param model_config: A dict containing all necessary model parameters.
        """
        super(PoseGraphPredictionNet, self).__init__()

        # Default parameters
        self.dropout_probability = 0.5
        self.edge_encoder_parameters = {"activation_type": "relu",
                                        "number_of_input_channels": 5,
                                        "number_of_hidden_channels": 50,
                                        "number_of_output_channels": 20,
                                        "number_of_hidden_layers": 3}
        self.node_encoder_parameters = {"activation_type": "relu",
                                        "number_of_input_channels": 4,
                                        "number_of_hidden_channels": 50,
                                        "number_of_output_channels": 20,
                                        "number_of_hidden_layers": 3}
        self.node_decoder_parameters = {"activation_type": "relu",
                                        "number_of_input_channels": 20,
                                        "number_of_hidden_channels": 50,
                                        "number_of_output_channels": 3,
                                        "number_of_hidden_layers": 3}
        self._get_parameters_from_config(model_config)

        # Encoders for edge- and node features
        dropout_prob_encoders = 0.0
        self.edge_features_encoder = generate_encoder(self.edge_encoder_parameters["number_of_input_channels"],
                                                      self.edge_encoder_parameters["number_of_hidden_channels"],
                                                      self.edge_encoder_parameters["number_of_output_channels"],
                                                      self.edge_encoder_parameters["activation_type"],
                                                      dropout_prob_encoders,
                                                      self.edge_encoder_parameters["number_of_hidden_layers"])
        self.node_features_encoder = generate_encoder(self.node_encoder_parameters["number_of_input_channels"],
                                                      self.node_encoder_parameters["number_of_hidden_channels"],
                                                      self.node_encoder_parameters["number_of_output_channels"],
                                                      self.node_encoder_parameters["activation_type"],
                                                      dropout_prob_encoders,
                                                      self.node_encoder_parameters["number_of_hidden_layers"])

        # PoseGraphPredictionLayer to predict the encoded joint positions at the next time step
        self.pose_graph_prediction_layer = PoseGraphPredictionLayer(model_config)

        # Decoder to retrieve the predicted joint positions in x,y,z format
        dropout_prob_decoder = 0.0
        self.node_decoder = generate_decoder(self.node_decoder_parameters["number_of_input_channels"],
                                             self.node_decoder_parameters["number_of_hidden_channels"],
                                             self.node_decoder_parameters["number_of_output_channels"],
                                             self.node_decoder_parameters["activation_type"],
                                             dropout_prob_decoder,
                                             self.node_decoder_parameters["number_of_hidden_layers"])

    def _get_parameters_from_config(self,
                                    model_config: dict):
        self.dropout_probability = model_config.get("dropout_probability", self.dropout_probability)
        self._get_mlp_parameters(self.edge_encoder_parameters, model_config["edge_encoder_parameters"])
        self._get_mlp_parameters(self.node_encoder_parameters, model_config["node_encoder_parameters"])
        self._get_mlp_parameters(self.node_decoder_parameters, model_config["node_decoder_parameters"])

    def _get_mlp_parameters(self,
                            mlp_parameters: dict,
                            config: dict):
        mlp_parameters["activation_type"] = config.get("activation_type",
                                                       mlp_parameters[
                                                           "activation_type"])
        mlp_parameters["number_of_input_channels"] = config.get("number_of_input_channels",
                                                                mlp_parameters[
                                                                    "number_of_input_channels"])
        mlp_parameters["number_of_hidden_channels"] = config.get("number_of_hidden_channels",
                                                                 mlp_parameters[
                                                                     "number_of_hidden_channels"])
        mlp_parameters["number_of_output_channels"] = config.get("number_of_output_channels",
                                                                 mlp_parameters[
                                                                     "number_of_output_channels"])
        mlp_parameters["number_of_hidden_layers"] = config.get("number_of_hidden_layers",
                                                               mlp_parameters[
                                                                   "number_of_hidden_layers"])

    def forward(self,
                data: Data) -> torch.Tensor:
        """
        Defines the network structure - how data is processed.

        :param data: Input data.
        :return: Predicted joint positions.
        """
        encoded_features_of_edges = self.edge_features_encoder(data.features_of_edges)
        encoded_features_of_nodes = self.node_features_encoder(data.x)

        # TODO: add node and edge type ids to the data
        residuals_of_node_features, _, _ = self.pose_graph_prediction_layer.forward(encoded_features_of_nodes,
                                                                                    data.node_type_ids,
                                                                                    data.node_indexes_connected_by_edges,
                                                                                    encoded_features_of_edges,
                                                                                    data.edge_type_ids,
                                                                                    global_features=None,
                                                                                    batch_ids=data.batch)

        predicted_encoded_nodes = encoded_features_of_nodes + residuals_of_node_features

        predicted_nodes = self.node_decoder(predicted_encoded_nodes)
        return predicted_nodes
