import torch

from torch.nn import Module, ReLU, LeakyReLU, Sigmoid

from torch_geometric.data import Data

from pose_graph_tracking.model.pose_graph_prediction_layer import PoseGraphPredictionLayer
from pose_graph_tracking.model.utils import generate_encoder, generate_decoder

from typing import Tuple


# TODO: das ganze ab hier weiter an pose graph tracking anpassen - eigentlich sollte es PoseGraphPrediction heißen ..
class PoseGraphPredictionNet(Module):
    def __init__(self,
                 model_config: dict,  # TODO: load parameters from config
                 num_features_per_node: int = 6,
                 num_global_features: int = 1,
                 activation_type: str = "relu",
                 num_encoder_layers: int = 1,
                 num_encoded_features_per_node: int = 16,
                 num_node_mlp_layers: int = 3,
                 num_hidden_units_for_node_mlp: int = 100,
                 num_features_per_node_after_prediction: int = 16,
                 num_features_per_node_after_association: int = 16,
                 num_encoded_features_per_edge: int = 16,
                 num_edge_mlp_layers: int = 4,
                 num_hidden_units_for_edge_mlp: int = 150,
                 num_features_per_edge_after_prediction: int = 16,
                 num_features_per_edge_after_association: int = 16,
                 num_encoded_global_features: int = 16,
                 num_decoder_hidden_layers: int = 1,
                 append_sigmoid_to_association_edges_decoder: bool = False):
        """
        Generates network layers and subnetworks necessary for the Pose Graph Prediction network.

        The network consists of several encoders for each type of features (node, edge and global features), followed by
        a PoseGraphPredictionLayer to predict the next states of the hypotheses, a PoseGraphPredictionLayer to associate
        corresponding detection-hypothesis pairs and to update the hypotheses states, and a decoder for the hypotheses'
        states to convert them to their original format as well as a decoder for the processed
        association_edges_features to a value indication which detection(s) correspond to which hypothesis.

        :param num_features_per_node: Number of features per node
          - for bouncing balls: position_x,    position_y,    velocity_x,        velocity_y,        radius
          - for MOT challenge : bb_center_col, bb_center_row, bb_center_col_vel, bb_center_row_vel, bb_width, bb_height
        :param num_global_features: Number of global features per graph
          - for bouncing balls: gravitational_center_x,    gravitational_center_y,    duration_between_frames
          - for MOT challenge : duration_between_frames
        :param activation_type: Activation type used for all activation layers within the network - relu and leaky_relu.
        :param num_encoder_layers: The number of layers within the encoders.
        :param num_encoded_features_per_node: The number of features per node after being encoded.
        :param num_node_mlp_layers: The number of layers within the node MLP of the PoseGraphPredictionLayer.
        :param num_hidden_units_for_node_mlp: The number of hidden units used in the node MLP of the PoseGraphPredictionLayer.
        :param num_features_per_node_after_prediction: The number of features per node after the Prediction-TGL.
        :param num_features_per_node_after_association: The number of features per node after the Association-TGL.
        :param num_encoded_features_per_edge: The number of features per edge after being encoded.
        :param num_edge_mlp_layers: The number of layers within the edge MLP of the PoseGraphPredictionLayer.
        :param num_hidden_units_for_edge_mlp: The number of hidden units used in the edge MLP of the PoseGraphPredictionLayer.
        :param num_features_per_edge_after_prediction: The number of features per edge after the Prediction-TGL.
        :param num_features_per_edge_after_association: The number of features per edge after the Association-TGL.
        :param num_encoded_global_features: The number of global features after being encoded.
        :param append_sigmoid_to_association_edges_decoder: If true, appends a sigmoid layer to the association edge
        decoder to ease up training, as the resulting values are always between 0 and 1.
       """
        super(PoseGraphPredictionNet, self).__init__()

        available_activation_types = ["relu", "leaky_relu"]
        assert activation_type in available_activation_types, "Requested activation type %r is not an available " \
                                                              "activation type: %r" % (activation_type,
                                                                                       available_activation_types)
        if activation_type == "relu":
            self.activation_function = ReLU
        else:
            self.activation_function = LeakyReLU

        # Specifying number of features for edges and globals
        num_features_per_prediction_edge = 1  # distance between the two nodes this edge connects
        num_features_per_association_edge = 1  # distance between the two nodes this edge connects

        num_global_features_for_prediction = num_global_features

        # Encoders for node-, edge- and global_features for the prediction step
        self.node_features_encoder = generate_encoder(num_features_per_node,
                                                      num_encoded_features_per_node,
                                                      num_encoded_features_per_node,
                                                      self.activation_function,
                                                      num_encoder_layers)
        self.prediction_edge_features_encoder = generate_encoder(num_features_per_prediction_edge,
                                                                 num_encoded_features_per_edge,
                                                                 num_encoded_features_per_edge,
                                                                 self.activation_function,
                                                                 num_encoder_layers)
        self.prediction_global_features_encoder = generate_encoder(num_global_features_for_prediction,
                                                                   num_encoded_global_features,
                                                                   num_encoded_global_features,
                                                                   self.activation_function,
                                                                   num_encoder_layers)

        # Predicts the states of the hypotheses nodes at the next time step
        self.tracking_graph_layer_for_prediction = PoseGraphPredictionLayer(self.activation_type,
                                                                            num_encoded_features_per_node,
                                                                            num_node_mlp_layers,
                                                                            num_hidden_units_for_node_mlp,
                                                                            num_features_per_node_after_prediction,
                                                                            num_encoded_features_per_edge,
                                                                            num_edge_mlp_layers,
                                                                            num_hidden_units_for_edge_mlp,
                                                                            num_features_per_edge_after_prediction,
                                                                            num_encoded_global_features)

        # Encoders for edge_features for the association step
        self.association_edge_features_encoder = generate_encoder(num_features_per_association_edge,
                                                                  num_encoded_features_per_edge,
                                                                  num_encoded_features_per_edge,
                                                                  self.activation_function,
                                                                  num_encoder_layers)

        # Associates corresponding detections and hypotheses and updates the hypotheses' states accordingly
        self.tracking_graph_layer_for_association = PoseGraphPredictionLayer(self.activation_type,
                                                                             num_features_per_node_after_prediction * 2,
                                                                             num_node_mlp_layers,
                                                                             num_hidden_units_for_node_mlp,
                                                                             num_features_per_node_after_association,
                                                                             num_encoded_features_per_edge,
                                                                             num_edge_mlp_layers,
                                                                             num_hidden_units_for_edge_mlp,
                                                                             num_features_per_edge_after_association,
                                                                             num_global_features=0)

        # Decoders for features_of_nodes and features_of_association_edges
        # Decodes node features to their original number and meaning
        self.node_decoder = generate_decoder(num_features_per_node_after_association,
                                             num_features_per_node_after_association,
                                             num_features_per_node,
                                             self.activation_function,
                                             num_decoder_hidden_layers)

        # Decodes association edge features to represent association values between 0 and 1
        self.edge_decoder = generate_decoder(num_features_per_edge_after_association,
                                             num_features_per_edge_after_association,
                                             1,
                                             self.activation_function,
                                             num_decoder_hidden_layers)
        if append_sigmoid_to_association_edges_decoder:
            self.edge_decoder.add_module("decoder_sigmoid", Sigmoid())

    def forward(self,
                data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the network structure - how data is processed.

        Each type of features is encoded using a designated encoder network.
        Next, the next hypotheses states are predicted using a Tracking Graph Net.
        Updated edges and global features are discarded, because we are not interested in those.
        Because the MetaLayer is updating every node feature, even those not targeted by any edges, the detections are
        updated as well.
        FIXME : change this behaviour.
        That's why we merge the predicted hypotheses with the original encoded detections afterwards.

        This graph, consisting of hypothesis and detection nodes, is then used to associate corresponding
        hypothesis-detection pairs and update the former wrt. the associated detection, all by a second Tracking Graph
        Net.
        The hypotheses are then decoded using a decoder network and returned as the result.

        :param data: Input data.
        :return: Processed data - more precisely hypotheses nodes.
        """
        # Encode features for the prediction step
        encoded_features_of_nodes = self.node_features_encoder(data.x)
        encoded_features_of_prediction_edges = self.prediction_edge_features_encoder(data.prediction_edges_features)
        encoded_global_features_for_prediction = self.prediction_global_features_encoder(data.prediction_global_features)

        # Predict next states of nodes - only states of hypotheses nodes are updated here.
        predicted_nodes, _, _ = self.tracking_graph_layer_for_prediction.forward(encoded_features_of_nodes,
                                                                                 data.node_indexes_for_prediction_edges,
                                                                                 encoded_features_of_prediction_edges,
                                                                                 encoded_global_features_for_prediction,
                                                                                 data.batch)

        # Even though nodes without incoming edges get zeros as summed edge attributes these zeros are treated as normal
        # values. These zeros for summed edge features + global features + node features are used in node mlp to update
        # node features of all nodes. That's why we need to use the original encoded detections for associations next.

        # TODO: Is it possible to update only target nodes - not update those without incoming edges or update them
        #  differently?
        # TODO: If all nodes are updated -> all nodes are updated during prediction, which was my guess why connecting
        #  only neighbors within a specified distance went wrong -> was i wrong? Using two mlps, one for target nodes
        #  and one for those not targeted doesn't seem to be an option.

        # Concatenate predicted hypotheses nodes from first PoseGraphPredictionLayer with detections from encoded input data
        hypotheses_start_index = 0
        predicted_encoded_features_of_nodes = []
        # There is a batch of several graphs in the data object -> all nodes of all those graphs are stored in the
        # updated_nodes tensor (hypotheses_of_graph_0, then detections_of_graph_0, then hypotheses_of_graph_1, ...).
        # We use the stored number of hypotheses and detections per graph to extract only the hypotheses from the nodes.
        for data_sample_index in range(data.num_hypotheses.shape[0]):
            num_hypotheses = data.num_hypotheses[data_sample_index].item()
            num_detections = data.num_detections[data_sample_index].item()
            hypotheses_end_index = hypotheses_start_index + int(num_hypotheses)
            detections_end_index = hypotheses_end_index + int(num_detections)

            predicted_encoded_features_of_nodes.append(torch.cat(
                [predicted_nodes[hypotheses_start_index: hypotheses_end_index],
                 encoded_features_of_nodes[hypotheses_end_index: detections_end_index]]))
            # Prepare next iteration for extraction of hypotheses and detections from next graph in batch
            hypotheses_start_index = detections_end_index
        predicted_encoded_features_of_nodes = torch.cat(predicted_encoded_features_of_nodes)

        # TODO: use appearances or difference of appearances as edge features - currently only the distance is used for
        #  edge features
        encoded_appearances_of_nodes = self.node_appearance_encoder(data.node_appearances)
        concatenated_features_of_nodes = torch.cat([predicted_encoded_features_of_nodes,
                                                    encoded_appearances_of_nodes], 1)
        # Encode edge and global features for the association step
        encoded_features_of_association_edges = self.association_edge_features_encoder(data.association_edges_features)

        # Association and Update Graph Net
        updated_nodes, updated_association_edges_features, _ = self.tracking_graph_layer_for_association.forward(
            concatenated_features_of_nodes,
            data.node_indexes_for_association_edges,
            encoded_features_of_association_edges)

        # Extract hypotheses from all nodes
        hypotheses_start_index = 0
        updated_encoded_hypotheses_nodes = []
        for data_sample_index in range(data.num_hypotheses.shape[0]):
            num_hypotheses = data.num_hypotheses[data_sample_index].item()
            num_detections = data.num_detections[data_sample_index].item()
            hypotheses_end_index = hypotheses_start_index + int(num_hypotheses)
            nodes_end_index = hypotheses_end_index + int(num_detections)
            updated_encoded_hypotheses_nodes.append(torch.cat([updated_nodes[hypotheses_start_index:
                                                                             hypotheses_end_index]]))
            hypotheses_start_index = nodes_end_index
        updated_encoded_hypotheses = torch.cat(updated_encoded_hypotheses_nodes)

        # Decode hypotheses and associations
        updated_hypotheses = self.node_decoder(updated_encoded_hypotheses)
        predicted_associations = self.edge_decoder(updated_association_edges_features)

        # TODO: Apply Softmax to predicted associations? - Test effect of appending a sigmoid layer vs. softmax vs. both

        return updated_hypotheses, predicted_associations
