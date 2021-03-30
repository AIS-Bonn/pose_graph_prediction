from pose_graph_tracking.model.utils import get_activation_function_from_type

import torch

from torch.nn import Dropout, LayerNorm, Linear as Lin, Module, Sequential, Sigmoid

from torch_scatter import scatter_add

from torch_geometric.nn import MetaLayer

from typing import Union


class PoseGraphPredictionLayer(Module):
    def __init__(self,
                 model_config: dict):
        """
        A graph network layer able to process graphs with a different number of nodes and edges.
        All nodes and edges have to have the same number of features respectively.
        The general MetaLayer is inspired by the MLPGraphNetwork from DeepMind.

        First an EdgeModel is applied to update the features of the edges.
        For each edge, its features are concatenated to the features of both nodes it connects and to the global
        features forming an extended feature vector for this edge.
        An MLP is trained to process this extended feature vector to update the edges' features.

        Then a NodeModel is applied to update the nodes' features.
        For each node, all updated edge features of incoming edges are summed.
        Caveat: For nodes without incoming edges a zero vector is used.
        The resulting vector is concatenated to the features of this node and the global features forming an extended
        feature vector.
        Again an MLP is trained to process this extended feature vector to update the nodes' features.

        The updated features and the original global features are returned as the result of this layer.

        :param model_config: Dict containing all parameters for this Layer.
        """
        super(PoseGraphPredictionLayer, self).__init__()

        activation_type = model_config["pose_graph_prediction_layer_parameters"]["activation_type"]
        activation_function = get_activation_function_from_type(activation_type)

        attention_dropout_probability = model_config["attention_dropout_probability"]
        dropout_probability = model_config["dropout_probability"]

        use_attention = model_config["use_attention"]

        number_of_features_per_edge = model_config["edge_encoder_parameters"]["number_of_output_channels"]
        number_of_features_per_node = model_config["node_encoder_parameters"]["number_of_output_channels"]
        number_of_global_features = model_config["pose_graph_prediction_layer_parameters"]["number_of_global_features"]

        edge_mlp_parameters = model_config["pose_graph_prediction_layer_parameters"]["edge_mlp_parameters"]
        node_mlp_parameters = model_config["pose_graph_prediction_layer_parameters"]["node_mlp_parameters"]

        def init_weights(m: Module):
            negative_slope = 0 if activation_type == "relu" else 1e-2
            if type(m) == Lin:
                torch.nn.init.kaiming_uniform_(m.weight, a=negative_slope, nonlinearity=activation_type)

        class EdgeModel(Module):
            def __init__(self):
                """
                Edge model consisting of an MLP to update the edges' features based on their own features, the features
                of the nodes the edges connect and global features.
                """
                super(EdgeModel, self).__init__()

                self.edge_mlp = self.generate_edge_mlp(dropout_probability=dropout_probability)
                if use_attention:
                    self.edge_attention_mlp = self.generate_edge_mlp(dropout_probability=attention_dropout_probability,
                                                                     generate_attention_mlp=True)

            def generate_edge_mlp(self,
                                  dropout_probability: float,
                                  generate_attention_mlp: bool = False) -> Sequential:
                mlp_name = "edge_mlp"
                if generate_attention_mlp:
                    mlp_name = "attention_" + mlp_name

                number_of_input_channels = number_of_features_per_node * 2 + \
                                           number_of_features_per_edge + \
                                           number_of_global_features
                number_of_hidden_channels = edge_mlp_parameters["number_of_hidden_channels"]
                number_of_output_channels = edge_mlp_parameters["number_of_output_channels"]
                number_of_hidden_layers = edge_mlp_parameters["number_of_hidden_layers"]

                edge_mlp = Sequential()
                edge_mlp.add_module(mlp_name + "input_layer",
                                    Lin(number_of_input_channels, number_of_hidden_channels))

                for layer_id in range(number_of_hidden_layers):
                    edge_mlp.add_module(mlp_name + "activation_function_" + str(layer_id), activation_function())
                    edge_mlp.add_module(mlp_name + "hidden_dropout_" + str(layer_id), Dropout(dropout_probability))
                    edge_mlp.add_module(mlp_name + "hidden_layer_" + str(layer_id),
                                             Lin(number_of_hidden_channels, number_of_hidden_channels))

                edge_mlp.add_module(mlp_name + "output_activation_function", activation_function())
                edge_mlp.add_module(mlp_name + "output_dropout", Dropout(dropout_probability))
                edge_mlp.add_module(mlp_name + "output", Lin(number_of_hidden_channels, number_of_output_channels))
                edge_mlp.add_module(mlp_name + "layer_norm", LayerNorm(number_of_output_channels))
                if generate_attention_mlp:
                    edge_mlp.add_module(mlp_name + "output_sigmoid", Sigmoid())
                edge_mlp.apply(init_weights)
                return edge_mlp

            def forward(self,
                        features_of_source_nodes: torch.Tensor,
                        features_of_target_nodes: torch.Tensor,
                        features_of_edges: torch.Tensor,
                        global_features: torch.Tensor,
                        batch_ids: torch.Tensor) -> torch.Tensor:
                """
                For each edge, concatenates the features of the edge's source node, the features of the target node, the
                own features of that edge and the global features of its graph.
                The resulting feature vector is processed by an MLP to become the updated feature vector of that edge.

                The MLP is able to process a batch of concatenated feature vectors separately in parallel.
                For that, a 2D-Tensor is generated containing the concatenated features of one edge per row.

                The inputs have the following format:

                features_of_source_nodes:
                [[feature_0_of_source_node_of_edge_0, feature_1_of_source_node_of_edge_0, ..],
                 [feature_0_of_source_node_of_edge_1, feature_1_of_source_node_of_edge_1, ..],
                 .. ]
                with size (number_of_edges_in_current_batch x number_of_features_per_node).

                features_of_target_nodes:
                [[feature_0_of_target_node_of_edge_0, feature_1_of_target_node_of_edge_0, ..],
                 [feature_0_of_target_node_of_edge_1, feature_1_of_target_node_of_edge_1, ..],
                 .. ]
                with size (number_of_edges_in_current_batch x number_of_features_per_node).

                features_of_edges:
                [[feature_0_of_edge_0, feature_1_of_edge_0, ..],
                 [feature_0_of_edge_1, feature_1_of_edge_1, ..],
                 .. ]
                with size (number_of_edges_in_current_batch x number_of_features_per_edge).

                global_features:
                [[global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..],
                 [global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..],
                 .. ]
                with size (number_of_graphs_in_current_batch x number_of_global_features_per_graph).

                batch_ids:
                [graph_id_of_edge_0, .., graph_id_of_edge_1, ..]
                with size (number_of_edges_in_current_batch). E.g. [0,0,0,1,1] for a batch with 2 graphs consisting of
                3 and 2 edges respectively. Is used to get the correct global features for each edge from the
                global_features tensor containing all sets of global features in the current batch.

                :param features_of_source_nodes: Features of nodes' serving as the sources of edges.
                :param features_of_target_nodes: Features of nodes' serving as the targets of edges.
                :param features_of_edges: Features of edges connecting source and target nodes.
                :param global_features: Global features of graphs - one set of features per graph.
                :param batch_ids: Tensor of ids encoding which node belongs to which graph.
                :return: Updated features of edges.
                """
                if global_features is None:
                    edge_neighborhoods = torch.cat([features_of_source_nodes,
                                                    features_of_target_nodes,
                                                    features_of_edges], 1)
                    resulting_edges = self.edge_mlp(edge_neighborhoods)
                else:
                    edge_neighborhoods = torch.cat([features_of_source_nodes,
                                                    features_of_target_nodes,
                                                    features_of_edges,
                                                    global_features[batch_ids]], 1)
                    resulting_edges = self.edge_mlp(edge_neighborhoods)

                if use_attention:
                    attentions_per_edge = self.edge_attention_mlp(edge_neighborhoods)
                    resulting_edges = resulting_edges * attentions_per_edge

                return resulting_edges

        class NodeModel(Module):
            def __init__(self):
                """
                Node model consisting of an MLP to update the nodes' features based on their own features, the summed
                features of all edges pointing to that node and the global features of the graph the nodes belong to.
                """
                super(NodeModel, self).__init__()

                number_of_input_channels = number_of_features_per_node + \
                                           edge_mlp_parameters["number_of_output_channels"] + \
                                           number_of_global_features
                number_of_hidden_channels = node_mlp_parameters["number_of_hidden_channels"]
                number_of_output_channels = node_mlp_parameters["number_of_output_channels"]
                number_of_hidden_layers = node_mlp_parameters["number_of_hidden_layers"]

                self.node_mlp = Sequential()
                self.node_mlp.add_module("node_mlp_input_layer", Lin(number_of_input_channels, number_of_hidden_channels))

                for layer_id in range(number_of_hidden_layers):
                    self.node_mlp.add_module("node_mlp_activation_function_" + str(layer_id), activation_function())
                    self.node_mlp.add_module("node_mlp_hidden_dropout_" + str(layer_id), Dropout(dropout_probability))
                    self.node_mlp.add_module("node_mlp_hidden_layer_" + str(layer_id),
                                             Lin(number_of_hidden_channels, number_of_hidden_channels))

                self.node_mlp.add_module("node_mlp_output_activation_function", activation_function())
                self.node_mlp.add_module("node_mlp_output_dropout", Dropout(dropout_probability))
                self.node_mlp.add_module("node_mlp_output_layer",
                                         Lin(number_of_hidden_channels, number_of_output_channels))
                self.node_mlp.add_module("node_mlp_layer_norm", LayerNorm(number_of_output_channels))
                self.node_mlp.apply(init_weights)

            def forward(self,
                        features_of_nodes: torch.Tensor,
                        node_ids_for_edges: torch.Tensor,
                        features_of_edges: torch.Tensor,
                        global_features: torch.Tensor,
                        batch_ids: torch.Tensor) -> torch.Tensor:
                """
                For each node, sums up the features of all edges pointing to that node.
                Concatenates the resulting feature vector with the features of the node itself and the global features
                of its graph.
                The concatenated feature vector is processed by an MLP.
                The output vector of that MLP becomes the updated feature vector of the node.

                The MLP is able to process a batch of concatenated feature vectors separately in parallel.
                For that, a 2D-Tensor is generated containing the concatenated features of one node per row.

                The inputs have the following format:

                features_of_nodes:
                [[feature_0_of_node_0, feature_1_of_node_0, ..],
                 [feature_0_of_node_1, feature_1_of_node_1, ..],
                 .. ]
                with size (number_of_nodes_in_current_batch x number_of_features_per_node).

                node_ids_for_edges:
                [[source_node_id_of_edge_0, source_node_id_of_edge_1, ..],
                 [target_node_id_of_edge_0, target_node_id_of_edge_1, ..]]
                with size (2 x number_of_edges_in_current_batch).

                features_of_edges:
                [[feature_0_of_edge_0, feature_1_of_edge_0, ..],
                 [feature_0_of_edge_1, feature_1_of_edge_1, ..],
                 .. ]
                with size (number_of_edges_in_current_batch x number_of_features_per_edge).

                global_features:
                [[global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..],
                 [global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..],
                 .. ]
                with size (number_of_graphs_in_current_batch x number_of_global_features_per_graph).

                batch_ids:
                [graph_id_of_node_0, .., graph_id_of_node_1, ..]
                with size (number_of_nodes_in_current_batch). E.g. [0,0,0,1,1] for a batch with 2 graphs consisting of
                3 and 2 nodes respectively. Is used to get the correct global features for each node from the
                global_features tensor containing all sets of global features in the current batch.

                :param features_of_nodes: Features of the nodes in the current batch.
                :param node_ids_for_edges: IDs of source and target nodes for each edge.
                :param features_of_edges: Features of nodes in the current batch.
                :param global_features: Global features of all graphs on current batch.
                :param batch_ids: Tensor of IDs encoding which node belongs to which graph.
                :return: Updated features of nodes.
                """
                source_node_ids, target_node_ids = node_ids_for_edges

                # Sum up features_of_edges of all edges pointing to the same target node
                # E.g. first entry of result contains sum of all features_of_edges from edges pointing to the first node
                # A feature vector containing all zeros is generated for nodes without incoming edges
                edge_features_summed_by_target = scatter_add(features_of_edges,
                                                             target_node_ids,
                                                             dim=0,
                                                             dim_size=features_of_nodes.size(0))

                if global_features is None:
                    node_neighborhoods = torch.cat([features_of_nodes,
                                                    edge_features_summed_by_target], dim=1)
                else:
                    node_neighborhoods = torch.cat([features_of_nodes,
                                                    edge_features_summed_by_target,
                                                    global_features[batch_ids]], dim=1)
                return self.node_mlp(node_neighborhoods)

        self.op = MetaLayer(EdgeModel(), NodeModel())

    def forward(self,
                features_of_nodes: torch.Tensor,
                node_ids_for_edges: torch.Tensor,
                features_of_edges: torch.Tensor,
                global_features: Union[torch.Tensor, None] = None,
                batch_ids: Union[torch.Tensor, None] = None) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calls EdgeModel and NodeModel in this order to update the features of the edges and nodes using trained MLPs.

        :param features_of_nodes: Tensor of features defining nodes.
        :param node_ids_for_edges: Tensor encoding the connections between nodes by edges.
        :param features_of_edges: Tensor of features for every edge.
        :param global_features: Tensor of global features - one set of features per graph.
        :param batch_ids: Tensor of ids encoding which node belongs to which graph.
        :return: Updated features of nodes and edges, and original global features.
        """
        return self.op.forward(features_of_nodes, node_ids_for_edges, features_of_edges, global_features, batch_ids)
