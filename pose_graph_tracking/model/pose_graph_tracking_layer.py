import torch

from torch.nn import Module, Sequential, Linear as Lin, ReLU, LeakyReLU, LayerNorm

from torch_scatter import scatter_add

from torch_geometric.nn import MetaLayer

from typing import Union


class TrackingGraphLayer(Module):
    def __init__(self,
                 activation_type: str,
                 num_input_features_per_node: int,
                 num_node_mlp_layers: int,
                 num_hidden_units_for_node_mlp: int,
                 num_output_features_per_node: int,
                 num_input_features_per_edge: int,
                 num_edge_mlp_layers: int,
                 num_hidden_units_for_edge_mlp: int,
                 num_output_features_per_edge: int,
                 num_global_features: int):
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
        The resulting vector is concatenated to the features of this node and the global features forming an extended
        feature vector.
        Again an MLP is trained to process this extended feature vector to update the nodes' features.

        The updated features and the original global features are returned as the result of this layer.

        :param activation_type: Activation type used for all activation layers within the layer - relu and leaky_relu.
        :param num_input_features_per_node: Number of features describing the state of each node.
        :param num_node_mlp_layers: The number of layers within the MLP of the NodeModel.
        :param num_hidden_units_for_node_mlp: The number of hidden units used in the MLP of the NodeModel.
        :param num_output_features_per_node: Number of features describing the state of each node for output.
        :param num_input_features_per_edge: Number of features describing the state of each edge.
        :param num_edge_mlp_layers: The number of layers within the MLP of the EdgeModel.
        :param num_hidden_units_for_edge_mlp: The number of hidden units used in the MLP of the EdgeModel.
        :param num_output_features_per_edge: Number of features describing the state of each edge for output.
        :param num_global_features: Number of features describing the global state of the graph.
        """
        super(TrackingGraphLayer, self).__init__()

        def init_weights(m: Module):
            negative_slope = 0 if activation_type == "relu" else 1e-2
            if type(m) == Lin:
                torch.nn.init.kaiming_uniform_(m.weight, a=negative_slope, nonlinearity=activation_type)

        class EdgeModel(Module):
            def __init__(self):
                """
                Edge model of the Tracking Graph Layer consisting of an MLP to update the edges' features based on their
                own features, the features of the nodes the edges connect and global features.
                """
                super(EdgeModel, self).__init__()

                edge_mlp_channels_in = 2 * num_input_features_per_node + num_input_features_per_edge + num_global_features
                self.edge_mlp = Sequential()
                self.edge_mlp.add_module("edge_mlp_input", Lin(edge_mlp_channels_in, num_hidden_units_for_edge_mlp))

                if num_edge_mlp_layers > 2:
                    for layer in range(num_edge_mlp_layers - 2):
                        self.addActivationLayerToEdgeMLP(layer, activation_type)
                        self.edge_mlp.add_module("edge_mlp_hidden_" + str(layer), Lin(num_hidden_units_for_edge_mlp,
                                                                                      num_hidden_units_for_edge_mlp))

                self.addActivationLayerToEdgeMLP(num_edge_mlp_layers - 2, activation_type)
                self.edge_mlp.add_module("edge_mlp_output", Lin(num_hidden_units_for_edge_mlp, num_output_features_per_edge))
                self.edge_mlp.add_module("edge_mlp_layer_norm", LayerNorm(num_output_features_per_edge))
                self.edge_mlp.apply(init_weights)

            def addActivationLayerToEdgeMLP(self,
                                            layer_id: int,
                                            activation_type: str = "relu"):
                """
                Adds an activation layer to the edge_mlp.

                :param layer_id: ID of the activation layer to be used within the name of this layer.
                :param activation_type: The activation type - relu or leaky_relu.
                """
                if activation_type == "relu":
                    self.edge_mlp.add_module("edge_mlp_relu_" + str(layer_id), ReLU())
                else:
                    self.edge_mlp.add_module("edge_mlp_leaky_relu_" + str(layer_id), LeakyReLU())

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
                 [feature_0_of_source_node_of_edge_1, feature_1_of_source_node_of_edge_1, ..], .. ]
                with size (number_of_edges_in_current_batch x number_of_nodes_features).

                features_of_target_nodes:
                [[feature_0_of_target_node_of_edge_0, feature_1_of_target_node_of_edge_0, ..],
                 [feature_0_of_target_node_of_edge_1, feature_1_of_target_node_of_edge_1, ..], .. ]
                with size (number_of_edges_in_current_batch x number_of_nodes_features).

                features_of_edges:
                [[feature_0_of_edge_0, feature_1_of_edge_0, ..],
                 [feature_0_of_edge_1, feature_1_of_edge_1, ..], .. ]
                with size (number_of_edges_in_current_batch x number_of_edges_features).

                global_features:
                [[global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..],
                 [global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..], .. ]
                with size (number_of_graphs_in_current_batch x number_of_global_features_per_graph).

                batch_ids:
                [graph_id_of_edge_0, graph_id_of_edge_1, ..]
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
                # TODO: test "attentional" methods that compute/learn a value "a" (usually between 0.0 and 1.0 - softmax
                #  could help here) in the edge model for each node pair and update each node state by computing
                #  u = a * LearnedWeights * state
                #  and then adding the node state by u or adding u to the node state - I tested this already but only
                #  with a rather simple model. Multi head attention would train k models with different parameters and
                #  feed the sum or mean of the results to an MLP to calculate the next state
                if global_features is None:
                    edge_neighborhoods = torch.cat([features_of_source_nodes,
                                                    features_of_target_nodes,
                                                    features_of_edges], 1)
                else:
                    edge_neighborhoods = torch.cat([features_of_source_nodes,
                                                    features_of_target_nodes,
                                                    features_of_edges,
                                                    global_features[batch_ids]], 1)
                return self.edge_mlp(edge_neighborhoods)

        class NodeModel(Module):
            def __init__(self):
                """
                Node model of the Tracking Graph Layer consisting of an MLP to update the nodes' features based on their
                own features, the summed features of all edges pointing to that node and the global features of the
                graph the nodes belongs to.
                """
                super(NodeModel, self).__init__()

                node_mlp_channels_in = num_input_features_per_node + num_input_features_per_edge + num_global_features
                self.node_mlp = Sequential()
                self.node_mlp.add_module("node_mlp_input", Lin(node_mlp_channels_in, num_hidden_units_for_node_mlp))

                if num_node_mlp_layers > 2:
                    for layer in range(num_node_mlp_layers - 2):
                        self.addActivationLayerToNodeMLP(layer, activation_type)
                        self.node_mlp.add_module("node_mlp_hidden_" + str(layer), Lin(num_hidden_units_for_node_mlp,
                                                                                      num_hidden_units_for_node_mlp))

                self.addActivationLayerToNodeMLP(num_node_mlp_layers - 2, activation_type)
                self.node_mlp.add_module("node_mlp_output", Lin(num_hidden_units_for_node_mlp, num_output_features_per_node))
                self.node_mlp.add_module("node_mlp_layer_norm", LayerNorm(num_output_features_per_node))
                self.node_mlp.apply(init_weights)

            def addActivationLayerToNodeMLP(self,
                                            layer_id: int,
                                            activation_type: str = "relu"):
                """
                Adds an activation layer to the node_mlp.

                :param layer_id: ID of the activation layer to be used within the name of this layer.
                :param activation_type: The activation type - relu or leaky_relu.
                """
                if activation_type == "relu":
                    self.node_mlp.add_module("node_mlp_relu_" + str(layer_id), ReLU())
                else:
                    self.node_mlp.add_module("node_mlp_leaky_relu_" + str(layer_id), LeakyReLU())

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
                 [feature_0_of_node_1, feature_1_of_node_1, ..], .. ]
                with size (number_of_nodes_in_current_batch x number_of_nodes_features).

                node_ids_for_edges:
                [[source_node_id_of_edge_0, source_node_id_of_edge_1, ..],
                 [target_node_id_of_edge_0, target_node_id_of_edge_1, ..]]
                with size (2 x number_of_edges_in_current_batch).

                features_of_edges:
                [[feature_0_of_edge_0, feature_1_of_edge_0, ..],
                 [feature_0_of_edge_1, feature_1_of_edge_1, ..], .. ]
                with size (number_of_edges_in_current_batch x number_of_edges_features).

                global_features:
                [[global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..],
                 [global_feature_0_of_graph_0, global_feature_1_of_graph_0, ..], .. ]
                with size (number_of_graphs_in_current_batch x number_of_global_features_per_graph).

                batch_ids:
                [graph_id_of_node_0, graph_id_of_node_1, ..]
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
        Calls EdgeModel and NodeModel in this order to update the features of the nodes and edges using trained MLPs.

        :param features_of_nodes: Tensor of features defining nodes.
        :param node_ids_for_edges: Tensor encoding the connections between nodes by edges.
        :param features_of_edges: Tensor of features for every edge.
        :param global_features: Tensor of global features - one set of features per graph.
        :param batch_ids: Tensor of ids encoding which node belongs to which graph.
        :return: Updated features of nodes and edges, and original global features.
        """
        return self.op.forward(features_of_nodes, node_ids_for_edges, features_of_edges, global_features, batch_ids)
