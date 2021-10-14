import torch
from torch import Tensor
from torch.nn import Module

from typing import Optional


class HeterogeneousMetaLayer(Module):
    r"""
    TODO: Update documentation

    A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    A graph network takes a graph as input and returns an updated graph as
    output (with same connectivity).
    The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
    as well as global-level features :obj:`u`.
    The output graph has the same structure, but updated features.

    Edge features, node features as well as global features are updated by
    calling the modules :obj:`edge_model`, :obj:`node_model` and
    :obj:`global_model`, respectively.

    To allow for batch-wise graph processing, all callable functions take an
    additional argument :obj:`batch`, which determines the assignment of
    edges or nodes to their specific graphs.

    Args:
        edge_model (Module, optional): A callable which updates a graph's edge
            features based on its source and target node features, its current
            edge features and its global features. (default: :obj:`None`)
        node_model (Module, optional): A callable which updates a graph's node
            features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (Module, optional): A callable which updates a graph's
            global features based on its node features, its graph connectivity,
            its edge features and its current global features.

    .. code-block:: python

        from torch.nn import Sequential as Seq, Linear as Lin, ReLU
        from torch_scatter import scatter_mean
        from torch_geometric.nn import HeterogeneousMetaLayer

        class EdgeModel(torch.nn.Module):
            def __init__(self):
                super(EdgeModel, self).__init__()
                self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, src, dest, edge_attr, u, batch):
                # source, target: [E, F_x], where E is the number of edges.
                # edge_attr: [E, F_e]
                # u: [B, F_u], where B is the number of graphs.
                # batch: [E] with max entry B - 1.
                out = torch.cat([src, dest, edge_attr, u[batch]], 1)
                return self.edge_mlp(out)

        class NodeModel(torch.nn.Module):
            def __init__(self):
                super(NodeModel, self).__init__()
                self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
                self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                row, col = edge_index
                out = torch.cat([x[row], edge_attr], dim=1)
                out = self.node_mlp_1(out)
                out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
                out = torch.cat([x, out, u[batch]], dim=1)
                return self.node_mlp_2(out)

        class GlobalModel(torch.nn.Module):
            def __init__(self):
                super(GlobalModel, self).__init__()
                self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
                return self.global_mlp(out)

        op = HeterogeneousMetaLayer(EdgeModel(), NodeModel(), GlobalModel())
        x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
    """
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(HeterogeneousMetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self,
                features_of_nodes: Tensor,
                node_ids_for_edges: Tensor,
                features_of_edges: Optional[Tensor] = None,
                global_features: Optional[Tensor] = None,
                node_type_ids: Optional[Tensor] = None,
                edge_type_ids: Optional[Tensor] = None,
                batch_ids: Optional[Tensor] = None):
        """"""
        source_node_ids, target_node_ids = node_ids_for_edges

        # If edge type ids are not specified, every edge is of type zero
        if edge_type_ids is None:
            edge_type_ids = torch.zeros(node_ids_for_edges.shape[-1])

        if self.edge_model is not None:
            features_of_edges = self.edge_model(features_of_nodes[source_node_ids],
                                                features_of_nodes[target_node_ids],
                                                features_of_edges,
                                                global_features,
                                                edge_type_ids,
                                                batch_ids if batch_ids is None else batch_ids[source_node_ids])

        # If node type ids are not specified, every node is of type zero
        if node_type_ids is None:
            node_type_ids = torch.zeros(features_of_nodes.shape[0])

        if self.node_model is not None:
            features_of_nodes = self.node_model(features_of_nodes,
                                                node_ids_for_edges,
                                                features_of_edges,
                                                global_features,
                                                node_type_ids,
                                                batch_ids)

        if self.global_model is not None:
            global_features = self.global_model(features_of_nodes,
                                                node_ids_for_edges,
                                                features_of_edges,
                                                global_features,
                                                batch_ids)

        return features_of_nodes, features_of_edges, global_features

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)
