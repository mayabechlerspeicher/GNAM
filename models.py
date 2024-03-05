import torch
from torch_geometric.nn import GraphConv, GCNConv, GINConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F

from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import floyd_warshall

class GATv2Model(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None,
                 bias=True, dropout=0.0, graph_level_task=False):
        super(GATv2Model, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.graph_convs.append(GATv2Conv(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          bias=bias))
        for l in range(1, num_layers):
            self.graph_convs.append(GATv2Conv(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              bias=bias))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GCNModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None,
                 bias=True, dropout=0.0, graph_level_task=False):
        super(GCNModel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.graph_convs.append(GCNConv(in_channels=in_channels,
                                        out_channels=hidden_channels,
                                        bias=bias))
        for l in range(1, num_layers):
            self.graph_convs.append(GCNConv(in_channels=hidden_channels,
                                            out_channels=hidden_channels,
                                            bias=bias))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # nn.init.normal_(param, mean=0, std=self.init_std)
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GraphConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 graph_level_task=False):
        super(GraphConvModel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.lns = []
        self.graph_convs.append(GraphConv(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          bias=bias))
        self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))
        for l in range(1, num_layers):
            self.graph_convs.append(GraphConv(in_channels=hidden_channels,
                                              out_channels=hidden_channels,
                                              bias=bias))
            self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.lns = nn.ModuleList(self.lns)
        if graph_level_task: self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)
            h = self.lns[l](x=h, batch=batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GINModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0,
                 graph_level_task=False):
        super(GINModel, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.graph_level_task = graph_level_task
        self.graph_convs = []
        self.lns = []
        nn_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=bias), nn.ReLU())
        self.graph_convs.append(GINConv(in_channels=in_channels,
                                        out_channels=hidden_channels,
                                        bias=bias, nn=nn_layer))
        self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))
        for l in range(1, num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU())
            self.graph_convs.append(GINConv(in_channels=hidden_channels,
                                            out_channels=hidden_channels,
                                            bias=bias, nn=nn_layer))
            self.lns.append(pyg.nn.norm.LayerNorm(hidden_channels, affine=False))

        self.graph_convs = nn.ModuleList(self.graph_convs)
        self.lns = nn.ModuleList(self.lns)
        if graph_level_task: self.pool = global_mean_pool
        self.readout = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        h = x.clone()
        for l in range(self.num_layers):
            h = self.graph_convs[l](x=h, edge_index=edge_index)
            h = self.activation(h)
            h = self.lns[l](x=h, batch=batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_level_task:
            h = self.pool(h, batch)

        y = self.readout(h)

        return y


class GNAM(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels=None, bias=True, dropout=0.0):
        super(GNAM, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.lns = []
        self.fs = nn.ModuleList([
            nn.Linear(1, 1, bias=bias) for i in range(in_channels)]) #a one layer network for each feature. #TODO: change each to a network of num_layers linear layer with hidden_channels hidden dimensions
        self.m = self.f = nn.Sequential(
            nn.Linear(1, 1, bias=bias))

    def forward(self, inputs):
        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch

        fx = torch.stack([fl(x[:, l]) for (l, fl) in enumerate(self.fs)])
        f_sums = fx.sum(dim=1)
        # Ron: compute this every forward pass? # Maya: We can actually compute it once and attach it to the pyg data object, then use is as inputs.distances or something like that
        adj = to_scipy_sparse_matrix(edge_index)
        node_distances = torch.from_numpy(floyd_warshall(adj))
        m_dist = self.m(node_distances)
        out = torch.matmul(m_dist, f_sums)

        # apply f_i for each feature i, to do this efficiently you can apply f_i for the column i of the feature matrix x.
        # initialize here some empty matrix of size x to store the stacked f(x). this matrix will be the same as x but with transformed values using fs.
        # for feature_index in range(x.size(1):
        # TODO: apply f_i for each feature i, to do this efficiently you can apply f_i for the column i of the feature matrix x.
        # initialize here some empty matrix of size x to store the stacked f(x). this matrix will be the same as x but with transformed values using fs.


        # for feature_index in range(x.size(1)):
        #     self.fs[feature_index](x[-1, feature_index])

        # TODO: sum the transformed features for each node


        # TODO: compute node distances matrix between all nodes (this is a matrix)


        # TODO: apply self.m to the distanced for value in this matrix, this can be done efficiently

        # TODO:weight sum the above sum using the transformed distances for eahc node

        # TODO: the output should be a scalar for each node

        return out
