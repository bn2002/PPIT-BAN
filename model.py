
from collections.abc import Sequence

import torch
from torch import nn
from torch_scatter import scatter_add

from torchdrug import core, layers
from torchdrug.core import Registry as R
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from ban import BANLayer


from torchdrug.models import GCN
from torch.nn.utils.weight_norm import weight_norm

class ConvsLayer(torch.nn.Module):

    def __init__(self, emb_dim):
        super(ConvsLayer, self).__init__()
        self.embedding_size = emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=512, kernel_size=3)
        self.mx1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3)
        self.mx2 = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3)
        #self.bn = nn.BatchNorm1d(256)
        self.mx3 = nn.MaxPool1d(130, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        features = self.relu(self.conv1(x))
        features = self.mx1(features)
        features = self.mx2(self.relu(self.conv2(features)))
        features = self.relu(self.conv3(features))
        features_local = features.permute(0, 2, 1)
        features = self.mx3(features)
        features_global = features.squeeze(2)
        return features_local, features_global




class GeometryAwareRelationalGraphNeuralNetwork(nn.Module, core.Configurable):
    """
    Geometry Aware Relational Graph Neural Network proposed in
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_angle_bin (int, optional): number of bins to discretize angles between edges.
            The discretized angles are used as relations in edge message passing.
            If not provided, edge message passing is disabled.
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GeometryAwareRelationalGraphNeuralNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "max":
            self.readout = layers.MaxReadout()


        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }

class PPI_BAN(torch.nn.Module):

    def __init__(self, args):
        super(PPI_BAN, self).__init__()
        torch.backends.cudnn.enabled = False
        self.ban_heads = args['bcn_heads']
        self.batch_size = args['batch_size']
        self.type = args['task_type']
        self.embedding_size = args['emb_dim']
        self.drop = args['dropout']
        self.output_dim = args['output_dim']

        gcn = GeometryAwareRelationalGraphNeuralNetwork(input_dim=1280, hidden_dims=[1280],  ########################## n_layers
                      num_relation=7,
                      batch_norm=False, concat_hidden=False, short_cut=True, readout="max")    #edge_input_dim=59,num_angle_bin=8,

       # model_dict = torch.load('/home/disk1/hanyong/MODEL/MODEL_TAGPPI-review/mc_gearnet_edge.pth',
         #                   map_location=torch.device('cpu'))
        #gcn.load_state_dict(model_dict)
        # gcn = GCN(input_dim=1280, hidden_dims=1280, edge_input_dim=None, short_cut=False, batch_norm=False, activation='relu', concat_hidden=False, readout='max')
        self.gcn = gcn
        self.relu = nn.ReLU()
        self.fc_g = torch.nn.Linear(1280, self.output_dim)
        self.dropout = nn.Dropout(self.drop)
        # textcnn
        self.textcnn = ConvsLayer(self.embedding_size)

        # bilinear attention

        self.bcn = weight_norm(
            BANLayer(v_dim=512, q_dim=1280, h_dim=self.output_dim, h_out=self.ban_heads),  #############128*n
            name='h_mat', dim=None)

        # combined layers
        # self.bn = nn.BatchNorm1d(self.output_dim)
        self.fc1 = nn.Linear(512 * 6, 512 * 2)  # self.output_dim * 2 +
        self.bn1 = nn.BatchNorm1d(512 * 2)
        self.fc2 = nn.Linear(512 * 2, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)

    # input1 input2
    def forward(self, G1, pad_dmap1, G2, pad_dmap2):
        # protein1
        # protein2
        p1_global = []
        p2_global = []
        p1_local = []
        p2_local = []

        for i in range(self.batch_size):
            p1_x = pad_dmap1[i][:int(G1[i].num_residue)].float()
            p1 = self.gcn(G1[i], p1_x)
            p2_x = pad_dmap2[i][:int(G2[i].num_residue)].float()
            p2 = self.gcn(G2[i], p2_x)

            #p1 = self.gcn(G1[i], G1[i].residue_feature.float())
            #p2 = self.gcn(G2[i], G2[i].residue_feature.float())
            p1_global.append(p1['graph_feature'])
            p2_global.append(p2['graph_feature'])

            # protein1_local
            node_number1 = p1['node_feature'].shape[0]
            if node_number1 == 1200:
                p1_local.append(p1['node_feature'])
            else:
                padding_number = 1200 - node_number1
                pad = nn.ZeroPad2d(padding=(0, 0, 0, padding_number))
                p1_local.append(pad(p1['node_feature']))

            # protein2_local
            node_number2 = p2['node_feature'].shape[0]
            if node_number2 == 1200:
                p2_local.append(p2['node_feature'])
            else:
                padding_number = 1200 - node_number2
                pad = nn.ZeroPad2d(padding=(0, 0, 0, padding_number))
                p2_local.append(pad(p2['node_feature']))

        p1_local = torch.stack(p1_local, dim=0)
        p2_local = torch.stack(p2_local, dim=0)
        p1_global = torch.stack(p1_global, dim=0)
        p1_global = self.relu(self.fc_g(torch.squeeze(p1_global, dim=1)))
        p2_global = torch.stack(p2_global, dim=0)
        p2_global = self.relu(self.fc_g(torch.squeeze(p2_global, dim=1)))

        # p1_global_att = self.fc_transform(p1_global_att)
        # p2_global_att = self.fc_transform(p2_global_att)

        # sequence feature
        pad_dmap1 = torch.stack(pad_dmap1, dim=0)
        pad_dmap2 = torch.stack(pad_dmap2, dim=0)

        seq1_local, seq1_global = self.textcnn(pad_dmap1)
        seq2_local, seq2_global = self.textcnn(pad_dmap2)

        # p1_global_att = torch.concat((seq1, p1_global_att), 1)
        # p1_global_att = torch.unsqueeze(seq1, dim=1)
        # p2_global_att = torch.concat((seq2, p2_global_att), 1)
        # p2_global_att = torch.unsqueeze(seq2, dim=1)

        # bilinear
        f1, att1 = self.bcn(seq1_local, p1_local)
        f2, att2 = self.bcn(seq2_local, p2_local)

        # add some dense layers
        f_all = torch.concat((seq1_global, f1,p1_global,p2_global,f2, seq2_global), 1)   #seq1_global, f1, f2, seq2_global
        # att_all = torch.concat((att1, att2), 1)

        gc = self.fc1(f_all)
        #gc = self.bn1(gc)
        gc = self.relu(gc)
        gc = self.dropout(gc)
        gc = self.fc2(gc)
        #gc = self.bn2(gc)
        gc = self.relu(gc)
        gc = self.dropout(gc)
        out = self.out(gc)
        output = F.sigmoid(out)

        return output