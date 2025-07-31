from typing import List
import hickle as hkl
import os, random
import torch
import torch.nn as nn
from typing import Dict
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data as GraphData
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    BatchNorm as GraphBatchNorm,
    Sequential as GraphSequential
    )


class DeepCDR(nn.Module):
    def __init__(self, drug_input_dim, mutation_dim, gexpr_dim, methy_dim,
                 units_list=[256,256,256], use_mut=False, use_gexp=True, use_methy=False,
                 use_relu=True, use_bn=True, use_GMP=True, regression=True):
        super(DeepCDR, self).__init__()
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.use_GMP = use_GMP
        self.regression = regression

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()

        in_dim = drug_input_dim
        for out_dim in units_list:
            self.gcn_layers.append(GCNConv(in_dim, out_dim))
            if self.use_bn:
                self.gcn_bns.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim
        self.gcn_final = GCNConv(in_dim, 100)

        # Mutation (Conv2D replacement)
        self.mut_conv1 = nn.Conv2d(1, 50, kernel_size=(1, 700), stride=(1, 5))
        self.mut_pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.mut_conv2 = nn.Conv2d(50, 30, kernel_size=(1, 5), stride=(1, 2))
        self.mut_pool2 = nn.MaxPool2d(kernel_size=(1, 10))
        self.mut_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * 1 * 2, 100),  # Adjust based on input shape
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Gene expression branch
        self.gexpr_net = nn.Sequential(
            nn.Linear(gexpr_dim, 256),
            nn.Tanh(),
            nn.BatchNorm1d(256) if use_bn else nn.Identity(),
            nn.Dropout(0.1),
            nn.Linear(256, 100),
            nn.ReLU()
        )

        # Methylation branch
        self.methy_net = nn.Sequential(
            nn.Linear(methy_dim, 256),
            nn.Tanh(),
            nn.BatchNorm1d(256) if use_bn else nn.Identity(),
            nn.Dropout(0.1),
            nn.Linear(256, 100),
            nn.ReLU()
        )

        # Fusion and CNN layers
        self.fusion_fc = nn.Sequential(
            nn.Linear(100 + (100 if use_mut else 0) + (100 if use_gexp else 0) + (100 if use_methy else 0), 300),
            nn.Tanh(),
            nn.Dropout(0.1)
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=150, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=30, out_channels=10, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=10, out_channels=5, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.cnn_1 = nn.Conv2d(1, 30, kernel_size=(1, 150))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.cnn_2 = nn.Conv2d(30, 10, kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.cnn_3 = nn.Conv2d(10, 5, kernel_size=(1, 5))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3))
        self.gcn_bn1 = nn.BatchNorm1d(256)
        self.gcn_bn2 = nn.BatchNorm1d(100)
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(30, 1 if regression else 1),
        )

    def forward(self, data, gexpr, methy, mutation):
        # GCN processing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, edge_index)
            x = F.relu(x) if self.use_relu else torch.tanh(x)
            if self.use_bn:
                x = self.gcn_bns[i](x)
            x = F.dropout(x, p=0.1, training=self.training)
        x = self.gcn_final(x, edge_index)
        x = F.relu(x) if self.use_relu else F.tanh(x)
        if self.use_bn:
            x = self.gcn_bn2(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x_drug = global_max_pool(x, batch) if self.use_GMP else global_mean_pool(x, batch)

        concat_features = [x_drug]

        if self.use_mut:
            x_mut = self.mut_conv1(mutation)
            x_mut = self.mut_pool1(x_mut)
            x_mut = self.mut_conv2(x_mut)
            x_mut = self.mut_pool2(x_mut)
            x_mut = self.mut_fc(x_mut)
            concat_features.append(x_mut)

        if self.use_gexp:
            concat_features.append(self.gexpr_net(gexpr))
        if self.use_methy:
            concat_features.append(self.methy_net(methy))

        x = torch.cat(concat_features, dim=1)
        x = self.fusion_fc(x)
        # x = self.projection(x)
        x = x.unsqueeze(1)
        x = self.conv(x)

        #
        # # CNN fusion
        # x = x.unsqueeze(1).unsqueeze(-1)  # [B, 1, 300, 1]
        # x = self.cnn_1(x)
        # x = self.pool1(x)
        # x = self.cnn_2(x)
        # x = self.pool2(x)
        # x = self.cnn_3(x)
        # x = self.pool3(x)

        x = self.final(x)
        return x.squeeze() if self.regression else torch.sigmoid(x).squeeze()

