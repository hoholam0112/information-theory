""" Define neural networks for two gaussian experiment """

import torch
import torch.nn as nn

class StatisticsNetwork(nn.Module):
    def __init__(self,
                 dim_x,
                 dim_y,
                 hidden_units,
                 model_name,
                 activation='relu'):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_units = hidden_units
        self.model_name = model_name
        self.activation = activation

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError('Invalid name for activation: {}'.format(activation))

        if model_name == 'basic':
            self.layer_x = nn.Identity()
            self.layer_y = nn.Identity()
            self.layer = nn.Sequential(
                    nn.Linear(dim_x + dim_y, hidden_units),
                    self.activation,
                    nn.Linear(hidden_units, hidden_units),
                    self.activation,
                    nn.Linear(hidden_units, 1)
                )
        elif model_name == 'seperate_layer':
            self.layer_x = nn.Sequential(
                    nn.Linear(dim_x, hidden_units),
                    self.activation
                )
            self.layer_y = nn.Sequential(
                    nn.Linear(dim_y, hidden_units),
                    self.activation
                )
            self.layer = nn.Sequential(
                    nn.Linear(2*hidden_units, 2*hidden_units),
                    self.activation,
                    nn.Linear(2*hidden_units, hidden_units),
                    self.activation,
                    nn.Linear(hidden_units, 1)
                )
        elif model_name == 'seperate_layer+batch_norm':
            self.layer_x = nn.Sequential(
                    nn.Linear(dim_x, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    self.activation
                )
            self.layer_y = nn.Sequential(
                    nn.Linear(dim_y, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    self.activation
                )
            self.layer = nn.Sequential(
                    nn.Linear(2*hidden_units, 2*hidden_units),
                    nn.BatchNorm1d(2*hidden_units),
                    self.activation,
                    nn.Linear(2*hidden_units, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    self.activation,
                    nn.Linear(hidden_units, 1)
                )
        elif model_name == 'basic+batch_norm':
            self.layer_x = nn.Identity()
            self.layer_y = nn.Identity()
            self.layer = nn.Sequential(
                    nn.Linear(dim_x + dim_y, hidden_units),
                    nn.BatchNorm1d(hidden_units, affine=False),
                    self.activation,
                    nn.Linear(hidden_units, hidden_units),
                    nn.BatchNorm1d(hidden_units, affine=False),
                    self.activation,
                    nn.Linear(hidden_units, 1)
                )
        else:
            raise ValueError('Invalid name for model_name: {}'.format(model_name))

    def forward(self, x, y):
        x = self.layer_x(x)
        y = self.layer_y(y)
        h = torch.cat([x, y], axis=1)
        output = self.layer(h)
        return output

