import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateVFNN(nn.Module):
    def __init__(self, state_shape, layers_type, layers_features):
        # state : Batch, W, H, Channels
        # action: Batch, A
        super(StateVFNN, self).__init__()
        self.layers_type = layers_type
        self.layers = []
        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer =='fc':
                if i == 0:
                    linear_input_size = state_shape[1] * state_shape[2] * state_shape[3]
                    self.layers.append(nn.Linear(linear_input_size, layers_features[i]))
                else:
                    self.layers.append(nn.Linear(layers_features[i-1], layers_features[i]))
            else:
                raise ValueError("layer is not defined")

        self.head = nn.Linear(layers_features[-1], 1)

    def forward(self, state):
        x = 0

        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    s = state.flatten(start_dim= 1)
                    x = s
                    x = self.layers[i](x.float())
                else:
                    x = self.layers[i](x.float())

            else:
                raise ValueError("layer is not defined")
        return self.head(x.float())

