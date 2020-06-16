import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateActionVFNN(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features, action_layer_num):
        # state : Batch, W, H, Channels
        # action: Batch, A
        super(StateActionVFNN, self).__init__()
        self.layers_type = layers_type
        self.layers = []

        linear_input_size = 0
        self.action_layer_num = action_layer_num
        state_size = state_shape[1] * state_shape[2] * state_shape[3]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer =='fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = action_shape[1]

                if i == 0:
                    linear_input_size = state_size + action_shape_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
                else:
                    layer = nn.Linear(layers_features[i - 1] + action_shape_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")


        if self.action_layer_num == len(self.layers_type):
            self.head = nn.Linear(layers_features[-1] + action_shape[1], 1)
        else:
            self.head = nn.Linear(layers_features[-1] , 1)

    def forward(self, state, action):
        x = 0
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)

                if i == self.action_layer_num:
                    # insert action to this layer
                    a = action.flatten(start_dim=1)
                    x = torch.cat((x.float(), a.float()), dim=1)

                x = self.layers[i](x.float())
                x = F.relu(x)


            else:
                raise ValueError("layer is not defined")


        if self.action_layer_num == len(self.layers_type):
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)

        x = self.head(x.float())
        # x = F.tanh(x)
        # x = F.gelu(x)
        x = F.relu(x)
        return x # -1 is for the batch size

# class StateActionVFNN(nn.Module):
#     def __init__(self, state_shape, action_shape, layers_type, layers_features, action_layer_num):
#         # state : Batch, W, H, Channels
#         # action: Batch, A
#         super(StateActionVFNN, self).__init__()
#         self.layers_type = layers_type
#         self.layers = []
#
#         linear_input_size = 0
#         self.action_layer_num = action_layer_num
#         state_size = state_shape[1] * state_shape[2] * state_shape[3]
#
#         action_shape_size = action_shape[1]
#         linear_input_size = state_size #+ action_shape_size
#         self.l1 = nn.Linear(linear_input_size, 8)
#         self.l2 = nn.Linear(8, 8)
#         self.l3 = nn.Linear(8, 8)
#         self.head = nn.Linear(8 + action_shape[1], 1)
#
#     def forward(self, state, action):
#
#         x = state.flatten(start_dim= 1)
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = torch.cat((x.float(), action.float()), dim=1)
#
#         x = self.head(x.float())
#         # x = F.tanh(x)
#         # x = F.gelu(x)
#         x = F.relu(x)
#         return x # -1 is for the batch size
