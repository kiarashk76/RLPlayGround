import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateActionVFNN1(nn.Module): # action inserted to a layer
    def __init__(self, state_shape, action_shape, layers_type, layers_features, action_layer_num):
        # state : Batch, W, H, Channels
        # action: Batch, A
        super(StateActionVFNN1, self).__init__()
        self.layers_type = layers_type
        self.layers = []

        self.action_layer_num = action_layer_num
        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer =='fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = action_shape[1]
                if i == 0:
                    linear_input_size = state_shape[1] * state_shape[2] * state_shape[3] + action_shape_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
                else:
                    layer = nn.Linear(layers_features[i-1], layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")


        if self.action_layer_num == len(self.layers_type):
            self.head = nn.Linear(layers_features[-1] + action_shape[1], 1)
        else:
            self.head = nn.Linear(layers_features[-1] , 1)

        if self.action_layer_num > len(self.layers_type) or self.action_layer_num < 0:
            raise ValueError('action layer is out of network')

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

        return self.head(x.float()) # -1 is for the batch size

class StateActionVFNN2(nn.Module): # one-hot action inserted to a layer
    def __init__(self, state_shape, num_actions, layers_type, layers_features):
        # state : Batch, W, H, Channels
        # action: Batch, A
        super(StateActionVFNN2, self).__init__()
        self.layers_type = layers_type
        self.layers = []

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer =='fc':

                if i == 0:
                    linear_input_size = state_shape[1] * state_shape[2] * state_shape[3]
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
                else:
                    layer = nn.Linear(layers_features[i-1], layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")



        self.head = nn.Linear(layers_features[-1] , num_actions)


    def forward(self, state, action):
        x = 0
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                x = self.layers[i](x.float())
                x = F.relu(x)
            else:
                raise ValueError("layer is not defined")

        return self.head(x.float()) # -1 is for the batch size

class StateActionVFNN3(nn.Module): # last layer has number of actions' output
    def __init__(self, state_shape, num_actions, layers_type, layers_features, action_layer_num):
        # state : Batch, W, H, Channels
        # action: Batch, A
        super(StateActionVFNN3, self).__init__()
        self.layers_type = layers_type
        self.action_layer_num = action_layer_num
        self.layers = []

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer =='fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = num_actions

                if i == 0:
                    linear_input_size = state_shape[1] * state_shape[2] * state_shape[3] + action_shape_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)

                else:
                    layer = nn.Linear(layers_features[i-1] + action_shape_size, layers_features[i])
                    self.add_module('hidden_layer_'+str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            self.head = nn.Linear(layers_features[-1] + num_actions, 1)

        elif self.action_layer_num == len(self.layers_type) + 1:
            self.head = nn.Linear(layers_features[-1], num_actions)

        else:
            self.head = nn.Linear(layers_features[-1], 1)



    def forward(self, state, action=None):
        if self.action_layer_num != len(self.layers) + 1 and action is None:
            raise ValueError("action is not given")
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
        return x

        # if self.action_layer_num == len(self.layers_type) + 1:
        #     return x.view((-1,) + (self.num_actions,) + state.shape[1:])  # -1 is for the batch size
        # else:
        #     return x.view(state.shape)  # -1 is for the batch size
