import torch
import torch.nn as nn


class StateTransitionModelwError(nn.Module):
    def __init__(self, state_shape, num_actions, layers_type, layers_features, action_layer_num):
        super(StateTransitionModelwError, self).__init__()
        # state : W, H, Channels
        # action: A
        self.layers_type = layers_type
        self.layers = []

        self.action_layer_num = action_layer_num
        self.num_actions = num_actions

        # self.state_size = state_shape[0] * state_shape[1] * state_shape[2]
        self.state_size = state_shape[0]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = num_actions

                if i == 0:
                    linear_input_size = self.state_size + action_shape_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    self.add_module('hidden_layer_' + str(i), layer)
                    self.layers.append(layer)
                else:
                    layer = nn.Linear(layers_features[i - 1] + action_shape_size, layers_features[i])
                    self.add_module('hidden_layer_' + str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            self.head = nn.Linear(layers_features[-1] + num_actions, self.state_size + 1)

        elif self.action_layer_num == len(self.layers_type) + 1:
            self.head = nn.Linear(layers_features[-1], self.num_actions * self.state_size + 1)
        else:
            self.head = nn.Linear(layers_features[-1], self.state_size + 1)

    def forward(self, state, action = None):
        if self.action_layer_num == len(self.layers) + 1 and action is None:
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
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)

        x = self.head(x.float())
        # x = torch.relu(x)
        pred_state, acc = x[0].split(self.state_size)

        if self.action_layer_num == len(self.layers_type) + 1:
            print('dd')
            pred_state = pred_state.view((-1,) + (self.num_actions,) + state.shape[1:])
        else:
            pred_state = pred_state.view(state.shape)
        return pred_state, acc
