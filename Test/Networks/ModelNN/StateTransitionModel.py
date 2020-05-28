import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTransitionModel(nn.Module):
    def __init__(self, state, action):
        super(StateTransitionModel, self).__init__()
        self.state_shape = state.shape
        linear_states_shape = 1
        action_shape = 1
        for shape in state.shape:
            print(shape)
            linear_states_shape *= shape
        for shape in action.shape:
            action_shape *= shape

        nh1 = 50
        nh2 = 50
        self.fc1 = nn.Linear(linear_states_shape + action_shape, nh1)
        self.fc2 = nn.Linear(nh1, nh2)
        self.head = nn.Linear(nh2, linear_states_shape)

    def forward(self, prev_state, prev_action):
        x = prev_state.flatten(start_dim= 1)
        x = torch.cat((x, prev_action),dim= 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.head(x)
        return x.view((-1,) + self.state_shape) # -1 is for the batch size
