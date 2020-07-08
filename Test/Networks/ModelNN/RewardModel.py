import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, state, action):
        super(RewardModel, self).__init__()
        self.state_shape = state.shape
        linear_states_shape = 1
        action_shape = 1
        for shape in state.shape:
            linear_states_shape *= shape
        for shape in action.shape:
            action_shape *= shape
        input_size = linear_states_shape + action_shape

        nh1 = 50
        nh2 = 50
        self.fc1 = nn.Linear(input_size, nh1)
        self.fc2 = nn.Linear(nh1, nh2)
        self.head = nn.Linear(nh2, 1)

    def forward(self, prev_state, prev_action):
        prev_state_fl = prev_state.flatten(start_dim = 1)
        x = torch.cat((prev_state_fl, prev_action), dim = 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.head(x)
        x = F.logsigmoid(x)

        return x.view(-1,)
