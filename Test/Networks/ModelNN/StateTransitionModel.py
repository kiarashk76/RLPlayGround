import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Test.Datasets.TransitionDataGrid import data_store

class StateTransitionModel(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features, action_layer_num):
        super(StateTransitionModel, self).__init__()

        # state : Batch, W, H, Channels
        # action: Batch, A
        self.layers_type = layers_type
        self.layers = []
        linear_input_size = 0
        self.action_layer_num = action_layer_num
        state_size = state_shape[1] * state_shape[2] * state_shape[3]
        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = action_shape[1]

                if i == 0:
                    linear_input_size = state_shape[1] * state_shape[2] * state_shape[3] + action_shape_size
                    self.layers.append(nn.Linear(linear_input_size, layers_features[i]))
                else:
                    self.layers.append(nn.Linear(layers_features[i - 1] + action_shape_size, layers_features[i]))


            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            self.head = nn.Linear(layers_features[-1] + action_shape[1], state_size)
        else:
            self.head = nn.Linear(layers_features[-1] , state_size)


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
                x = F.tanh(x)
            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)

        x = self.head(x.float())
        # x = F.tanh(x)
        # x = F.gelu(x)
        x = F.tanh(x)

        return x.view(state.shape) # -1 is for the batch size


def preTrainForwad():
    train, test = data_store()
    model_layers_type = ['fc', 'fc']
    model_layers_features = [32, 32]
    model_step_size = 0.1
    batch_size = 5
    num_epochs = 1000

    state_transition_model = (StateTransitionModel((batch_size, ) + train[0].state.shape,
                                                   (batch_size, ) + train[0].action.shape,
                                                   model_layers_type,
                                                   model_layers_features,
                                                   action_layer_num= 2))

    for i in range(num_epochs):
        batch_count = 0
        state_list = []
        next_state_list = []
        action_list = []

        for data in train:
            state, action, next_state, reward = data
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            batch_count += 1

            if batch_count == batch_size:
                x_old = torch.from_numpy(np.asarray(state_list)).float()
                x_new = torch.from_numpy(np.asarray(next_state_list)).float()
                action = torch.from_numpy(np.asarray(action_list)).float()

                input = state_transition_model(x_old, action)
                target = x_new
                loss = nn.MSELoss()(input, target)
                loss.backward()

                for f in state_transition_model.parameters():
                    f.data.sub_(model_step_size * f.grad.data)
                state_transition_model.zero_grad()
                batch_count = 0
                state_list = []
                action_list = []
                next_state_list = []

        if batch_count > 0:
            x_old = torch.from_numpy(np.asarray(state_list)).float()
            x_new = torch.from_numpy(np.asarray(next_state_list)).float()
            action = torch.from_numpy(np.asarray(action_list)).float()

            input = state_transition_model(x_old, action)
            target = x_new

            optimizer = optim.SGD(state_transition_model.parameters(), lr = model_step_size)
            optimizer.zero_grad()
            loss = nn.MSELoss()(input, target)
            loss.backward()
            optimizer.step()

            # for f in state_transition_model.parameters():
            #     f.data.sub_(model_step_size * f.grad.data)
            # state_transition_model.zero_grad()

            batch_count = 0
            state_list = []
            action_list = []
            next_state_list = []

        sum = 0.0
        for data in test:
            state, action, next_state, reward = data
            state = torch.from_numpy(state).unsqueeze(0).float()
            next_state = torch.from_numpy(next_state).unsqueeze(0).float()
            action = torch.from_numpy(action).unsqueeze(0).float()
            pred_state = state_transition_model(state, action).detach()
            err = (np.square(next_state - pred_state)).mean()

            sum += err
        mse = sum / len(test)
        # print(mse)

    return state_transition_model


def preTrainBackward():
    train, test = data_store()
    model_layers_type = ['fc', 'fc']
    model_layers_features = [32, 32]
    model_step_size = 0.1
    batch_size = 5
    num_epochs = 1000

    state_transition_model = (StateTransitionModel((batch_size, ) + train[0].state.shape,
                                                   (batch_size, ) + train[0].action.shape,
                                                   model_layers_type,
                                                   model_layers_features,
                                                   action_layer_num= 2))

    for i in range(num_epochs):
        batch_count = 0
        state_list = []
        next_state_list = []
        action_list = []

        for data in train:
            state, action, next_state, reward = data
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            batch_count += 1

            if batch_count == batch_size:
                x_old = torch.from_numpy(np.asarray(state_list)).float()
                x_new = torch.from_numpy(np.asarray(next_state_list)).float()
                action = torch.from_numpy(np.asarray(action_list)).float()

                input = state_transition_model(x_new, action)
                target = x_old
                loss = nn.MSELoss()(input, target)
                loss.backward()

                for f in state_transition_model.parameters():
                    f.data.sub_(model_step_size * f.grad.data)
                state_transition_model.zero_grad()
                batch_count = 0
                state_list = []
                action_list = []
                next_state_list = []

        if batch_count > 0:
            x_old = torch.from_numpy(np.asarray(state_list)).float()
            x_new = torch.from_numpy(np.asarray(next_state_list)).float()
            action = torch.from_numpy(np.asarray(action_list)).float()

            input = state_transition_model(x_new, action)
            target = x_old

            optimizer = optim.SGD(state_transition_model.parameters(), lr = model_step_size)
            optimizer.zero_grad()
            loss = nn.MSELoss()(input, target)
            loss.backward()
            optimizer.step()

            # for f in state_transition_model.parameters():
            #     f.data.sub_(model_step_size * f.grad.data)
            # state_transition_model.zero_grad()


        sum = 0.0
        for data in test:
            state, action, next_state, reward = data
            state = torch.from_numpy(state).unsqueeze(0).float()
            next_state = torch.from_numpy(next_state).unsqueeze(0).float()
            action = torch.from_numpy(action).unsqueeze(0).float()
            pred_state = state_transition_model(next_state, action).detach()
            err = (np.square(state - pred_state)).mean()

            sum += err
        mse = sum / len(test)
        # print(mse)
    return state_transition_model

if __name__ =='__main__':
    forward_model = preTrainForwad()
    backward_model = preTrainBackward()