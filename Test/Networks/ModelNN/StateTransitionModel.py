import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from Test.Datasets.TransitionDataGrid import data_store
import Test.utils as utils
import Test.config as config
class StateTransitionModel(nn.Module):
    def __init__(self, state_shape, num_actions, layers_type, layers_features, action_layer_num):
        super(StateTransitionModel, self).__init__()
        # state : W, H, Channels
        # action: A
        self.layers_type = layers_type
        self.layers = []

        linear_input_size = 0
        self.action_layer_num = action_layer_num
        self.num_actions = num_actions
        state_size = state_shape[0] * state_shape[1] * state_shape[2]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = num_actions

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
            self.head = nn.Linear(layers_features[-1] + num_actions, state_size)

        elif self.action_layer_num == len(self.layers_type) + 1:
            self.head = nn.Linear(layers_features[-1] , self.num_actions * state_size)
        else:
            self.head = nn.Linear(layers_features[-1], state_size)

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
                x = F.tanh(x)
            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)

        x = self.head(x.float())
        x = F.tanh(x)

        if self.action_layer_num == len(self.layers_type) + 1:
            return x.view((-1,) + (self.num_actions,) + state.shape[1:])  # -1 is for the batch size
        else:
            return x.view(state.shape) # -1 is for the batch size


def preTrainForward(env):

    def getActionIndex(action):
        for i, a in enumerate(env.getAllActions()):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(action):
        res = np.zeros([len(env.getAllActions())])
        res[getActionIndex(action)] = 1
        return res

    def testAccuracy(state_transition_model, test):
        sum = 0.0
        for data in test:
            state, action, next_state, reward = data
            state = torch.from_numpy(state).unsqueeze(0).float()
            next_state = torch.from_numpy(next_state).unsqueeze(0).float()
            action_index = getActionIndex(action)
            action_onehot = torch.from_numpy(getActionOnehot(action)).float().unsqueeze(0)
            if action_layer_num == len(model_layers_type) + 1:
                pred_state = state_transition_model(state, action_onehot).detach()[:, action_index]
            else:
                pred_state = state_transition_model(state, action_onehot).detach()

            err = (np.square(next_state - pred_state)).mean()
            sum += err
        mse = sum / len(test)
        return mse

    def createVisitCountsDict(env):
        visit_count = {}
        for state in env.getAllStates():
            for action in env.getAllActions():
                pos = env.stateToPos(state)
                visit_count[(pos, tuple(action))] = 0
        return visit_count

    train, test = data_store(env)
    model_layers_type = ['fc','fc']
    model_layers_features = [256,64]
    action_layer_num = 3
    model_step_size = 1.0
    batch_size = 1

    state_transition_model = StateTransitionModel(train[0].state.shape,
                                                   len(env.getAllActions()),
                                                   model_layers_type,
                                                   model_layers_features,
                                                   action_layer_num)
    num_samples = 0
    max_samples = 10000
    plot_y = []
    plot_x = []
    visit_count = createVisitCountsDict(env)
    print("Forward model is being trained")
    while num_samples < max_samples:
        transition_batch = random.choices(train, k=batch_size)
        num_samples += batch_size
        for transition in transition_batch:
            state, action, next_state, reward = transition

            pos = env.stateToPos(state)
            visit_count[(pos, tuple(action))] += 1

            x_old = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
            x_new = torch.from_numpy(np.asarray(next_state)).float().unsqueeze(0)
            action_index = getActionIndex(action)
            action_onehot = torch.from_numpy(getActionOnehot(action)).float().unsqueeze(0)

            if action_layer_num == len(model_layers_type) + 1:
                input = state_transition_model(x_old, action_onehot)[:, action_index]
            else:
                input = state_transition_model(x_old, action_onehot)

            target = x_new
            assert target.shape == input.shape, "target and input should have same shapes"

            loss = nn.MSELoss()(input, target)
            loss.backward()
            optimizer = optim.SGD(state_transition_model.parameters(), lr=(1 / batch_size) * model_step_size)
            optimizer.step()
            optimizer.zero_grad()

        mse = testAccuracy(state_transition_model, test)
        plot_y.append(mse)
        plot_x.append(num_samples)

        print(mse)
    print(visit_count)

    print("model training finished")
    return state_transition_model, visit_count, plot_y, plot_x


def preTrainBackward(env):
    train, test = data_store(env)
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
    print("Backward model is being trained")
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
    print("model training finished")
    return state_transition_model