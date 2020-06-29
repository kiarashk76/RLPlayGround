from BaseAgent import BaseAgent
from ..Networks.ValueFunction.StateActionValueFunction import StateActionVFNN3
from ..Networks.ValueFunction.StateValueFunction import StateVFNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
from .. import config, utils
import random

class BaseDynaAgent(BaseAgent):
    def __init__(self, params = {}):
        self.time_step = 0

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.transition_buffer = []
        self.transition_buffer_size = 30

        self.policy_values = 'q' # 'q' or 's' or 'qs'
        self.vf = {'q': dict(network=None,
                    layers_type=['fc', 'fc'],
                    layers_features=[64, 32],
                    action_layer_num=3, # if one more than layer numbers => we will have num of actions output
                    batch_size=1,
                    step_size=0.01,
                    training=True),
                   's': dict(network=None,
                    layers_type=['fc', 'fc'],
                    layers_features=[64, 32],
                    batch_size=5,
                    step_size=0.01,
                    training=False)}

        self.reward_function = params['reward_function']
        self.goal = params['goal']
        self.device = params['device']

    def start(self, observation):
        self.batch_counter = 0
        self.prev_state = self.agentState(observation)

        if self.vf['q']['network'] is None and self.vf['q']['training']:
            self.init_q_value_function_network() # a general state action VF for all actions
        if self.vf['s']['network'] is None and self.vf['s']['training']:
            self.init_s_value_function_network() # a separate state VF for each action
        self.initModel()

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.prev_action = self.policy(x_old)


        return self.prev_action

    def step(self, reward, observation):
        self.time_step += 1
        self.state = self.agentState(observation)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        x_new = torch.from_numpy(self.state).unsqueeze(0)

        self.action = self.policy(x_new)
        self.updateValueFunction(reward, x_old, x_new)

        self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
                                                     self.state, self.time_step))
        self.trainModel()
        self.plan()

        self.prev_state = self.state
        self.prev_action = self.action # another option:** we can again call self.policy function **



        return self.prev_action

    def end(self, reward):

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)

        self.updateValueFunction(reward, x_old, None)

        # self.trainModel(self.prev_state, self.action, self.state, terminal=True)


    def policy(self, state_torch):
        if np.random.rand() > self.epsilon:
            v = []
            for i, action in enumerate(self.action_list):
                if self.policy_values == 'q':
                    action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0)
                    if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                        v.append(self.vf['q']['network'](state_torch, action_onehot).detach()[0, i])
                    else:
                        v.append(self.vf['q']['network'](state_torch, action_onehot).detach())
                elif self.policy_values == 's':
                    v.append(self.vf['s']['network'][i](state_torch).detach())
                elif self.policy_values == 'qs':
                    if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                        q = self.vf['q']['network'](state_torch, action_onehot).detach()[0, i]
                    else:
                        q = self.vf['q']['network'](state_torch, action_onehot).detach()
                    s = self.vf['s']['network'][i](state_torch).detach()
                    v.append( (q+s) /2)
                else:
                   raise ValueError('policy is not defined')
            action = self.action_list[np.argmax(v)]
        else:
            action = self.action_list[int(np.random.rand() * self.num_actions)]

        return action

    def agentState(self, observation):
        return np.copy(observation)

    def updateNetworkWeights(self, network, step_size):
        # another option: ** can use a optimizer here later**
        optimizer = optim.SGD(network.parameters(), lr= step_size)
        optimizer.step()
        optimizer.zero_grad()

        # for f in network.parameters():
        #     f.data.sub_(step_size * f.grad.data)
        # network.zero_grad()

    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(self, action):
        res = np.zeros([len(self.action_list)])
        res[self.getActionIndex(action)] = 1
        return res

    def init_q_value_function_network(self):
        nn_state_shape = (self.vf['q']['batch_size'],) + self.prev_state.shape
        self.vf['q']['network'] = StateActionVFNN3(nn_state_shape, self.num_actions,
                                                 self.vf['q']['layers_type'],
                                                 self.vf['q']['layers_features'],
                                                 self.vf['q']['action_layer_num']).to(self.device)

    def init_s_value_function_network(self):
        nn_state_shape = (self.vf['s']['batch_size'],) + self.prev_state.shape
        self.vf['s']['network'] = []
        for i in range(self.num_actions):
            self.vf['s']['network'].append(StateVFNN(nn_state_shape,
                                                     self.vf['s']['layers_type'],
                                                     self.vf['s']['layers_features']).to(self.device))

    def updateValueFunction(self, reward, x_old, x_new = None, prev_action=None, action=None):
        self.batch_counter += 1
        if prev_action is None:
            prev_action = self.prev_action
        if x_new is not None: # Not a terminal State
            if action is None:
                action = self.action
            assert x_old.shape == x_new.shape, 'x_old and x_new have different shapes'
            prev_action_index = self.getActionIndex(prev_action)
            action_index = self.getActionIndex(action)
            prev_action_onehot = torch.from_numpy(self.getActionOnehot(prev_action)).float().unsqueeze(0)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0)
            if self.vf['q']['training']:
                if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                    target = reward + self.gamma * self.vf['q']['network'](x_new).detach()[:, action_index]
                    input = self.vf['q']['network'](x_old)[:, prev_action_index]
                else:
                    target = reward + self.gamma * self.vf['q']['network'](x_new, action_onehot).detach()
                    input = self.vf['q']['network'](x_old, prev_action_onehot)
                assert target.shape == input.shape, 'target and input must have same shapes'
                loss = nn.MSELoss()(input, target)
                loss.backward()
                if self.batch_counter % self.vf['q']['batch_size'] == 0:
                    step_size = self.vf['q']['step_size'] / self.vf['q']['batch_size']
                    self.updateNetworkWeights(self.vf['q']['network'],
                                              step_size)

            if self.vf['s']['training']:
                target = reward + self.gamma * self.vf['s']['network'][action_index](x_new).detach()
                input = self.vf['s']['network'][prev_action_index](x_old)
                assert target.shape == input.shape, 'target and input must have same shapes'
                loss = nn.MSELoss()(input, target)
                loss.backward()
                if self.batch_counter % self.vf['s']['batch_size'] == 0:
                    step_size = self.vf['s']['step_size'] / self.vf['s']['batch_size']
                    self.updateNetworkWeights(self.vf['s']['network'][prev_action_index],
                                              step_size)
        else : # terminal state
            prev_action_index = self.getActionIndex(prev_action)
            prev_action_onehot = torch.from_numpy(self.getActionOnehot(prev_action)).float().unsqueeze(0)
            if self.vf['q']['training']:
                if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                    target = torch.tensor(reward).unsqueeze(0)
                    input = self.vf['q']['network'](x_old)[:, prev_action_index]
                else:
                    target = torch.tensor(reward).unsqueeze(0).unsqueeze(0)
                    input = self.vf['q']['network'](x_old, prev_action_onehot)

                assert target.shape == input.shape, 'target and input must have same shapes'
                loss = nn.MSELoss()(input.float(), target.float())
                loss.backward()

                if self.batch_counter % self.vf['q']['batch_size'] == 0 :
                    step_size = self.vf['q']['step_size'] / self.vf['q']['batch_size']
                else:
                    step_size = self.vf['q']['step_size'] /\
                                (self.batch_counter % self.vf['q']['batch_size'])

                self.updateNetworkWeights(self.vf['q']['network'], step_size)
            if self.vf['s']['training']:
                target = torch.tensor(reward).unsqueeze(0).unsqueeze(0)
                input = self.vf['s']['network'][prev_action_index](x_old)
                assert target.shape == input.shape, 'target and input must have same shapes'
                loss = nn.MSELoss()(input.float(), target.float())
                loss.backward()

                if self.batch_counter % self.vf['s']['batch_size'] == 0 :
                    step_size = self.vf['s']['step_size'] / self.vf['s']['batch_size']
                else:
                    step_size = self.vf['s']['step_size'] /\
                                (self.batch_counter % self.vf['s']['batch_size'])
                self.updateNetworkWeights(self.vf['s']['network'][prev_action_index], step_size)

    def getStateActionValue(self, x, action, type='q'):
        action_index = self.getActionIndex(action)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0)

        if type == 'q':
            if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                value = self.vf['q']['network'](x).detach()[:, action_index]
            else:
                value = self.vf['q']['network'](x, action_onehot).detach()

        elif type == 's':
            value = self.vf['s']['network'][action_index](x).detach()

        else:
            raise ValueError('state action value type is not defined')

        return value

    def getTransitionFromBuffer(self, n=1):
        pass

    @abstractmethod
    def trainModel(self):
        pass

    @abstractmethod
    def plan(self):
        pass

    @abstractmethod
    def initModel(self):
        pass

    def updateTransitionBuffer(self, transition):
        self.transition_buffer.append(transition)
        if len(self.transition_buffer) > self.transition_buffer_size:
            self.removeFromTransitionBuffer()

    @abstractmethod
    def removeFromTransitionBuffer(self):
        self.transition_buffer.pop(0)

