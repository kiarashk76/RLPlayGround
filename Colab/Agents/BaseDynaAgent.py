import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
import random

import Colab.config as config, Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent
from Colab.Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN3, StateActionVFNN4
from Colab.Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Colab.Networks.RepresentationNN.StateRepresentation import StateRepresentation

class BaseDynaAgent3(BaseAgent):
    def __init__(self, params = {}):
        self.time_step = 0

        self.prev_obs = None
        self.obs = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.transition_buffer = []
        self.transition_buffer_size = 100

        self.policy_values = 'q' # 'q' or 's' or 'qs'
        self.vf = {'q': dict(network=None,
                    layers_type=['fc', 'fc'],
                    layers_features=[64, 32],
                    action_layer_num=3, # if one more than layer numbers => we will have num of actions output
                    batch_size=20,
                    batch_counter=1,
                    step_size=0.01,
                    training=True),
                   's': dict(network=None,
                    layers_type=['fc', 'fc'],
                    layers_features=[64, 32],
                    batch_size=5,
                    step_size=0.01,
                    batch_counter=0,
                    training=False)}

        self.reward_function = params['reward_function']
        self.device = params['device']
        self.goal = torch.from_numpy(params['goal']).float().to(self.device)

    def start(self, observation):
        self.batch_counter = 0
        self.prev_state = self.getStateRepresentation(observation)

        if self.vf['q']['network'] is None and self.vf['q']['training']:
            self.init_q_value_function_network() # a general state action VF for all actions
        if self.vf['s']['network'] is None and self.vf['s']['training']:
            self.init_s_value_function_network() # a separate state VF for each action

        self.prev_action = self.policy(self.prev_state)

        self.initModel()

        return self.prev_action

    def step(self, reward, observation):
        self.time_step += 1
        self.state = self.getStateRepresentation(observation)

        self.action = self.policy(self.state)
        self.updateValueFunction(reward, self.prev_state, self.state)

        self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
                                                     self.state, self.action, False, self.time_step))
        self.trainModel()
        self.plan()

        self.prev_state = self.state
        self.prev_action = self.action # another option:** we can again call self.policy function **



        return self.prev_action

    def end(self, reward):

        self.updateValueFunction(reward, self.prev_state, None)

        # self.trainModel(self.prev_state, self.action, self.state, terminal=True)

    def policy(self, state_torch):
        state_torch = state_torch.to(self.device).unsqueeze(0)
        if np.random.rand() > self.epsilon:
            v = []
            for i, action in enumerate(self.action_list):
                if self.policy_values == 'q':
                    action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)
                    if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                        v.append(self.vf['q']['network'](state_torch, action_onehot).detach()[0, i])
                    else:
                        v.append(self.vf['q']['network'](state_torch, action_onehot).detach())
                elif self.policy_values == 's':
                    v.append(self.vf['s']['network'][i](state_torch).detach())
                elif self.policy_values == 'qs':
                    action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)
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

    def getStateRepresentation(self, observation):
        return torch.from_numpy(observation).to(self.device)

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
            prev_action_onehot = torch.from_numpy(self.getActionOnehot(prev_action)).float().unsqueeze(0).to(self.device)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

            x_old = x_old.unsqueeze(0).to(self.device)
            x_new = x_new.unsqueeze(0).to(self.device)
            reward = torch.tensor(reward).to(self.device).unsqueeze(0)

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
            prev_action_onehot = torch.from_numpy(self.getActionOnehot(prev_action)).float().unsqueeze(0).to(self.device)

            x_old = x_old.unsqueeze(0).to(self.device)
            reward = torch.tensor(reward).to(self.device).unsqueeze(0)

            if self.vf['q']['training']:
                if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                    target = reward
                    input = self.vf['q']['network'](x_old)[:, prev_action_index]
                else:
                    target = reward.unsqueeze(0)
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
                target = torch.tensor(reward).unsqueeze(0)
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

    def getStateActionValue(self, x, action=None, type='q'):
        x = x.unsqueeze(0)
        if action is not None:
            action_index = self.getActionIndex(action)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

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
        else:
            sum = 0
            for action in self.action_list:
                action_index = self.getActionIndex(action)
                action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

                if type == 'q':
                    if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
                        value = self.vf['q']['network'](x).detach()[:, action_index]
                    else:
                        value = self.vf['q']['network'](x, action_onehot).detach()

                elif type == 's':
                    value = self.vf['s']['network'][action_index](x).detach()

                else:
                    raise ValueError('state action value type is not defined')

                sum+= value

            return sum / len(self.action_list)

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

# class BaseDynaAgent(BaseAgent):
#     def __init__(self, params={}):
#         self.time_step = 0
#
#         self.prev_state = None
#         self.state = None
#
#         self.action_list = params['action_list']
#         self.num_actions = self.action_list.shape[0]
#         self.actions_shape = self.action_list.shape[1:]
#
#         self.gamma = params['gamma']
#         self.epsilon = params['epsilon']
#
#         self.transition_buffer = []
#         self.transition_buffer_size = 20
#
#         self.policy_values = 'q'  # 'q' or 's' or 'qs'
#         self.vf = {'q': dict(network=None,
#                              layers_type=['fc', 'fc'],
#                              layers_features=[64, 32],
#                              action_layer_num=3,  # if one more than layer numbers => we will have num of actions output
#                              batch_size=1,
#                              step_size=0.01,
#                              training=True),
#                    's': dict(network=None,
#                              layers_type=['fc', 'fc'],
#                              layers_features=[64, 32],
#                              batch_size=5,
#                              step_size=0.01,
#                              training=False)}
#
#         self.representation = dict(network=None, layers_type=['fc'], layers_features=[64], step_size=0.01)
#
#         self.reward_function = params['reward_function']
#         self.device = params['device']
#         self.goal = torch.from_numpy(params['goal']).float().to(self.device)
#
#     def start(self, observation):
#         self.batch_counter = 0
#         self.prev_state = self.agentState(observation)
#
#         if self.vf['q']['network'] is None and self.vf['q']['training']:
#             self.init_q_value_function_network()  # a general state action VF for all actions
#         if self.vf['s']['network'] is None and self.vf['s']['training']:
#             self.init_s_value_function_network()  # a separate state VF for each action
#         if self.representation['network'] is None:
#             self.init_s_representation_network()
#         self.prev_action = self.policy(self.prev_state)
#
#         self.initModel()
#
#         return self.prev_action
#
#     def step(self, reward, observation):
#         self.time_step += 1
#         self.state = self.agentState(observation)
#
#         self.action = self.policy(self.state)
#         self.updateValueFunction(reward, self.prev_state, self.prev_action, x_new=self.state, action=self.action)
#
#         self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
#                                                      self.state, self.time_step))
#         self.trainModel()
#         self.plan()
#
#         self.prev_state = self.state
#         self.prev_action = self.action  # another option:** we can again call self.policy function **
#
#         return self.prev_action
#
#     def end(self, reward):
#
#         self.updateValueFunction(reward, self.prev_state, self.prev_action)
#
#         # self.trainModel(self.prev_state, self.action, self.state, terminal=True)
#
#     def policy(self, state_torch):
#         state_torch = state_torch.to(self.device).unsqueeze(0)
#         if np.random.rand() > self.epsilon:
#             v = []
#             for i, action in enumerate(self.action_list):
#                 if self.policy_values == 'q':
#                     v.append(self.getStateActionValue(state_torch, action, type='q'))
#                 elif self.policy_values == 's':
#                     v.append(self.getStateActionValue(state_torch, type='s'))
#
#                 elif self.policy_values == 'qs':
#                     q = self.getStateActionValue(state_torch, action, type='q')
#                     s = self.getStateActionValue(state_torch, type='s')
#                     v.append((q + s) / 2)
#                 else:
#                     raise ValueError('policy is not defined')
#             action = self.action_list[np.argmax(v)]
#         else:
#             action = self.action_list[int(np.random.rand() * self.num_actions)]
#
#         return action
#
#     def agentState(self, observation):
#         # rep = self.getStateRepresentation(torch.from_numpy(observation).to(self.device))
#         # return rep
#         return torch.from_numpy(observation).to(self.device)
#         return np.copy(observation)
#
#     def updateNetworkWeights(self, network, step_size):
#         # another option: ** can use a optimizer here later**
#         optimizer = optim.SGD(network.parameters(), lr=step_size)
#         optimizer.step()
#         optimizer.zero_grad()
#
#         # for f in network.parameters():
#         #     f.data.sub_(step_size * f.grad.data)
#         # network.zero_grad()
#
#     def getActionIndex(self, action):
#         for i, a in enumerate(self.action_list):
#             if list(a) == list(action):
#                 return i
#         raise ValueError("action is not defined")
#
#     def getActionOnehot(self, action):
#         res = np.zeros([len(self.action_list)])
#         res[self.getActionIndex(action)] = 1
#         return res
#
#     def init_q_value_function_network(self):
#         nn_state_shape = (self.vf['q']['batch_size'],) + self.prev_state.shape
#         self.vf['q']['network'] = StateActionVFNN3(nn_state_shape, self.num_actions,
#                                                    self.vf['q']['layers_type'],
#                                                    self.vf['q']['layers_features'],
#                                                    self.vf['q']['action_layer_num']).to(self.device)
#
#     def init_s_value_function_network(self):
#         nn_state_shape = (self.vf['s']['batch_size'],) + self.prev_state.shape
#         self.vf['s']['network'] = []
#         for i in range(self.num_actions):
#             self.vf['s']['network'].append(StateVFNN(nn_state_shape,
#                                                      self.vf['s']['layers_type'],
#                                                      self.vf['s']['layers_features']).to(self.device))
#
#     def init_s_representation_network(self):
#         nn_state_shape =  self.prev_state.shape
#         self.representation['network'] = StateRepresentation(nn_state_shape,
#                                                              self.representation['layers_type'],
#                                                              self.representation['layers_features']).to(self.device)
#
#
#     def updateValueFunction(self, reward, x_old, prev_action, x_new=None, action=None):
#         self.batch_counter += 1
#         if prev_action is None:
#             raise ValueError('previous action not given')
#         if x_new is not None:  # Not a terminal State
#             if action is None:
#                 raise ValueError('action not given')
#
#             assert x_old.shape == x_new.shape, 'x_old and x_new have different shapes'
#             prev_action_index = self.getActionIndex(prev_action)
#
#             x_old = x_old.to(self.device)
#             x_new = x_new.to(self.device)
#             reward = torch.tensor(reward).to(self.device).unsqueeze(0)
#
#             if self.vf['q']['training']:
#                 target = reward + self.gamma * self.getStateActionValue(x_new, action, type='q', gradient=False)
#                 input = self.getStateActionValue(x_old, prev_action, type='q', gradient=True)
#
#                 assert target.shape == input.shape, 'target and input must have same shapes'
#                 loss = nn.MSELoss()(input, target)
#                 loss.backward()
#                 if self.batch_counter % self.vf['q']['batch_size'] == 0:
#                     step_size = self.vf['q']['step_size'] / self.vf['q']['batch_size']
#                     self.updateNetworkWeights(self.vf['q']['network'],
#                                               step_size)
#
#             if self.vf['s']['training']:
#                 target = reward + self.gamma * self.getStateActionValue(x_new, action, type='s', gradient=False)
#                 input = self.getStateActionValue(x_old, prev_action, type='s', gradient=True)
#                 assert target.shape == input.shape, 'target and input must have same shapes'
#                 loss = nn.MSELoss()(input, target)
#                 loss.backward()
#                 if self.batch_counter % self.vf['s']['batch_size'] == 0:
#                     step_size = self.vf['s']['step_size'] / self.vf['s']['batch_size']
#                     self.updateNetworkWeights(self.vf['s']['network'][prev_action_index],
#                                               step_size)
#         else:  # terminal state
#             prev_action_index = self.getActionIndex(prev_action)
#             prev_action_onehot = torch.from_numpy(self.getActionOnehot(prev_action)).float().unsqueeze(0).to(
#                 self.device)
#
#             x_old = x_old.unsqueeze(0).to(self.device)
#             reward = torch.tensor(reward).to(self.device).unsqueeze(0)
#
#             if self.vf['q']['training']:
#                 if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
#                     target = reward
#                     input = self.getStateActionValue(x_old, prev_action, type='q', gradient=True)
#                 else:
#                     target = reward.unsqueeze(0)
#                     input = self.getStateActionValue(x_old, prev_action, type='q', gradient=True)
#
#                 assert target.shape == input.shape, 'target and input must have same shapes'
#                 loss = nn.MSELoss()(input.float(), target.float())
#                 loss.backward()
#
#                 if self.batch_counter % self.vf['q']['batch_size'] == 0:
#                     step_size = self.vf['q']['step_size'] / self.vf['q']['batch_size']
#                 else:
#                     step_size = self.vf['q']['step_size'] / \
#                                 (self.batch_counter % self.vf['q']['batch_size'])
#
#                 self.updateNetworkWeights(self.vf['q']['network'], step_size)
#             if self.vf['s']['training']:
#                 target = torch.tensor(reward).unsqueeze(0)
#                 input = self.getStateActionValue(x_old, prev_action, type='s', gradient=True)
#                 assert target.shape == input.shape, 'target and input must have same shapes'
#                 loss = nn.MSELoss()(input.float(), target.float())
#                 loss.backward()
#
#                 if self.batch_counter % self.vf['s']['batch_size'] == 0:
#                     step_size = self.vf['s']['step_size'] / self.vf['s']['batch_size']
#                 else:
#                     step_size = self.vf['s']['step_size'] / \
#                                 (self.batch_counter % self.vf['s']['batch_size'])
#                 self.updateNetworkWeights(self.vf['s']['network'][prev_action_index], step_size)
#
#     def getStateActionValue(self, x, action=None, type='q', gradient=False):
#         x = x.unsqueeze(0)
#         if action is not None:
#             action_index = self.getActionIndex(action)
#             action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)
#
#             if type == 'q':
#                 if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
#                     value = self.vf['q']['network'](x).detach()[:, action_index] if not gradient \
#                         else self.vf['q']['network'](x)[:, action_index]
#                 else:
#                     value = self.vf['q']['network'](x, action_onehot).detach() if not gradient \
#                         else self.vf['q']['network'](x, action_onehot)
#
#             elif type == 's':
#                 value = self.vf['s']['network'][action_index](x).detach() if not gradient \
#                     else self.vf['s']['network'][action_index](x)
#
#             else:
#                 raise ValueError('state action value type is not defined')
#
#             return value
#         else:
#             # state value (no gradient)
#             if gradient :
#                 raise ValueError("cannot calculate the gradient for state values!")
#             sum = 0
#             for action in self.action_list:
#                 action_index = self.getActionIndex(action)
#                 action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)
#
#                 if type == 'q':
#                     if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
#                         value = self.vf['q']['network'](x).detach()[:, action_index]
#                     else:
#                         value = self.vf['q']['network'](x, action_onehot).detach()
#
#                 elif type == 's':
#                     value = self.vf['s']['network'][action_index](x).detach()
#
#                 else:
#                     raise ValueError('state action value type is not defined')
#
#                 sum += value
#
#             return sum / len(self.action_list)
#
#     def getStateRepresentation(self, state, gradient=False):
#         x = state.unsqueeze(0)
#         rep = self.representation['network'](x).detach() if not gradient else self.representation['network'](x)
#         return rep[0]
#
#     def getTransitionFromBuffer(self, n=1):
#         pass
#
#     @abstractmethod
#     def trainModel(self):
#         pass
#
#     @abstractmethod
#     def plan(self):
#         pass
#
#     @abstractmethod
#     def initModel(self):
#         pass
#
#     def updateTransitionBuffer(self, transition):
#         self.transition_buffer.append(transition)
#         if len(self.transition_buffer) > self.transition_buffer_size:
#             self.removeFromTransitionBuffer()
#
#     @abstractmethod
#     def removeFromTransitionBuffer(self):
#         self.transition_buffer.pop(0)


class BaseDynaAgent(BaseAgent):
    def __init__(self, params={}):
        self.time_step = 0
        self.writer = SummaryWriter()

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.transition_buffer = []
        self.transition_buffer_size = 100

        self.policy_values = 'q'  # 'q' or 's' or 'qs'

        self._vf = {'q': dict(network=None,
                             layers_type=['fc','fc'],
                             layers_features=[64,32],
                             action_layer_num=3,  # if one more than layer numbers => we will have num of actions output
                             batch_size=10,
                             step_size=0.001/10,
                             training=True),
                   's': dict(network=None,
                             layers_type=['fc'],
                             layers_features=[32],
                             batch_size=1,
                             step_size=0.01,
                             training=False)}

        self._sr = dict(network=None,
                       layers_type=[],
                       layers_features=[],
                       batch_size=None,
                       step_size=None,
                       batch_counter=None,
                       training=False)

        self._target_vf = dict(network=None,
                              counter=0,
                              layers_num=None,
                              action_layer_num=None,
                              update_rate=1,
                              type=None)

        self.reward_function = params['reward_function']
        self.device = params['device']
        self.goal = torch.from_numpy(params['goal']).float().to(self.device)

    def start(self, observation):
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)

        if self._vf['q']['network'] is None and self._vf['q']['training']:
            self.init_q_value_function_network(self.prev_state)  # a general state action VF for all actions
        if self._vf['s']['network'] is None and self._vf['s']['training']:
            self.init_s_value_function_network(self.prev_state)  # a separate state VF for each action
        if self._target_vf['network'] is None:
            self.setTargetValueFunction(self._vf['q'], 'q')


        self.prev_action = self.policy(self.prev_state)

        self.initModel()

        return self.prev_action

    def step(self, reward, observation):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)

        reward = torch.tensor(reward).unsqueeze(0).to(self.device)
        self.action = self.policy(self.state)

        #store the new transition in buffer
        self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
                                                     self.state, self.action, False, self.time_step))
        #update target
        if self._target_vf['counter'] >= self._target_vf['update_rate']:
            self.setTargetValueFunction(self._vf['q'], 'q')
            # self.setTargetValueFunction(self._vf['s'], 's')
            self._target_vf['counter'] = 0

        #update value function with the buffer
        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        #train/plan with model
        self.trainModel()
        self.plan()

        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action  # another option:** we can again call self.policy function **

        return self.prev_action

    def end(self, reward):
        reward = torch.tensor(reward).unsqueeze(0).to(self.device)

        self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
                                                     None, None, True, self.time_step))

        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        self.trainModel()
        self.updateStateRepresentation()

    def policy(self, state):
        '''
        :param state: torch -> (1, state_shape)
        :return: action: numpy array
        '''
        if np.random.rand() <= self.epsilon:
            ind = int(np.random.rand() * self.num_actions)
            return self.action_list[ind]
        v = []
        for i, action in enumerate(self.action_list):
            if self.policy_values == 'q':
                v.append(self.getStateActionValue(state, action, vf_type='q'))
            elif self.policy_values == 's':
                v.append(self.getStateActionValue(state, vf_type='s'))

            elif self.policy_values == 'qs':
                q = self.getStateActionValue(state, action, vf_type='q')
                s = self.getStateActionValue(state, vf_type='s')
                v.append((q + s) / 2)
            else:
                raise ValueError('policy is not defined')
        ind = np.argmax(v)
        return self.action_list[ind]


# ***
    def updateNetworkWeights(self, network, step_size):
        # another option: ** can use a optimizer here later**
        optimizer = optim.SGD(network.parameters(), lr=step_size)
        optimizer.step()
        optimizer.zero_grad()

        # for f in network.parameters():
        #     f.data.sub_(step_size * f.grad.data)
        # network.zero_grad()

# ***
    def init_q_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = state.shape
        self._vf['q']['network'] = StateActionVFNN4(nn_state_shape, self.num_actions,
                                                   self._vf['q']['layers_type'],
                                                   self._vf['q']['layers_features'],
                                                   self._vf['q']['action_layer_num']).to(self.device)

    def init_s_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = state.shape
        self._vf['s']['network'] = []
        for i in range(self.num_actions):
            self._vf['s']['network'].append(StateVFNN(nn_state_shape,
                                                     self._vf['s']['layers_type'],
                                                     self._vf['s']['layers_features']).to(self.device))

    def init_s_representation_network(self, observation):
        '''
        :param observation: numpy array
        :return: None
        '''
        nn_state_shape = observation.shape
        self._sr['network'] = StateRepresentation(nn_state_shape,
                                                 self._sr['layers_type'],
                                                 self._sr['layers_features']).to(self.device)

# ***
    def updateValueFunction(self, transition_batch, vf_type):
        for i, data in enumerate(transition_batch):
            prev_state, prev_action, reward, state, action, _, t = data
            self.calculateGradientValueFunction(vf_type, reward, prev_state, prev_action, state, action)
        self.updateNetworkWeights(self._vf[vf_type]['network'], self._vf[vf_type]['step_size'])
        self._target_vf['counter'] += 1

    def calculateGradientValueFunction(self, vf_type, reward, prev_state, prev_action, state=None, action=None):
        if prev_action is None:
            raise ValueError('previous action not given')
        prev_action_index = self.getActionIndex(prev_action)
        if vf_type == 'q':
            target = reward.float()
            if state is not None:  # Not a terminal State
                assert prev_state.shape == state.shape, 'x_old and x_new have different shapes'
                # target += self.gamma * self.getStateActionValue(state, action, gradient=False, type='q')
                target += self.gamma * self.getTargetValue(state, action)
            input = self.getStateActionValue(prev_state, prev_action, vf_type='q', gradient=True)
            assert target.shape == input.shape, 'target and input must have same shapes'
            loss = nn.MSELoss()(input, target)
            loss.backward()

        if vf_type == 's':
            target = reward.float()
            if state is not None:  # Not a terminal State
                assert prev_state.shape == state.shape, 'x_old and x_new have different shapes'
                # target += self.gamma * self.getStateActionValue(state, action, gradient=False, type='s')
                target += self.gamma * self.getTargetValue(state, action)
            input = self.getStateActionValue(prev_state, prev_action, vf_type='s', gradient=True)
            assert target.shape == input.shape, 'target and input must have same shapes'
            loss = nn.MSELoss()(input, target)
            loss.backward()

    def getStateActionValue(self, state, action=None, vf_type='q', gradient=False):
        '''
        :param state: torch -> [1, state_shape]
        :param action: numpy array
        :param vf_type: str -> 'q' or 's'
        :param gradient: boolean
        :return: value: int
        '''
        if action is not None:
            action_index = self.getActionIndex(action)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

            if vf_type == 'q':
                if len(self._vf['q']['layers_type']) + 1 == self._vf['q']['action_layer_num']:
                    value = self._vf['q']['network'](state).detach()[:, action_index] if not gradient \
                        else self._vf['q']['network'](state)[:, action_index]
                else:
                    value = self._vf['q']['network'](state, action_onehot).detach()[0] if not gradient \
                        else self._vf['q']['network'](state, action_onehot)[0]

            elif vf_type == 's':
                value = self._vf['s']['network'][action_index](state).detach()[0] if not gradient \
                    else self._vf['s']['network'][action_index](state)[0]

            else:
                raise ValueError('state action value type is not defined')
            return value
        else:
            # state value (no gradient)
            if gradient :
                raise ValueError("cannot calculate the gradient for state values!")
            sum = 0
            for action in self.action_list:
                action_index = self.getActionIndex(action)
                action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

                if vf_type == 'q':
                    if len(self._vf['q']['layers_type']) + 1 == self._vf['q']['action_layer_num']:
                        value = self._vf['q']['network'](state).detach()[:, action_index]
                    else:
                        value = self._vf['q']['network'](state, action_onehot).detach()[0]

                elif vf_type == 's':
                    value = self._vf['s']['network'][action_index](state).detach()[0]

                else:
                    raise ValueError('state action value type is not defined')

                sum += value

            return sum / len(self.action_list)

# ***
    def getStateRepresentation(self, observation, gradient=False):
        '''
        :param observation: numpy array -> [obs_shape]
        :param gradient: boolean
        :return: torch including batch -> [1, state_shape]
        '''
        if gradient:
            self._sr['batch_counter'] += 1
        observation = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        rep = self._sr['network'](observation).detach() if not gradient else self._sr['network'](observation)
        return rep

    def updateStateRepresentation(self):
        if len(self._sr['layers_type']) == 0:
            return None
        if self._sr['batch_counter'] == self._sr['batch_size'] and self._sr['training']:
            self.updateNetworkWeights(self._sr['network'], self._sr['step_size'] / self._sr['batch_size'])
            self._sr['batch_counter'] = 0

# ***
    def setTargetValueFunction(self, vf, type):
        if self._target_vf['network'] is None:
            nn_state_shape = self.prev_state.shape
            self._target_vf['network'] = StateActionVFNN4(
                                             nn_state_shape,
                                             self.num_actions,
                                             vf['layers_type'],
                                             vf['layers_features'],
                                             vf['action_layer_num']).to(self.device)

        self._target_vf['network'].load_state_dict(vf['network'].state_dict())  # copy weights and stuff
        if type != 's':
            self._target_vf['action_layer_num'] = vf['action_layer_num']
        self._target_vf['layers_num'] = len(vf['layers_type'])
        self._target_vf['counter'] = 0
        self._target_vf['type'] = type

    def getTargetValue(self, state, action=None):
        '''
        :param state: torch -> (1, state_shape)
        :param action: numpy array
        :return value: int
        '''
        if action is not None:
            action_index = self.getActionIndex(action)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

            if self._target_vf['type'] == 'q':
                if self._target_vf['layers_num'] + 1 == self._target_vf['action_layer_num']:
                    value = self._target_vf['network'](state).detach()[:, action_index]
                else:
                    value = self._target_vf['network'](state, action_onehot).detach()[0]

            elif self._target_vf['type'] == 's':
                value = self._target_vf['network'][action_index](state).detach()[0]

            else:
                raise ValueError('state action value type is not defined')
            return value
        else:
            # state value (no gradient)
            sum = 0
            for action in self.action_list:
                action_index = self.getActionIndex(action)
                action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

                if self._target_vf['type'] == 'q':
                    if self._target_vf['layers_num'] + 1 == self._target_vf['action_layer_num']:
                        value = self._target_vf['network'](state).detach()[:, action_index]
                    else:
                        value = self._target_vf['network'](state, action_onehot).detach()[0]

                elif self._target_vf['type'] == 's':
                    value = self._target_vf['network'][action_index](state).detach()

                else:
                    raise ValueError('state action value type is not defined')

                sum += value

            return sum / len(self.action_list)

# ***
    def getTransitionFromBuffer(self, n):
        # both model and value function are using this buffer
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.choices(self.transition_buffer, k=n)

    def updateTransitionBuffer(self, transition):
        self.transition_buffer.append(transition)
        if len(self.transition_buffer) > self.transition_buffer_size:
            self.removeFromTransitionBuffer()

    def removeFromTransitionBuffer(self):
        self.transition_buffer.pop(0)

# ***
    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(self, action):
        res = np.zeros([len(self.action_list)])
        res[self.getActionIndex(action)] = 1
        return res

#***
    @abstractmethod
    def trainModel(self):
        pass

    @abstractmethod
    def plan(self):
        pass

    @abstractmethod
    def initModel(self):
        pass

