
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
import random

import Colab.config as config, Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent
from Colab.Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN3
from Colab.Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Colab.Networks.RepresentationNN.StateRepresentation import StateRepresentation
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
        self.transition_buffer_size = 20

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
        self.device = params['device']
        self.goal = torch.from_numpy(params['goal']).float().to(self.device)

    def start(self, observation):
        self.batch_counter = 0
        self.prev_state = self.agentState(observation)

        if self.vf['q']['network'] is None and self.vf['q']['training']:
            self.init_q_value_function_network() # a general state action VF for all actions
        if self.vf['s']['network'] is None and self.vf['s']['training']:
            self.init_s_value_function_network() # a separate state VF for each action

        self.prev_action = self.policy(self.prev_state)

        self.initModel()

        return self.prev_action

    def step(self, reward, observation):
        self.time_step += 1
        self.state = self.agentState(observation)

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

    def agentState(self, observation):
        return torch.from_numpy(observation).to(self.device)
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
        if action != None:
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





# class BaseDynaAgent(BaseAgent):
#     def __init__(self, params={}):
#         self.time_step = 0
#         self.writer = SummaryWriter()
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
#                              layers_type=['fc'],
#                              layers_features=[32],
#                              action_layer_num=2,  # if one more than layer numbers => we will have num of actions output
#                              batch_size=5,
#                              batch_counter=0,
#                              step_size=0.01,
#                              training=True),
#                    's': dict(network=None,
#                              layers_type=['fc', 'fc'],
#                              layers_features=[64, 32],
#                              batch_size=5,
#                              batch_counter=0,
#                              step_size=0.01,
#                              training=False)}
#
#         self.representation = dict(network=None,
#                                    layers_type=['fc'],
#                                    layers_features=[64],
#                                    step_size=0.01,
#                                    batch_size=1,
#                                    batch_counter=0,
#                                    training=True)
#
#         self.reward_function = params['reward_function']
#         self.device = params['device']
#         self.goal = torch.from_numpy(params['goal']).float().to(self.device)
#
#     def start(self, observation):
#         if self.representation['network'] is None:
#             self.init_s_representation_network(observation)
#
#         self.prev_state = self.getStateRepresentation(observation, gradient=True)
#         self.prev_obs = observation
#
#         if self.vf['q']['network'] is None and self.vf['q']['training']:
#             self.init_q_value_function_network()  # a general state action VF for all actions
#         if self.vf['s']['network'] is None and self.vf['s']['training']:
#             self.init_s_value_function_network()  # a separate state VF for each action
#
#         self.prev_action = self.policy(self.prev_state)
#
#         self.initModel()
#
#         return self.prev_action
#
#     def step(self, reward, observation):
#
#         self.time_step += 1
#
#         self.state = self.getStateRepresentation(observation, gradient=True)
#
#         reward = torch.tensor(reward).unsqueeze(0).to(self.device)
#         self.action = self.policy(self.state)
#
#         # self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
#         #                                              self.state.detach(), self.action, False, self.time_step))
#         self.updateTransitionBuffer(utils.transition(self.prev_obs, self.prev_action, reward,
#                                                      observation, self.action, False, self.time_step))
#         self.updateValueFunction()
#         self.trainModel()
#         self.plan()
#         self.updateStateRepresentation()
#         # reward, self.prev_state, self.prev_action, x_new = self.state, action = self.action
#
#         self.prev_state = self.getStateRepresentation(observation, gradient = True)
#         self.prev_obs = observation
#         self.prev_action = self.action  # another option:** we can again call self.policy function **
#
#         return self.prev_action
#
#     def end(self, reward):
#         reward = torch.tensor(reward).unsqueeze(0).to(self.device)
#
#         self.updateTransitionBuffer(utils.transition(self.prev_obs, self.prev_action, reward,
#                                                      None, None, True, self.time_step))
#
#         self.updateValueFunction()
#         self.trainModel()
#         self.updateStateRepresentation()
#
#     def policy(self, state):
#         '''
#         :param state: torch -> (1, state_shape)
#         :return: action: numpy array
#         '''
#         if np.random.rand() <= self.epsilon:
#             ind = int(np.random.rand() * self.num_actions)
#             return self.action_list[ind]
#         state = state.detach()
#         v = []
#         for i, action in enumerate(self.action_list):
#             if self.policy_values == 'q':
#                 v.append(self.getStateActionValue(state, action, type='q'))
#             elif self.policy_values == 's':
#                 v.append(self.getStateActionValue(state, type='s'))
#
#             elif self.policy_values == 'qs':
#                 q = self.getStateActionValue(state, action, type='q')
#                 s = self.getStateActionValue(state, type='s')
#                 v.append((q + s) / 2)
#             else:
#                 raise ValueError('policy is not defined')
#         ind = np.argmax(v)
#         return self.action_list[ind]
#
#
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
#         nn_state_shape = self.prev_state.shape
#         self.vf['q']['network'] = StateActionVFNN4(nn_state_shape, self.num_actions,
#                                                    self.vf['q']['layers_type'],
#                                                    self.vf['q']['layers_features'],
#                                                    self.vf['q']['action_layer_num']).to(self.device)
#
#     def init_s_value_function_network(self):
#         nn_state_shape = self.prev_state.shape
#         self.vf['s']['network'] = []
#         for i in range(self.num_actions):
#             self.vf['s']['network'].append(StateVFNN(nn_state_shape,
#                                                      self.vf['s']['layers_type'],
#                                                      self.vf['s']['layers_features']).to(self.device))
#
#     def init_s_representation_network(self, observation):
#         nn_state_shape = observation.shape
#         self.representation['network'] = StateRepresentation(nn_state_shape,
#                                                              self.representation['layers_type'],
#                                                              self.representation['layers_features']).to(self.device)
#
#     def updateValueFunction(self):
#         batch = self.getTransitionFromBuffer()
#         for i, data in enumerate(batch):
#             prev_state, prev_action, reward, state, action, _, t = data
#             self.__updateValueFunction(reward, prev_state, prev_action, state, action)
#
#     def __updateValueFunction(self, reward, prev_obs, prev_action, obs=None, action=None):
#         if prev_action is None:
#             raise ValueError('previous action not given')
#         prev_action_index = self.getActionIndex(prev_action)
#         prev_state = self.getStateRepresentation(prev_obs, gradient=True)
#         # print(type(prev_obs))
#         if self.vf['q']['training']:
#             target = reward.float()
#             if obs is not None:  # Not a terminal State
#                 state = self.getStateRepresentation(obs, gradient=False)
#                 assert prev_state.shape == state.shape, 'x_old and x_new have different shapes'
#                 state = state.detach()
#                 target += self.gamma * self.getStateActionValue(state, action, type='q', gradient=False).float()
#             input = self.getStateActionValue(prev_state, prev_action, type='q', gradient=True)
#             assert target.shape == input.shape, 'target and input must have same shapes'
#             loss = nn.MSELoss()(input, target)
#             loss.backward()
#
#             self.vf['q']['batch_counter'] += 1
#             if self.vf['q']['batch_counter'] % self.vf['q']['batch_size'] == 0:
#                 step_size = self.vf['q']['step_size'] / self.vf['q']['batch_size']
#                 self.updateNetworkWeights(self.vf['q']['network'],
#                                           step_size)
#
#         if self.vf['s']['training']:
#             target = reward
#             if state is not None:  # Not a terminal State
#                 assert prev_state.shape == state.shape, 'x_old and x_new have different shapes'
#                 state = state.detach()
#                 target += self.gamma * self.getStateActionValue(state, action, type='s', gradient=False)
#
#             input = self.getStateActionValue(prev_state, prev_action, type='s', gradient=True)
#             assert target.shape == input.shape, 'target and input must have same shapes'
#             loss = nn.MSELoss()(input, target)
#             loss.backward()
#             self.vf['s']['batch_counter'] += 1
#             if self.vf['s']['batch_counter'] % self.vf['s']['batch_size'] == 0:
#                 step_size = self.vf['s']['step_size'] / self.vf['s']['batch_size']
#                 self.updateNetworkWeights(self.vf['s']['network'][prev_action_index],
#                                           step_size)
#
#
#     def getStateActionValue(self, state, action=None, type='q', gradient=False):
#         if action is not None:
#             action_index = self.getActionIndex(action)
#             action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)
#
#             if type == 'q':
#                 if len(self.vf['q']['layers_type']) + 1 == self.vf['q']['action_layer_num']:
#                     value = self.vf['q']['network'](state).detach()[:, action_index] if not gradient \
#                         else self.vf['q']['network'](state)[:, action_index]
#                 else:
#                     value = self.vf['q']['network'](state, action_onehot).detach() if not gradient \
#                         else self.vf['q']['network'](state, action_onehot)
#
#             elif type == 's':
#                 value = self.vf['s']['network'][action_index](state).detach() if not gradient \
#                     else self.vf['s']['network'][action_index](state)
#
#             else:
#                 raise ValueError('state action value type is not defined')
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
#                         value = self.vf['q']['network'](state).detach()[:, action_index]
#                     else:
#                         value = self.vf['q']['network'](state, action_onehot).detach()
#
#                 elif type == 's':
#                     value = self.vf['s']['network'][action_index](state).detach()
#
#                 else:
#                     raise ValueError('state action value type is not defined')
#
#                 sum += value
#
#             return sum / len(self.action_list)
#
#     def getStateRepresentation(self, observation, gradient=False):
#         '''
#         :param observation: numpy array
#         :param gradient: boolean
#         :return: torch including batch -> [1, state_shape]
#         '''
#         observation = torch.from_numpy(observation).unsqueeze(0).to(self.device)
#         rep = self.representation['network'](observation).detach() if not gradient else self.representation['network'](observation)
#         return rep
#         # return observation.flatten(start_dim=1)
#
#     def updateStateRepresentation(self):
#         self.updateNetworkWeights(self.representation['network'], self.representation['step_size'])
#
#
#
#     def getTransitionFromBuffer(self, n=1):
#         # both model and value function are using this buffer
#         if len(self.transition_buffer) < n:
#             n = len(self.transition_buffer)
#         return random.choices(self.transition_buffer, k=n)
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
