from Test.Agents.BaseAgent import BaseAgent
from ValueFunction.StateActionValueFunction import StateActionVFNN3
from ValueFunction.StateValueFunction import StateVFNN
from ModelNN.StateTransitionModel import StateTransitionModel
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod

class BaseDynaAgent(BaseAgent):
    def __init__(self, params = {}):
        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.greedy = False
        self.epsilon = params['epsilon']

        self.policy_values = 'qs' # 'q' or 's' or 'qs'
        self.vf = {'q': dict(network=None,
                             layers_type=['fc', 'fc'],
                             layers_features=[64, 32],
                             batch_size=10,
                             step_size=0.01,
                             training=True),
                   's': dict(network=None,
                             layers_type=['fc', 'fc'],
                             layers_features=[64, 32],
                             batch_size=5,
                             step_size=0.01,
                             training=True)}

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

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.prev_action = self.policy(x_old, greedy = self.greedy)

        self.initModel()

        return self.prev_action

    def step(self, reward, observation):
        self.batch_counter += 1
        self.state = self.agentState(observation)

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        x_new = torch.from_numpy(self.state).unsqueeze(0)

        self.action = self.policy(x_new, greedy=self.greedy)

        self.updateValueFunction(reward, x_old, x_new)

        self.prev_state = self.state
        self.prev_action = self.action # another option:** we can again call self.policy function **

        self.trainModel(self.prev_state, self.action, self.state)
        self.plan()

        return self.prev_action

    def end(self, reward):
        self.batch_counter += 1

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)

        self.updateValueFunction(reward, x_old, None)

    def policy(self, state_torch, greedy = False):
        if np.random.rand() > self.epsilon or greedy:
            v = []
            for i, action in enumerate(self.action_list):
                if self.policy_values == 'q':
                    v.append(self.vf['q']['network'](state_torch).detach()[0,i])
                elif self.policy_values == 's':
                    v.append(self.vf['s']['network'][i](state_torch).detach())
                elif self.policy_values == 'qs':
                    q = self.vf['q']['network'](state_torch).detach()[0,i]
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
        for f in network.parameters():
            f.data.sub_(step_size * f.grad.data)
        network.zero_grad()

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
                                                 self.vf['q']['layers_features']).to(self.device)

    def init_s_value_function_network(self):
        nn_state_shape = (self.vf['s']['batch_size'],) + self.prev_state.shape
        self.vf['s']['network'] = []
        for i in range(self.num_actions):
            self.vf['s']['network'].append(StateVFNN(nn_state_shape,
                                                     self.vf['s']['layers_type'],
                                                     self.vf['s']['layers_features']).to(self.device))

    def updateValueFunction(self, reward, x_old, x_new = None, prev_action=None, action=None):
        if x_new is not None: # Not a terminal State
            if prev_action is None:
                prev_action = self.prev_action
            if action is None:
                action = self.action

            prev_action_index = self.getActionIndex(self.prev_action)
            action_index = self.getActionIndex(self.action)

            if self.vf['q']['training']:
                target = reward + self.gamma * self.vf['q']['network'](x_new).detach()[:, action_index]
                input = self.vf['q']['network'](x_old)[:, prev_action_index]
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
            prev_action_index = self.getActionIndex(self.prev_action)

            if self.vf['q']['training']:
                target = torch.tensor(reward).unsqueeze(0)
                input = self.vf['q']['network'](x_old)[:, prev_action_index]
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

    @abstractmethod
    def trainModel(self, state, action, next_state):
        pass

    @abstractmethod
    def plan(self):
        pass

    @abstractmethod
    def initModel(self):
        pass
