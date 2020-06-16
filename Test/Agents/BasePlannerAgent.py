from Test.Agents.BaseAgent import BaseAgent
from ValueFunction.StateValueFunction import StateVFNN
from ModelNN.StateTransitionModel import StateTransitionModel
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod

class BasePlannerAgent(BaseAgent):
    def __init__(self, params = {}):
        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1]

        self.gamma = params['gamma']
        self.step_size = params['step_size']
        self.greedy = False
        self.epsilon = params['epsilon']

        self.q_value_function = None
        self.vf_layers_type = ['fc','fc']
        self.vf_layers_features = [32,32]
        self.batch_size = params['batch_size']

        self.model = params['model']
        self.model_step_size = params['model_step_size']
        self.model_layers_type = ['fc','fc']
        self.model_layers_features = [32,32]
        self.planning_steps = 3
        self.buffer_size = 1
        self.buffer = []

        self.reward_function = params['reward_function']
        self.goal = params['goal']
        self.device = params['device']

    def start(self, observation):
        '''
        :param observation: numpy array
        :return: action:
        '''

        self.prev_state = self.agentState(observation)
        if self.q_value_function is None:
            nn_state_shape = (self.batch_size,) + self.prev_state.shape
            self.q_value_function = []
            for i in range(len(self.action_list)):
                v = StateVFNN(nn_state_shape, self.vf_layers_type, self.vf_layers_features)
                v.to(self.device)
                self.q_value_function.append(v)

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.prev_action = self.policy(x_old, greedy = self.greedy)

        if self.model is None:
            nn_state_shape = (self.batch_size,) + self.prev_state.shape
            nn_action_shape = (self.batch_size,) + self.prev_action.shape
            self.model = StateTransitionModel(nn_state_shape, nn_action_shape,
                                                                   self.model_layers_type,
                                                                   self.model_layers_features,
                                                                   action_layer_num = 1)
            self.model.to(self.device)

        return self.prev_action

    def step(self, reward, observation):

        '''
        :param reward: int
        :param observation: numpy array
        :return: action
        '''

        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(self.prev_state)

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.state = self.agentState(observation)
        x_new = torch.from_numpy(self.state).unsqueeze(0)
        prev_action_index = self.getActionIndex(self.prev_action)
        self.action = self.policy(x_new, greedy = self.greedy)
        action_index = self.getActionIndex(self.action)

        target = reward + self.gamma * self.q_value_function[action_index](x_new).detach()
        input = self.q_value_function[prev_action_index](x_old)
        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateWeights(prev_action_index)

        self.prev_state = self.state
        self.prev_action = self.action

        self.trainModel(self.prev_state, self.prev_action, self.state)
        self.plan()

        return self.prev_action

    def end(self, reward):
        '''
        :param reward: int
        :return: none
        '''
        print("reached the goal!!!")
        target = torch.tensor(reward).unsqueeze(0).unsqueeze(0)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)

        prev_action_index = self.getActionIndex(self.prev_action)

        input = self.q_value_function[prev_action_index](x_old)
        loss = nn.MSELoss()(input.float(), target.float())
        loss.backward()
        self.updateWeights(prev_action_index)

    def policy(self, agent_state, greedy = False):
        if np.random.rand() > self.epsilon or greedy:
            v = []
            for i, action in enumerate(self.action_list):
                v.append(self.q_value_function[i](agent_state))
            action = self.action_list[np.argmax(v)]
        else:
            action = self.action_list[int(np.random.rand() * self.num_actions)]

        return action

    def agentState(self, observation):
        return np.copy(observation)

    def updateWeights(self, action_index):
        for f in self.q_value_function[action_index].parameters():
            f.data.sub_(self.step_size * f.grad.data)
        self.q_value_function[action_index].zero_grad()

    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    @abstractmethod
    def trainModel(self, state, action, next_state):
        raise NotImplementedError('call child plan')

    def updateModelWeights(self):
        for f in self.model.parameters():
            f.data.sub_(self.model_step_size * f.grad.data)
        self.model.zero_grad()

    @abstractmethod
    def plan(self):
        raise NotImplementedError('call child plan')
