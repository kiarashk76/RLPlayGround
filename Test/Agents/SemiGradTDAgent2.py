from Test.Agents.BaseAgent import BaseAgent
from Test.Networks.StateValueFunction import StateVFNN
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

class SemiGradientTD(BaseAgent):
    def __init__(self, params = {}):
        self.q_value_function = None
        self.action_list = params['action_list']
        self.actions_shape = self.action_list.shape[1]
        self.num_actions = self.action_list.shape[0]
        self.prev_state = None
        self.state = None
        self.gamma = params['gamma']
        self.step_size = params['step_size']
        self.epsilon = params['epsilon']
        self.batch_size = params['batch_size']
        self.layers_type = ['fc']
        self.layers_features = [32]
        self.greedy = False

        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter('runs/')

    def start(self, observation):
        '''
        :param observation: numpy array
        :return: action:
        '''
        # raise NotImplementedError('Expected `start` to be implemented')

        self.prev_state = self.agentState(observation)
        if self.q_value_function is None:
            nn_state_shape = (self.batch_size,) + self.prev_state.shape
            self.q_value_function = []
            for i in range(len(self.action_list)):
                self.q_value_function.append(StateVFNN(nn_state_shape, self.layers_type, self.layers_features))

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.prev_action = self.policy(x_old, greedy= self.greedy)

        return self.prev_action

    def step(self, reward, observation):

        '''
        :param reward: int
        :param observation: numpy array
        :return: action
        '''
        self.state = self.agentState(observation)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        x_new = torch.from_numpy(self.state).unsqueeze(0)
        self.action = self.policy(x_new, greedy= self.greedy)

        action_index = self.getActionIndex(self.action)
        prev_action_index = self.getActionIndex(self.prev_action)

        target = reward + self.gamma * self.q_value_function[action_index](x_new).detach()
        input = self.q_value_function[prev_action_index](x_old)
        loss = nn.MSELoss()(input, target)

        loss.backward()
        self.updateWeights(prev_action_index)

        self.prev_state = self.state
        self.prev_action = self.action

        self.writer.add_scalar('loss', loss.item())
        self.writer.close()
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

    def updateWeights(self, action):
        for f in self.q_value_function[action].parameters():
            f.data.sub_(self.step_size * f.grad.data)
        self.q_value_function[action].zero_grad()


    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")
