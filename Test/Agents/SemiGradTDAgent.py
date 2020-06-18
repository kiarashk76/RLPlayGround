from Test.Agents.BaseAgent import BaseAgent
from ValueFunction.StateActionValueFunction import StateActionVFNN3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

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
        self.layers_type = ['fc','fc']
        self.layers_features = [64,32]

        # default `log_dir` is "runs" - we'll be more specific here
        self.writer = SummaryWriter('runs/')

    def start(self, observation):
        '''
        :param observation: numpy array
        :return: action:
        '''

        self.prev_state = self.agentState(observation)
        if self.q_value_function is None:
            nn_state_shape = (self.batch_size, ) + self.prev_state.shape
            nn_action_shape = self.num_actions
            self.q_value_function = StateActionVFNN3(nn_state_shape, nn_action_shape,
                                                    self.layers_type, self.layers_features)
        self.q_value_function.zero_grad()
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.prev_action = self.policy(x_old)

        return self.prev_action

    def step(self, reward, observation):

        '''
        :param reward: int
        :param observation: numpy array
        :return: action
        '''
        self.state = self.agentState(observation)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0).float()
        x_new = torch.from_numpy(self.state).unsqueeze(0).float()
        self.action = self.policy(x_new)

        target = reward + self.gamma * \
                 self.q_value_function(x_new).detach()[:, self.getActionIndex(self.action)]

        input = self.q_value_function(x_old)[:, self.getActionIndex(self.prev_action)]

        loss = nn.MSELoss()(input.float(), target.float())
        loss.backward()
        self.updateWeights()

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
        target = torch.tensor(reward).unsqueeze(0)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)

        input = self.q_value_function(x_old)[:, self.getActionIndex(self.prev_action)]

        loss = nn.MSELoss()(input.float(), target.float())
        loss.backward()
        self.updateWeights()

    def policy(self, agent_state, greedy = False):
        if np.random.rand() > self.epsilon or greedy:
            v = []
            for a in self.action_list:
                v.append(self.q_value_function(agent_state.float()).detach()[0, self.getActionIndex(a)])
            action = self.action_list[np.argmax(v)]
        else:
            action = self.action_list[int(np.random.rand() * self.num_actions)]

        return action

    def agentState(self, observation):
        return np.copy(observation)

    def updateWeights(self):
        for f in self.q_value_function.parameters():
            # print("grad:",f.grad.data)
            f.data.sub_(self.step_size * f.grad.data)
        self.q_value_function.zero_grad()

    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def getActionIndexOneHot(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                a = np.zeros([self.num_actions])
                a[i] = 1
                return a
        raise ValueError("action is not defined")