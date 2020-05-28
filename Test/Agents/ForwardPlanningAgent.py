from Test.Agents.BaseAgent import BaseAgent
from Test.Networks.StateValueFunction import StateVFNN
from Test.Networks.ModelNN.StateTransitionModel import StateTransitionModel
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class ForwardPlannerAgent(BaseAgent):
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
        self.vf_layers_type = ['fc']
        self.vf_layers_features = [32]
        self.model_layers_type = ['fc']
        self.model_layers_features = [32]
        self.greedy = False
        self.planning_steps = 2
        self.buffer_size = 1
        self.buffer = []
        self.state_transition_model = None
        self.reward_function = params['reward_function']

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
                self.q_value_function.append(StateVFNN(nn_state_shape, self.vf_layers_type, self.vf_layers_features))
        if self.state_transition_model is None:
            nn_state_shape = (self.batch_size,) + self.prev_state.shape
            self.state_transition_model = []
            for i in range(len(self.action_list)):
                self.state_transition_model.append(StateTransitionModel(nn_state_shape, self.vf_layers_type, self.vf_layers_features))

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.prev_action = self.policy(x_old, greedy= self.greedy)

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

        self.planForward()

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

    def planForward(self):
        for i in range (len(self.buffer)):
            state = self.buffer[i]
            next_state = None
            action = self.rollout_policy(state)
            action_index = self.getActionIndex(action)
            next_action = None
            for j in range(self.planning_steps):
                next_state = self.state_transition_model[action_index](state)
                next_action = self.rollout_policy(next_state)
                next_action_index = self.getActionIndex(next_action)
                x_old = torch.from_numpy(state).unsqueeze(0)
                x_new = torch.from_numpy(next_state).unsqueeze(0)
                reward = self.reward_function(next_state)
                target = reward + self.gamma * self.q_value_function[next_action_index](x_new).detach()
                action_index = self.getActionIndex(action)

                input = self.q_value_function[action_index](x_old)
                loss = nn.MSELoss()(input, target)

                loss.backward()
                self.updateWeights(action_index)

                state = next_state
                action = next_action


    def rollout_policy(self, state):
        action = self.action_list[int(np.random.rand() * self.num_actions)]
        return action




