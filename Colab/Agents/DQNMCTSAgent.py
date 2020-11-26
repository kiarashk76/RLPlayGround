import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

import Colab.utils as utils
from Colab.Agents.BaseDynaAgent import BaseDynaAgent
from Colab.Agents.MCTSAgent import BaseMCTSAgent


import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from abc import abstractmethod
import random

import Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent
from Colab.Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN3, StateActionVFNN4
from Colab.Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Colab.Networks.RepresentationNN.StateRepresentation import StateRepresentation


class DQNMCTSAgent(BaseMCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent"

    def __init__(self, params={}):
        BaseMCTSAgent.__init__(self, params)
        BaseDynaAgent.__init__(self, params)
        self.action_consistency = 0

    def start(self, observation):
        self.action_consistency = 0
        BaseDynaAgent.start(self, observation)
        action = BaseMCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        self.time_step += 1

        action = BaseMCTSAgent.step(self, reward, observation)

        # update target
        if self._target_vf['counter'] >= self._target_vf['update_rate']:
            self.setTargetValueFunction(self._vf['q'], 'q')

        # update value function with the buffer
        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        self.updateStateRepresentation()

        state = self.getStateRepresentation(observation)
        dqn_action_index = (self.policy(state)).item()
        dqn_action = self.action_list[dqn_action_index]
        if np.array_equal(dqn_action, action):
            self.action_consistency += 1
        return action

    def end(self, reward):
        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

    def expansion(self, node):
        for a in self.action_list:
            state = node.get_state()
            next_state, is_terminal, reward = self.true_model(state, a)  # with the assumption of deterministic model
            torch_state = self.getStateRepresentation(state)
            torch_next_state = self.getStateRepresentation(next_state)
            torch_reward = torch.tensor([reward], device=self.device)
            torch_action = torch.tensor([self.getActionIndex(a)], device=self.device).view(1,1)
            transition = utils.transition(torch_state,
                             torch_action,
                             torch_reward,
                             torch_next_state,
                             None, is_terminal, self.time_step, 0)
            self.updateTransitionBuffer(transition)
            if np.array_equal(next_state, state):
                continue
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward)
            node.add_child(child)

    def rollout(self, node):
        is_terminal = False
        state = node.get_state()
        returns = np.zeros([self.num_rollouts])
        for i in range(self.num_rollouts):
            depth = 0
            while not is_terminal and depth < self.rollout_depth:
                a = random.choice(self.action_list)
                next_state, is_terminal, reward = self.true_model(state, a)
                returns[i] += reward
                depth += 1
                state = next_state
        return np.average(returns)

class Node():
    def __init__(self, parent, state, value=0, is_terminal=False, action_from_par=None, reward_from_par=0):
        self.state = state
        self.sum_values = value
        self.num_visits = 0
        self.childs_list = []
        self.parent = parent
        self.is_terminal = is_terminal
        self.action_from_par = action_from_par
        self.reward_from_par = reward_from_par

    def get_action_from_par(self):
        return np.copy(self.action_from_par)

    def add_child(self, child):
        self.childs_list.append(child)

    def get_childs(self):
        return self.childs_list.copy()

    def add_to_values(self, value):
        self.sum_values += value

    def get_avg_value(self):
        if self.num_visits > 0:
            return self.sum_values / self.num_visits
        else:
            return np.inf

    def inc_visits(self):
        self.num_visits += 1

    def get_state(self):
        return np.copy(self.state)

    def show(self):
        try:
            print("state: ", self.state, " value: ", self.sum_values, " num_visits: ", self.num_visits, " parent: ",
                  self.parent.get_state())
        except AttributeError:
            print("state: ", self.state, " value: ", self.sum_values, " num_visits: ", self.num_visits, " parent: ",
                  None)
