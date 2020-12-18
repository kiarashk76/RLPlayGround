import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from ete3 import Tree
import Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent
from Colab.Agents.MCTSAgent import BaseMCTSAgent
from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent
from Colab.Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN3, StateActionVFNN4
from Colab.Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Colab.Networks.RepresentationNN.StateRepresentation import StateRepresentation
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from ete3 import Tree, TreeStyle, TextFace, add_face_to_node
import time

class BaseDQNMCTSAgent(BaseMCTSAgent, BaseDynaAgent):
    name = "BaseMCTSAgent"
    def __init__(self, params={}):
        BaseDynaAgent.__init__(self, params)
        BaseMCTSAgent.__init__(self, params)
        self.episode_counter = 0

    def start(self, observation):
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.start(self, observation)
        else:
            action = BaseMCTSAgent.start(self, observation)
        return action

    def step(self, reward, observation):
        if self.episode_counter % 2 == 0:
            action = BaseDynaAgent.step(self, reward, observation)
        else:
            action = BaseMCTSAgent.step(self, reward, observation)

        return action

    def end(self, reward):
        BaseDynaAgent.end(self, reward)
        self.episode_counter += 1

    def get_initial_value(self, state):
        state_representation = self.getStateRepresentation(state)
        value = self.getStateActionValue(state_representation)
        return value.item()

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
            print("state: ", self.state, " value: ",self.sum_values," num_visits: ", self.num_visits, " parent: ", self.parent.get_state())
        except AttributeError:
            print("state: ", self.state, " value: ", self.sum_values, " num_visits: ", self.num_visits, " parent: ",None)