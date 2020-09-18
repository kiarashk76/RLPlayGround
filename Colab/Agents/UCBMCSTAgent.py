from Colab.Agents.BaseMCTSAgent import BaseMCTSAgent, Node

import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from abc import abstractmethod
import random

import Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent
from Colab.Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN3, StateActionVFNN4
from Colab.Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Colab.Networks.RepresentationNN.StateRepresentation import StateRepresentation

debug = False

class UCBMCTSAgent(BaseMCTSAgent):
    name = 'BaseDynaAgent'

    def __init__(self, params={}):
        # BaseMCTSAgent.__init__(self, params)
        super().__init__(params)
        self.c = params['c']


    def policy(self, state):
        '''
        :param state: torch -> (1, state_shape)
        :return: action: numpy array
        '''
        with torch.no_grad():
            return self.mcts(state)


    def mcts(self, state):
        # print('h')
        state = state[0].cpu().numpy()
        num_iteration = 20
        tree = Node(state, val=self.getStateValue(state))
        for i in range(num_iteration):
            x = self.selection(tree)
            child = self.expansion(x)
            val = self.simulation(child)
            self.back_propagation(child, val)

        max_child_node = tree.children[0]
        max_ind = 0
        for i in range(1, len(tree.children)):
            next_child_node = tree.children[i]
            # if next_child_node.search_count > max_child_node.search_count:
            if self.get_ucb_val(next_child_node) > self.get_ucb_val(max_child_node):
                max_child_node = next_child_node
                max_ind = i
        selected_action = self.action_list[max_ind]
        return selected_action

    def selection(self, tree):
        node = tree
        while node.is_expanded:
            max_child_node = self.selection_policy(node, node.search_count)
            node = max_child_node
        return node

    def expansion(self, node):
        child = node.expand(self.true_model, self.action_list, self)
        return child

    def simulation(self, node):
        simulation_depth = 5
        reward_sum = 0
        state = node.state
        node_val = 0
        is_terminal = False
        for i in range(simulation_depth):
            action = self.rollout_policy(state)
            child_state, is_terminal, reward = self.true_model(state, action)
            reward_sum += reward
            if is_terminal:
                break
            state = child_state
        if not is_terminal:
            node_val = self.getStateValue(state)
        return node_val + reward_sum

    def back_propagation(self, node, new_val):
        # average of children
        if node.back_prop_type == 0:
            while node is not None:
                sum_val = node.search_count * node.search_val + new_val
                node.search_count += 1
                node.search_val = sum_val / node.search_count
                new_val += node.from_par_reward
                node = node.par

        # max of children
        elif node.back_prop_type == 1:
            while node is not None:
                # if node.search_count == 0:
                #     node.search_val = -np.inf
                node.search_count += 1
                node.search_val = max(node.search_val, new_val)
                new_val = node.search_val + node.from_par_reward
                node = node.par

    def getStateValue(self, state):
        epsilon = 0.1
        values = []
        torch_state = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(self.device)
        for action in self.action_list:
            torch_value = self.getTargetValue(torch_state, action=action)
            value = torch_value[0].cpu().numpy()
            values.append(value)
        return (1 - epsilon) * max(values) + epsilon * sum(values) / len(values)

    def selection_policy(self, node, N):
        non_terminal_children = []
        for i in range(len(node.children)):
            child = node.children[i]
            if child.state is None:
                continue
            non_terminal_children.append(child)
        max_val = -np.inf
        max_ind = 0
        for i in range(len(non_terminal_children)):
            child_val = self.get_ucb_val(non_terminal_children[i])
            if child_val > max_val:
                max_val = child_val
                max_ind = i
        return non_terminal_children[max_ind]

    def rollout_policy(self, state):
        epsilon = 0.3
        if np.random.rand() < epsilon:
            return self.action_list[int(np.random.rand() * len(self.action_list))]
        max_value = -np.inf
        max_action = None
        torch_state = torch.from_numpy(np.asarray(state)).unsqueeze(0)
        for action in self.action_list:
            value = self.getStateActionValue(torch_state, action=action, vf_type='q')
            if value > max_value:
                max_value = value
                max_action = action
        return max_action

    def get_ucb_val(self, node):
        exploit = node.search_val + node.from_par_reward
        explore = 2 * np.sqrt(node.par.search_count / node.search_count)
        return exploit + self.c * explore