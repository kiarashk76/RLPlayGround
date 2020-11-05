import copy
import numpy as np

from abc import abstractmethod
import random
from torch.utils.tensorboard import SummaryWriter
import torch

import Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent

debug = False


class BaseMCTSAgent:
    name = "BaseMCTSAgent"
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

        self.device = params['device']
        self.true_model = params['true_fw_model']

    def start(self, observation):
        return self.mcts(observation)


    def step(self, reward, observation):

        return self.mcts(observation)

    def end(self, reward):
        pass


    def mcts(self, state):
        num_iteration = 5
        self.num_simulation = 300
        tree = Node(state, val=0)
        for i in range(num_iteration):
            # if debug:
            # print('-----------------------')
            # print('mcts iteration num: ', i)
            # print('selection----')
            x = self.selection(tree)
            if debug:
                print('selected:', type(x.state))
                print('expansion----')
            child = self.expansion(x)
            if debug:
                print('expanded:', type(child.state))
                print('simulation----')
            val = self.simulation(child)
            if debug:
                print('back propagation----')
            self.back_propagation(child, val)

        max_child_node = tree.children[0]
        max_ind = 0
        for i in range(1, len(tree.children)):
            next_child_node = tree.children[i]
            if next_child_node.get_mcts_val() > max_child_node.get_mcts_val():
                max_child_node = next_child_node
                max_ind = i
        selected_action = self.action_list[max_ind]
        if debug:
            for i in range(0, len(tree.children)):
                print('mcts val:', tree.children[i].get_mcts_val())
                print(state, ' ----- ', max_ind)
        return selected_action


    def selection(self, tree):
        node = tree
        while node.is_expanded:
            max_child_node = self.expansion_policy(node, tree.search_count)
            node = max_child_node
        return node


    def expansion(self, node):
        child = node.expand(self.true_model, self.action_list, self)
        return child

    def simulation(self, node):
        sum = 0
        for i in range(self.num_simulation):
            sum += self.single_simulation(node)
        sum /= self.num_simulation
        return sum

    def single_simulation(self, node):
        simulation_depth = 0
        reward_sum = 0
        state = node.state
        is_terminal = False
        while not is_terminal:
            rand = int(np.random.rand() * len(self.action_list))
            action = self.action_list[rand]
            child_state, is_terminal, reward = self.true_model(state, action)
            reward_sum += reward
            state = child_state
            # node_val = -np.abs(state[0] - self.goal[0]) - np.abs(state[1] - self.goal[1])
        return reward_sum


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


    def expansion_policy(self, node, N):
        non_terminal_children = []
        for i in range(len(node.children)):
            child = node.children[i]
            if child.state is None:
                continue
            non_terminal_children.append(child)
        return np.random.choice(non_terminal_children)


class Node:
    def __init__(self, state, par=None, val=0, from_par_reward=0, from_root_reward=0):
        self.state = state
        self.par = par
        self.children = []
        self.is_expanded = False
        self.val = val #state value function
        self.from_par_reward = from_par_reward
        self.from_root_reward = from_root_reward
        self.back_prop_type = 0 #0: average, 1: max
        if self.back_prop_type == 0:
            self.search_val = val
        else:
            self.search_val = val
        self.search_count = 1

    def expand(self, model, action_list, agent):
        non_terminal_children = []
        for action in action_list:
            child_state, is_terminal, reward = model(self.state, action)
            child_from_root_reward = self.from_root_reward + reward
            if is_terminal:
                child = Node(None, self, 0, reward, child_from_root_reward)
                child.search_val = 0
            else:
                child_val = 0 #agent.getStateValue(child_state)
                child = Node(child_state, self, child_val, reward, child_from_root_reward)
                non_terminal_children.append(child)
            self.children.append(child)
        self.is_expanded = True
        rand = int(np.random.rand() * len(non_terminal_children))
        return non_terminal_children[rand]

    def get_mcts_val(self):
        #change
        return self.search_val + self.from_root_reward
        # return self.from_root_reward