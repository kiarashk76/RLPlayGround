import copy
import numpy as np

from abc import abstractmethod
import random
from torch.utils.tensorboard import SummaryWriter
from ete3 import Tree
import Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent

from ete3 import Tree, TreeStyle, TextFace, add_face_to_node
import time

class BaseMCTSAgent(BaseAgent):
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


        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_tree = True

    def start(self, observation):

        self.root_node = Node(None, observation)
        self.expansion(self.root_node)

        for i in range(self.num_iterations):
            a, sub_tree = self.MCTS_iteration()
        # self.render_tree()
        self.root_node = sub_tree

        return a

    def step(self, reward, observation):

        if not self.keep_tree:
            self.root_node = Node(None, observation)
            self.expansion(self.root_node)

        for i in range(self.num_iterations):
            a, sub_tree = self.MCTS_iteration()
        # self.render_tree()
        self.root_node = sub_tree

        return a

    def end(self, reward):
        pass

    def get_initial_value(self, state):
        return 0

    def MCTS_iteration(self):
        # self.render_tree()
        selected_node = self.selection()
        #now we decide to expand the leaf or rollout
        if selected_node.num_visits == 0: # don't expand just roll-out
            rollout_value = self.rollout(selected_node)
            self.backpropagate(selected_node, rollout_value)

        else: # expand then roll_out
            self.expansion(selected_node)
            rollout_value = self.rollout(selected_node.get_childs()[0])
            self.backpropagate(selected_node.get_childs()[0], rollout_value)

        max_visit = -np.inf
        max_action = None
        max_child = None
        for child in self.root_node.get_childs():
            if child.get_avg_value() > max_visit:
                max_visit = child.get_avg_value()
                max_action = child.get_action_from_par()
                max_child = child
        return max_action, max_child

    def selection(self):
        selected_node = self.root_node
        while len(selected_node.get_childs()) > 0:
            max_uct_value = -np.inf
            child_values = list(map(lambda n: n.get_avg_value(), selected_node.get_childs()))
            max_child_value = max(child_values)
            min_child_value = min(child_values)
            for ind, child in enumerate(selected_node.get_childs()):
                if child.num_visits == 0:
                    selected_node = child
                    break
                else:
                    child_value = child_values[ind]
                    if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                        child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                    uct_value = child_value + \
                                self.C * ((selected_node.num_visits / child.num_visits)**0.5)
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child
        return selected_node

    def expansion(self, node):
        for a in self.action_list:
            next_state, is_terminal, reward = self.true_model(node.get_state(), a) # with the assumption of deterministic model
            if np.array_equal(next_state, node.get_state()):
                continue
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par = reward, value = value)
            node.add_child(child)

    def rollout(self, node):
        sum_returns = 0
        for i in range(self.num_rollouts):
            depth = 0
            single_return = 0
            is_terminal = False
            state = node.get_state()
            while not is_terminal and depth < self.rollout_depth:
                a = random.choice(self.action_list)
                next_state, is_terminal, reward = self.true_model(state, a)
                single_return += reward
                depth += 1
                state = next_state
            sum_returns += single_return
        return sum_returns / self.num_rollouts

    def backpropagate(self, node, value):
        while node is not None:
            node.add_to_values(value)
            node.inc_visits()
            value *= self.gamma
            value += node.reward_from_par
            node = node.parent


    def show(self):
        queue = [self.root_node,"*"]
        while queue:
            node = queue.pop(0)
            if node == "*":
                print("********")
                continue
            node.show()
            for child in node.get_childs():
                queue.append(child)
            if len(node.get_childs()) > 0:
                queue.append("*")

    def render_tree(self):
        def my_layout(node):
            F = TextFace(node.name, tight_text=True)
            add_face_to_node(F, node, column=0, position="branch-right")
        t = Tree()
        ts = TreeStyle()
        ts.show_leaf_name = False
        queue = [(self.root_node, None)]
        while queue:
            node, parent = queue.pop(0)
            node_face = str(node.get_state())+","+str(node.num_visits)+","+str(node.get_avg_value())
            if parent is None:
                p = t.add_child(name=node_face)
            else:
                p = parent.add_child(name=node_face)
            for child in node.get_childs():
                queue.append((child, p))

        ts.layout_fn = my_layout
        # t.render('t.png', tree_style=ts)
        # print(t.get_ascii(show_internal=Tree))
        t.show(tree_style=ts)


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
        return self.sum_values / (self.num_visits + 1)

        if self.num_visits > 0:
            return self.sum_values / (self.num_visits + 1)
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