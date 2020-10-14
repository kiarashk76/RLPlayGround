import numpy as np
import torch

from Colab.Agents.BaseMCTSAgent import BaseMCTSAgent
from Colab.DataStructures.Node import Node

debug = False

class UCBMCTSAgent(BaseMCTSAgent):
    name = 'BaseDynaAgent'

    def __init__(self, params={}):
        super().__init__(params)
        self.c = params['c']
        self.root = None
        self.num_iteration = params['num_iteration'] #2
        self.simulation_depth = params['simulation_depth'] #20
        self.num_simulation = params['num_simulation'] #1

    def start(self, observation):
        self.root = None
        return super(UCBMCTSAgent, self).start(observation)

    def policy(self, state):
        '''
        :param state: torch -> (1, state_shape)
        :return: action: numpy array
        '''
        with torch.no_grad():
            action, self.root = self.mcts(state)
            return action

    def mcts(self, state):
        state = state[0].cpu().numpy()
        tree = self.root
        if tree is None:
            tree = Node(state, val=self.getStateValue(state))
        tree.par = None
        for i in range(self.num_iteration):
            x = self.selection(tree)
            child = self.expansion(x)
            val = self.simulation(child)
            self.back_propagation(child, val)

        max_ind = 0
        max_val = -np.inf
        for i in range(len(tree.children)):
            next_child_node = tree.children[i]
            next_child_value = self.get_select_val(next_child_node)
            if next_child_value > max_val:
                max_ind = i
                max_val = next_child_value
        selected_action = self.action_list[max_ind]
        return selected_action, tree.children[max_ind]

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
        sum = 0
        for i in range(self.num_simulation):
            sum += self.single_simulation(node)
        sum /= self.num_simulation
        return sum

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
                node.search_count += 1
                node.search_val = max(node.search_val, new_val)
                new_val = node.search_val + node.from_par_reward
                node = node.par

    def getStateValue(self, state):
        approx_state_val = True
        if approx_state_val:
            epsilon = 0.1
            values = []
            torch_state = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(self.device)
            for action in self.action_list:
                torch_value = self.getTargetValue(torch_state, action=action)
                value = torch_value[0].cpu().numpy()
                values.append(value)
            return (1 - epsilon) * max(values) + epsilon * sum(values) / len(values)
        else:
            x = state[0]
            y = state[1]
            value = -(np.abs(x - 0) + np.abs(y - 3)) + 10
            mu = 10
            sigma = 1
            noise = np.abs(np.random.normal(mu - value, sigma))
            return value + noise

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
        epsilon = 0.0
        if np.random.rand() < epsilon:
            return self.action_list[int(np.random.rand() * len(self.action_list))]
        max_value = -np.inf
        max_action = None
        torch_state = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(self.device)
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

    def get_select_val(self, node):
        return node.search_val + node.from_par_reward

    def single_simulation(self, node):
        reward_sum = 0
        state = node.state
        node_val = 0
        is_terminal = False
        for i in range(self.simulation_depth):
            action = self.rollout_policy(state)
            child_state, is_terminal, reward = self.true_model(state, action)
            reward_sum += reward
            if is_terminal:
                break
            state = child_state
        if not is_terminal:
            node_val = self.getStateValue(state)
        return node_val + reward_sum