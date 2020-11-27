import numpy as np
import torch
import random
import time

import Colab.utils as utils
from Colab.Agents.BaseDynaAgent import BaseDynaAgent
from Colab.Agents.MCTSAgent import BaseMCTSAgent


class DQNMCTSAgent(BaseMCTSAgent, BaseDynaAgent):
    name = "DQNMCTSAgent"

    def __init__(self, params={}):
        BaseMCTSAgent.__init__(self, params)
        BaseDynaAgent.__init__(self, params)
        self.action_consistency = 0
        self.learn_from_roullout = False
        self.learn_from_selection = False
        self.learn_from_expansion = True

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

    def selection(self):
        selected_node = self.root_node
        while len(selected_node.get_childs()) > 0:
            max_uct_value = -np.inf
            child_values = list(map(lambda n: n.get_avg_value(), selected_node.get_childs()))
            max_child_value = max(child_values)
            min_child_value = min(child_values)
            for child in selected_node.get_childs():
                if child.num_visits > 0:
                    child_value = child.get_avg_value()
                    if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                        child_value = (child.get_avg_value() - min_child_value) / (max_child_value - min_child_value)
                    uct_value = child_value + self.C * ((selected_node.num_visits / child.num_visits)**0.5)
                else:
                    uct_value = np.inf
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child

            if self.learn_from_selection:
                state = selected_node.parent.get_state()
                action = selected_node.action_from_par
                reward = selected_node.reward_from_par
                next_state = selected_node.get_state()
                is_terminal = selected_node.is_terminal
                self.learn_from_transition(state, action, reward, next_state, is_terminal)

        return selected_node

    def expansion(self, node):
        for a in self.action_list:
            state = node.get_state()
            next_state, is_terminal, reward = self.true_model(state, a)  # with the assumption of deterministic model
            if np.array_equal(next_state, state):  # To be deleted later - Gridworld related
                continue
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward)
            node.add_child(child)
            if self.learn_from_expansion:
                self.learn_from_transition(state, a, reward, next_state, is_terminal)

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
                if self.learn_from_roullout:
                    self.learn_from_transition(state, a, reward, next_state, is_terminal)
                state = next_state
        return np.average(returns)

    def learn_from_transition(self, state, a, reward, next_state, is_terminal):
        torch_state = self.getStateRepresentation(state)
        torch_next_state = self.getStateRepresentation(next_state)
        torch_reward = torch.tensor([reward], device=self.device)
        torch_action = torch.tensor([self.getActionIndex(a)], device=self.device).view(1, 1)
        transition = utils.transition(torch_state,
                                      torch_action,
                                      torch_reward,
                                      torch_next_state,
                                      None, is_terminal, self.time_step, 0)
        self.updateTransitionBuffer(transition)


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
