import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import random

import Colab.utils as utils
from Colab.Agents.BaseAgent import BaseAgent
from Colab.Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN3, StateActionVFNN4
from Colab.Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Colab.Networks.RepresentationNN.StateRepresentation import StateRepresentation
from Colab.DataStructures.Node import Node

debug = False

class BaseMCTSAgent(BaseAgent):
    name = 'BaseDynaAgent'

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

        self.transition_buffer = []
        self.transition_buffer_size = 100

        self.policy_values = 'q'  # 'q' or 's' or 'qs'

        self._vf = {'q': dict(network=None,
                              layers_type=['fc', 'fc'],
                              layers_features=[2, 2],
                              action_layer_num=3,
                              # if one more than layer numbers => we will have num of actions output
                              batch_size=10,
                              step_size=params['max_stepsize'] / 10,
                              training=True),
                    's': dict(network=None,
                              layers_type=['fc'],
                              layers_features=[32],
                              batch_size=1,
                              step_size=0.01,
                              training=False)}

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

        self._target_vf = dict(network=None,
                               counter=0,
                               layers_num=None,
                               action_layer_num=None,
                               update_rate=10,
                               type=None)

        self.reward_function = params['reward_function']
        self.device = params['device']
        self.goal = torch.from_numpy(params['goal']).float().to(self.device)
        self.true_model = params['true_fw_model']
        self.model = {'forward': dict(network=params['model'],
                                      step_size=0.01,
                                      layers_type=['fc'],
                                      layers_features=[64],
                                      action_layer_number=2,
                                      batch_size=4,
                                      halluc_steps=2,
                                      training=True,
                                      plan_horizon=3,
                                      plan_buffer_size=1,
                                      plan_buffer=[])}


    def start(self, observation):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)

        if self._vf['q']['network'] is None and self._vf['q']['training']:
            self.init_q_value_function_network(self.prev_state)  # a general state action VF for all actions
        if self._vf['s']['network'] is None and self._vf['s']['training']:
            self.init_s_value_function_network(self.prev_state)  # a separate state VF for each action
        if self._target_vf['network'] is None:
            self.setTargetValueFunction(self._vf['q'], 'q')
        self.prev_action = self.policy(self.prev_state)

        return self.prev_action

    def step(self, reward, observation):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)

        reward = torch.tensor(reward).unsqueeze(0).to(self.device)
        self.action = self.policy(self.state)

        # store the new transition in buffer
        self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
                                                     self.state, self.action, False, self.time_step, 0))
        # update target
        if self._target_vf['counter'] >= self._target_vf['update_rate']:
            self.setTargetValueFunction(self._vf['q'], 'q')
            # self.setTargetValueFunction(self._vf['s'], 's')
            self._target_vf['counter'] = 0

        # update value function with the buffer
        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action  # another option:** we can again call self.policy function **

        return self.prev_action

    def end(self, reward):
        reward = torch.tensor(reward).unsqueeze(0).to(self.device)

        self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
                                                     None, None, True, self.time_step, 0))

        if self._vf['q']['training']:
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        if self._vf['s']['training']:
            if len(self.transition_buffer) >= self._vf['s']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 's')

        self.updateStateRepresentation()

    def policy(self, state):
        '''
        :param state: torch -> (1, state_shape)
        :return: action: numpy array
        '''
        with torch.no_grad():
            if np.random.rand() <= self.epsilon:
                ind = int(np.random.rand() * self.num_actions)
                return self.action_list[ind]
            return self.mcts(state)

    def updateNetworkWeights(self, network, step_size):
        # another option: ** can use a optimizer here later**
        optimizer = optim.SGD(network.parameters(), lr=step_size)
        optimizer.step()
        optimizer.zero_grad()

    def init_q_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = state.shape
        self._vf['q']['network'] = StateActionVFNN4(nn_state_shape, self.num_actions,
                                                    self._vf['q']['layers_type'],
                                                    self._vf['q']['layers_features'],
                                                    self._vf['q']['action_layer_num']).to(self.device)

    def init_s_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = state.shape
        self._vf['s']['network'] = []
        for i in range(self.num_actions):
            self._vf['s']['network'].append(StateVFNN(nn_state_shape,
                                                      self._vf['s']['layers_type'],
                                                      self._vf['s']['layers_features']).to(self.device))

    def init_s_representation_network(self, observation):
        '''
        :param observation: numpy array
        :return: None
        '''
        nn_state_shape = observation.shape
        self._sr['network'] = StateRepresentation(nn_state_shape,
                                                  self._sr['layers_type'],
                                                  self._sr['layers_features']).to(self.device)


    def updateValueFunction(self, transition_batch, vf_type):
        for i, data in enumerate(transition_batch):
            prev_state, prev_action, reward, state, action, _, t, error = data
            self.calculateGradientValueFunction(vf_type, reward, prev_state, prev_action, state, action)
        self.updateNetworkWeights(self._vf[vf_type]['network'], self._vf[vf_type]['step_size'] * np.exp(-error))
        self._target_vf['counter'] += 1

    def calculateGradientValueFunction(self, vf_type, reward, prev_state, prev_action, state=None, action=None):
        if prev_action is None:
            raise ValueError('previous action not given')
        prev_action_index = self.getActionIndex(prev_action)
        if vf_type == 'q':
            target = reward.float()
            if state is not None:  # Not a terminal State
                assert prev_state.shape == state.shape, 'x_old and x_new have different shapes'
                # target += self.gamma * self.getStateActionValue(state, action, gradient=False, type='q')
                target += self.gamma * max(self.getTargetValue(state, action) for action in self.action_list)
            input = self.getStateActionValue(prev_state, prev_action, vf_type='q', gradient=True)
            assert target.shape == input.shape, 'target and input must have same shapes'
            loss = nn.MSELoss()(input, target)
            loss.backward()

        if vf_type == 's':
            target = reward.float()
            if state is not None:  # Not a terminal State
                assert prev_state.shape == state.shape, 'x_old and x_new have different shapes'
                # target += self.gamma * self.getStateActionValue(state, action, gradient=False, type='s')
                target += self.gamma * self.getTargetValue(state, action)
            input = self.getStateActionValue(prev_state, prev_action, vf_type='s', gradient=True)
            assert target.shape == input.shape, 'target and input must have same shapes'
            loss = nn.MSELoss()(input, target)
            loss.backward()

    def getStateActionValue(self, state, action=None, vf_type='q', gradient=False):
        '''
        :param state: torch -> [1, state_shape]
        :param action: numpy array
        :param vf_type: str -> 'q' or 's'
        :param gradient: boolean
        :return: value: int
        '''
        if action is not None:
            action_index = self.getActionIndex(action)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

            if vf_type == 'q':
                if len(self._vf['q']['layers_type']) + 1 == self._vf['q']['action_layer_num']:
                    value = self._vf['q']['network'](state).detach()[:, action_index] if not gradient \
                        else self._vf['q']['network'](state)[:, action_index]
                else:
                    value = self._vf['q']['network'](state, action_onehot).detach()[0] if not gradient \
                        else self._vf['q']['network'](state, action_onehot)[0]

            elif vf_type == 's':
                value = self._vf['s']['network'][action_index](state).detach()[0] if not gradient \
                    else self._vf['s']['network'][action_index](state)[0]

            else:
                raise ValueError('state action value type is not defined')
            return value
        else:
            # state value (no gradient)
            if gradient:
                raise ValueError("cannot calculate the gradient for state values!")
            sum = 0
            for action in self.action_list:
                action_index = self.getActionIndex(action)
                action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

                if vf_type == 'q':
                    if len(self._vf['q']['layers_type']) + 1 == self._vf['q']['action_layer_num']:
                        value = self._vf['q']['network'](state).detach()[:, action_index]
                    else:
                        value = self._vf['q']['network'](state, action_onehot).detach()[0]

                elif vf_type == 's':
                    value = self._vf['s']['network'][action_index](state).detach()[0]

                else:
                    raise ValueError('state action value type is not defined')

                sum += value

            return sum / len(self.action_list)

    # ***
    def getStateRepresentation(self, observation, gradient=False):
        '''
        :param observation: numpy array -> [obs_shape]
        :param gradient: boolean
        :return: torch including batch -> [1, state_shape]
        '''
        if gradient:
            self._sr['batch_counter'] += 1
        observation = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        rep = self._sr['network'](observation).detach() if not gradient else self._sr['network'](observation)
        return rep

    def updateStateRepresentation(self):
        if len(self._sr['layers_type']) == 0:
            return None
        if self._sr['batch_counter'] == self._sr['batch_size'] and self._sr['training']:
            self.updateNetworkWeights(self._sr['network'], self._sr['step_size'] / self._sr['batch_size'])
            self._sr['batch_counter'] = 0

    # ***
    def setTargetValueFunction(self, vf, type):
        if self._target_vf['network'] is None:
            nn_state_shape = self.prev_state.shape
            self._target_vf['network'] = StateActionVFNN4(
                nn_state_shape,
                self.num_actions,
                vf['layers_type'],
                vf['layers_features'],
                vf['action_layer_num']).to(self.device)
        self._target_vf['network'].load_state_dict(vf['network'].state_dict())  # copy weights and stuff
        if type != 's':
            self._target_vf['action_layer_num'] = vf['action_layer_num']
        self._target_vf['layers_num'] = len(vf['layers_type'])
        self._target_vf['counter'] = 0
        self._target_vf['type'] = type

    def getTargetValue(self, state, action=None):
        '''
        :param state: torch -> (1, state_shape)
        :param action: numpy array
        :return value: int
        '''
        if action is not None:
            action_index = self.getActionIndex(action)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

            if self._target_vf['type'] == 'q':
                if self._target_vf['layers_num'] + 1 == self._target_vf['action_layer_num']:
                    value = self._target_vf['network'](state).detach()[:, action_index]
                else:
                    value = self._target_vf['network'](state, action_onehot).detach()[0]

            elif self._target_vf['type'] == 's':
                value = self._target_vf['network'][action_index](state).detach()[0]

            else:
                raise ValueError('state action value type is not defined')
            return value
        else:
            # state value (no gradient)
            sum = 0
            for action in self.action_list:
                action_index = self.getActionIndex(action)
                action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0).to(self.device)

                if self._target_vf['type'] == 'q':
                    if self._target_vf['layers_num'] + 1 == self._target_vf['action_layer_num']:
                        value = self._target_vf['network'](state).detach()[:, action_index]
                    else:
                        value = self._target_vf['network'](state, action_onehot).detach()[0]

                elif self._target_vf['type'] == 's':
                    value = self._target_vf['network'][action_index](state).detach()

                else:
                    raise ValueError('state action value type is not defined')

                sum += value

            return sum / len(self.action_list)

    def getTransitionFromBuffer(self, n):
        # both model and value function are using this buffer
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.choices(self.transition_buffer, k=n)

    def updateTransitionBuffer(self, transition):
        self.transition_buffer.append(transition)
        if len(self.transition_buffer) > self.transition_buffer_size:
            self.removeFromTransitionBuffer()

    def removeFromTransitionBuffer(self):
        self.transition_buffer.pop(0)

    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(self, action):
        res = np.zeros([len(self.action_list)])
        res[self.getActionIndex(action)] = 1
        return res

    def mcts(self, state):
        state = state[0].cpu().numpy()
        num_iteration = 10
        tree = Node(state, val=self.getStateValue(state))
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
        simulation_depth = 0
        reward_sum = 0
        state = node.state
        node_val = 0
        is_terminal = False
        for i in range(simulation_depth):
            rand = int(np.random.rand() * len(self.action_list))
            action = self.action_list[rand]
            child_state, is_terminal, reward = self.true_model(state, action)
            reward_sum += reward
            if is_terminal:
                break
            state = child_state
        if not is_terminal:
            node_val = self.getStateValue(state)
            # node_val = -np.abs(state[0] - self.goal[0]) - np.abs(state[1] - self.goal[1])
        return node_val + reward_sum

    def back_propagation(self, node, new_val):
        #average of children
        if node.back_prop_type == 0:
            while node is not None:
                sum_val = node.search_count * node.search_val + new_val
                node.search_count += 1
                node.search_val = sum_val / node.search_count
                new_val += node.from_par_reward
                node = node.par

        #max of children
        elif node.back_prop_type == 1:
            while node is not None:
                # if node.search_count == 0:
                #     node.search_val = -np.inf
                node.search_count += 1
                node.search_val = max(node.search_val, new_val)
                new_val = node.search_val + node.from_par_reward
                node = node.par

    def getStateValue(self, state):
        value = []
        torch_state = torch.from_numpy(np.asarray(state)).unsqueeze(0)
        for action in self.action_list:
            value.append(self.getStateActionValue(torch_state, action=action, vf_type='q'))
        return max(value)

    def expansion_policy(self, node, N):
        non_terminal_children = []
        for i in range(len(node.children)):
            child = node.children[i]
            if child.state is None:
                continue
            non_terminal_children.append(child)
        return np.random.choice(non_terminal_children)