# from typing import Dict, List, Any, Union
#
# import torch
# import torch.nn as nn
# import numpy as np
# import random
#
# from Colab.Agents.ForwardDynaAgent import ForwardDynaAgent
# from Colab.Agents.BackwardDynaAgent import BackwardDynaAgent
# from Colab.Agents.BaseDynaAgent import BaseDynaAgent
# from Colab.Networks.ModelNN.ModelError import ModelError
#
#
# class TestAgent(ForwardDynaAgent):
#     def __init__(self, params):
#         ForwardDynaAgent.__init__(self, params)
#         self.H_max = 5
#         self.model_error_params = dict(step_size=0.01,
#                                  layers_type=['fc'],
#                                  layers_features=[64],
#                                  action_layer_number=2,
#                                  batch_size=4,
#                                  training=True)
#
#         self.model_error = [dict(network=None)] * (self.H_max + 1)
#
#     def start(self, observation):
#         prev_action = ForwardDynaAgent.start(self, observation)
#         self.initModelError(self.prev_state)
#         return prev_action
#
#     def step(self, reward, observation):
#         prev_action = ForwardDynaAgent.step(self, reward, observation)
#         self.trainModelError()
#         return prev_action
#
#     def initModelError(self, state):
#         for h, model_error in enumerate(self.model_error):
#             if model_error['network'] is None and h != 0:
#                 model_error['network'] = \
#                     ModelError(
#                     state.shape,
#                     len(self.action_list),
#                     self.model_error_params['layers_type'],
#                     self.model_error_params['layers_features'],
#                     self.model_error_params['action_layer_number'])
#                 model_error['network'].to(self.device)
#
#     def trainModelError(self):
#         if len(self.transition_buffer) >= self.model_error_params['batch_size']:
#             transition_batch = self.getTransitionFromBuffer(n=self.model_error_params['batch_size'])
#             for transition in transition_batch:
#                 if self.model_error_params['training'] is True:
#                     self._calculateModelErrorGradients(self.model_error, transition.prev_state,
#                                              transition.prev_action, transition.state)
#             for h in range(1, self.H_max + 1):
#                 self.updateNetworkWeights(self.model_error[h]['network'],
#                                           self.model_error_params['step_size'])
#
#     def _calculateModelErrorGradients(self, model_error, state, action, next_state):
#         # state and next_state are tensor, action is numpy array
#         if next_state is not None:
#             # not a terminal state
#             state = state.float().to(self.device)
#             next_state = next_state.float().to(self.device)
#             action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
#             action_index = self.getActionIndex(action)
#
#             pred_next_state, _ = self.rolloutWithModel(state, action, self.model['forward'], h=1)
#             w = torch.sum(torch.sub(next_state, pred_next_state) ** 2).unsqueeze(0).unsqueeze(0)
#             for h in range(1, self.H_max + 1):
#                 # state transition training
#                 if len(self.model_error_params['layers_type']) + 1 == self.model_error_params['action_layer_number']:
#                     input = model_error[h]['network'](state, action_onehot)[:, action_index]
#                 else:
#                     input = model_error[h]['network'](state, action_onehot)
#                 target = self.gamma * self.getModelError(state, action, self.model_error, h-1) + w
#                 assert target.shape == input.shape, 'target and input must have same shapes'
#                 loss = nn.MSELoss()(input, target)
#                 loss.backward()
#
#     def getModelError(self, state, action, model_error, h=1):
#         '''
#         :param state: torch -> (B, state_shape)
#         :param action: numpy array
#         :param model_error: model_error dictionary
#         :param h: depth -> int
#         :return: error
#         '''
#         if h == 0:
#             return torch.tensor([0]).unsqueeze(0)
#
#         current_state = torch.tensor(state.data.clone())
#         current_action = np.copy(action)
#         with torch.no_grad():
#             action_onehot = torch.from_numpy(self.getActionOnehot(current_action)).unsqueeze(0).to(self.device)
#             action_index = self.getActionIndex(current_action)
#             if len(self.model_error_params['layers_type']) + 1 == self.model_error_params['action_layer_number']:
#                 pred_error = model_error[h]['network'](current_state, action_onehot).detach()[:, action_index]
#             else:
#                 pred_error = model_error[h]['network'](current_state, action_onehot).detach()
#
#         return pred_error
#
#     def plan(self):
#         pass


import torch
import torch.nn as nn
import numpy as np
import random

from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.Networks.ModelNN.StateTransitionModel as STModel
from Colab.Networks.ModelNN.ModelError import ModelError
import Colab.utils as utils




class BackwardDynaAgent2(BaseDynaAgent):
    name = 'BackwardDynaAgent'

    def __init__(self, params):
        super(BackwardDynaAgent2, self).__init__(params)

        self.model_batch_counter = 0
        self.model = {'backward': dict(network=params['model'],
                                       step_size=params['model_stepsize'],
                                       layers_type=['fc'],
                                       layers_features=[32],
                                       action_layer_number=2,
                                       batch_size=4,
                                       batch_counter=None,
                                       training=True,
                                       halluc_steps=2,
                                       plan_number=3,
                                       plan_horizon=1,
                                       plan_buffer_size=10,
                                       plan_buffer=[])}
        self.true_model = params['true_bw_model']
        self.error_network = dict(network=params['model'],
                                       step_size=0.01,
                                       layers_type=['fc'],
                                       layers_features=[32],
                                       action_layer_number=2,
                                       batch_size=4,
                                       batch_counter=None,
                                       training=True,
                                       halluc_steps=2)

        self.planning_transition_buffer = []
        self.planning_transition_buffer_size = 10


    def initModel(self, state):
        if self.model['backward']['network'] is not None:
            return
        self.model['backward']['network'] = \
            STModel.StateTransitionModel(
                state.shape,
                len(self.action_list),
                self.model['backward']['layers_type'],
                self.model['backward']['layers_features'],
                self.model['backward']['action_layer_number'])
        self.model['backward']['network'].to(self.device)

        if self.error_network['network'] is not None:
            return
        self.error_network['network'] = \
            ModelError(
                state.shape,
                len(self.action_list),
                self.error_network['layers_type'],
                self.error_network['layers_features'],
                self.error_network['action_layer_number'])
        self.error_network['network'].to(self.device)


    def trainModel(self, terminal=False):
        if len(self.transition_buffer) < self.model['backward']['batch_size']:
            return
        transition_batch = self.getTransitionFromBuffer(n=self.model['backward']['batch_size'])
        for i, data in enumerate(transition_batch):
            prev_state, prev_action, reward, state, action, terminal, t, _ = data
            if self.model['backward']['training'] is True:
                self._calculateGradients(self.model['backward'], prev_state, prev_action, state, terminal=terminal)
            if self.error_network['training']:
                self._calculateErrorGradients(self.error_network, prev_state, prev_action, state)
        step_size = self.model['backward']['step_size'] / len(transition_batch)
        self.updateNetworkWeights(self.model['backward']['network'], step_size)

    def plan(self):
        if self._vf['q']['training']:
            if len(self.planning_transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromPlanningBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')

        with torch.no_grad():
            self.updatePlanningBuffer(self.model['backward'], self.state)
            for state in self.getStateFromPlanningBuffer(self.model['backward']):
                action = self.policy(state)
                for j in range(self.model['backward']['plan_horizon']):
                    prev_action = self.backwardRolloutPolicy(state)
                    prev_state = self.rolloutWithModel(state, prev_action, self.model['backward'])
                    reward = -1
                    terminal = self.isTerminal(state)
                    if terminal:
                        reward = 10
                    reward = torch.tensor(reward).unsqueeze(0).to(self.device)
                    x_old = prev_state.float().to(self.device)
                    x_new = state.float().to(self.device) if not terminal else None

                    error = 0
                    if self.is_using_error :
                        error = self.calculateTrueError(state, prev_action)
                    self.update_planning_transition_buffer(utils.transition(x_old, prev_action, reward,
                                                         x_new, action, terminal, self.time_step, error))
                    action = prev_action
                    state = prev_state

    def rolloutWithModel(self, state, action, model, h=1):
        current_state = torch.tensor(state.data.clone())
        current_action = np.copy(action)
        with torch.no_grad():
            for i in range(h):
                action_onehot = torch.from_numpy(self.getActionOnehot(current_action)).unsqueeze(0).to(self.device)
                action_index = self.getActionIndex(current_action)
                if len(model['layers_type']) + 1 == model['action_layer_number']:
                    prev_state = model['network'](current_state, action_onehot)[0][:, action_index]
                else:
                    prev_state = model['network'](current_state, action_onehot)[0]
                current_action = self.backwardRolloutPolicy(prev_state)
                current_state = prev_state
        return prev_state

    def calculateError(self, state, action, error_network):
        with torch.no_grad():
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
            action_index = self.getActionIndex(action)
            if len(error_network['layers_type']) + 1 == error_network['action_layer_number']:
                error = error_network['network'](state, action_onehot)[:, action_index]
            else:
                error = error_network['network'](state, action_onehot)

        return error

    def calculateTrueError(self, state, action, h=1):
        pred_prev_state = self.rolloutWithModel(state, action, self.model['backward'], h=h)
        state = state.cpu().numpy()[0]
        for i in range(h):
            prev_state = self.true_model(state, action)
            state = prev_state
            action = self.backwardRolloutPolicy(state)
        true_prev_state = torch.from_numpy(prev_state).unsqueeze(0)
        error = torch.mean((true_prev_state - pred_prev_state) ** 2)
        return error

    def backwardRolloutPolicy(self, state):
        # with torch.no_grad:
        random_action = self.action_list[int(np.random.rand() * self.num_actions)]
        return random_action

    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training
        if next_state is None:
            return
        state = state.float()
        next_state = next_state.float()
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)

        if len(model['layers_type']) + 1 == model['action_layer_number']:
            input = model['network'](next_state, action_onehot)[0][:, action_index]
        else:
            input = model['network'](next_state, action_onehot)[0]
        target = state

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

    def _calculateErrorGradients(self, error_network, state, action, next_state):
        if next_state is None:
            return
        state = state.float()
        next_state = next_state.float()
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)

        if len(error_network['layers_type']) + 1 == error_network['action_layer_number']:
            input = error_network['network'](next_state, action_onehot)[:, action_index]
        else:
            input = error_network['network'](next_state, action_onehot)
        pred_state = self.rolloutWithModel(next_state, action, self.model['backward'])
        target = torch.mean((state - pred_state) ** 2).unsqueeze(0).unsqueeze(0)
        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()


    def updatePlanningBuffer(self, model, state):
        model['plan_buffer'].append(state)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        number_of_samples = min(model['plan_number'], len(model['plan_buffer']))
        return random.choices(model['plan_buffer'], k=number_of_samples)

    def isTerminal(self, state):
        diff = np.abs(np.multiply(self.goal, state.cpu().numpy())).sum()
        return diff <= 2

    def getTransitionFromPlanningBuffer(self, n):
        if len(self.planning_transition_buffer) < n:
            n = len(self.planning_transition_buffer)
        return random.choices(self.planning_transition_buffer, k=n)

    def update_planning_transition_buffer(self, transition):
        self.planning_transition_buffer.append(transition)
        if len(self.planning_transition_buffer) > self.planning_transition_buffer_size:
            self.removeFromPlanningTransitionBuffer()

    def removeFromPlanningTransitionBuffer(self):
        self.planning_transition_buffer.pop(0)



class ForwardDynaAgent2(BaseDynaAgent):
    name = 'ForwardDynaAgent'

    def __init__(self, params):
        super(ForwardDynaAgent, self).__init__(params)

        self.model = {'forward': dict(network=params['model'],
                                      step_size=params['model_stepsize'],
                                      layers_type=['fc'],
                                      layers_features=[32],
                                      action_layer_number=2,
                                      batch_size=4,
                                      halluc_steps=2,
                                      training=True,
                                      plan_number=3,
                                      plan_horizon=1,
                                      plan_buffer_size=10,
                                      plan_buffer=[])}
        self.true_model = params['true_fw_model']
    def initModel(self, state):
        if self.model['forward']['network'] is None:
            self.model['forward']['network'] = \
                STModel.StateTransitionModel(
                state.shape,
                len(self.action_list),
                self.model['forward']['layers_type'],
                self.model['forward']['layers_features'],
                self.model['forward']['action_layer_number'])
            self.model['forward']['network'].to(self.device)

        self.planning_transition_buffer = []
        self.planning_transition_buffer_size = 10
    def trainModel(self):
        if len(self.transition_buffer) < self.model['forward']['batch_size']:
            return
        transition_batch = self.getTransitionFromBuffer(n=self.model['forward']['batch_size'])
        for transition in transition_batch:
            if self.model['forward']['training'] is True:
                self._calculateGradients(self.model['forward'], transition.prev_state,
                                         transition.prev_action, transition.state, terminal=transition.is_terminal)
        step_size = self.model['forward']['step_size'] / len(transition_batch)
        self.updateNetworkWeights(self.model['forward']['network'], step_size)

    def plan(self):
        if self._vf['q']['training']:
            if len(self.planning_transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromPlanningBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')

        with torch.no_grad():
            self.updatePlanningBuffer(self.model['forward'], self.state)
            for state in self.getStateFromPlanningBuffer(self.model['forward']):
                for j in range(self.model['forward']['plan_horizon']):
                    action = self.forwardRolloutPolicy(state)
                    next_state = self.rolloutWithModel(state, action, self.model['forward'])
                    next_action = self.forwardRolloutPolicy(next_state)
                    reward = -1
                    terminal = self.isTerminal(next_state)
                    if terminal:
                        reward = 10
                    reward = torch.tensor(reward).unsqueeze(0).to(self.device)
                    x_old = state.float().to(self.device)
                    x_new = next_state.float().to(self.device) if not terminal else None

                    error = 0
                    # if self.is_using_error:
                    #     error = self.calculateTrueError(state, prev_action)
                    self.update_planning_transition_buffer(utils.transition(x_old, action, reward,
                                                                            x_new, next_action, terminal, self.time_step,
                                                                            error))
                    action = next_action
                    state = next_state



            # current_state = torch.tensor(self.prev_state.data.clone())
            # true_current_state = self.prev_state.cpu().numpy()[0]
            # current_action = np.copy(self.prev_action)
            # for h in range(self.model['forward']['plan_horizon']):
            #     next_state, is_terminal = self.rolloutWithModel(current_state, current_action, self.model['forward'], 1)
            #     is_terminal = np.random.binomial(n=1, p=float(is_terminal.data.cpu().numpy()), size=1)
            #     true_next_state, _, reward = self.true_model(true_current_state, current_action)
            #     reward = torch.tensor(reward).unsqueeze(0).to(self.device)
            #     next_action = self.forwardRolloutPolicy(next_state)
            #
            #     if is_terminal:
            #         self.updateTransitionBuffer(utils.transition(current_state, current_action, reward,
            #                                                  None, None, True, self.time_step))
            #     else:
            #         self.updateTransitionBuffer(utils.transition(current_state, current_action, reward,
            #                                                      next_state, next_action, False, self.time_step))
            #     current_state = next_state
            #     current_action = next_action
            #     true_current_state = true_next_state

    def rolloutWithModel(self, state, action, model, h=1):
        '''
        :param state: torch -> (B, state_shape)
        :param action: numpy array
        :param model: model dictionary
        :param rollout_policy: function
        :param h: int
        :return: next state and is_terminal
        '''
        current_state = torch.tensor(state.data.clone())
        current_action = np.copy(action)
        with torch.no_grad():
            for i in range(h):
                action_onehot = torch.from_numpy(self.getActionOnehot(current_action)).unsqueeze(0).to(self.device)
                action_index = self.getActionIndex(current_action)
                if len(model['layers_type']) + 1 == model['action_layer_number']:
                    pred_state = model['network'](current_state, action_onehot)[0].detach()[:,action_index]
                    is_terminal = model['network'](current_state, action_onehot)[1].detach()[:,action_index]

                else:
                    pred_state = model['network'](current_state, action_onehot)[0].detach()
                    is_terminal = model['network'](current_state, action_onehot)[1].detach()

                current_action = self.forwardRolloutPolicy(pred_state)
                current_state = pred_state
        return pred_state#, is_terminal

    def forwardRolloutPolicy(self, state):
        return self.policy(state)

    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training
        # state and next_state are tensor, action is numpy array
        if next_state is not None:
            # not a terminal state
            state = state.float().to(self.device)
            next_state = next_state.float().to(self.device)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
            action_index = self.getActionIndex(action)

            #state transition training
            if len(model['layers_type']) + 1 == model['action_layer_number']:
                input = model['network'](state, action_onehot)[0][:, action_index]
            else:
                input = model['network'](state, action_onehot)[0]
            target = next_state

            assert target.shape == input.shape, 'target and input must have same shapes'
            loss = nn.MSELoss()(input, target)
            loss.backward()

            #terminal training
            if len(model['layers_type']) + 1 == model['action_layer_number']:
                input = model['network'](state, action_onehot)[1][:, action_index]
            else:
                input = model['network'](state, action_onehot)[1]
            target = torch.tensor([0.0]).float().unsqueeze(0).to(self.device)
            assert target.shape == input.shape, 'target and input must have same shapes'
            loss = nn.MSELoss()(input, target)
            loss.backward()
        else:
            #no transition training needed
            state = state.float().to(self.device)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
            action_index = self.getActionIndex(action)

            # terminal training
            if len(model['layers_type']) + 1 == model['action_layer_number']:
                input = model['network'](state, action_onehot)[1][:, action_index]
            else:
                input = model['network'](state, action_onehot)[1]
            target = torch.tensor([1.0]).float().unsqueeze(0).to(self.device)
            assert target.shape == input.shape, 'target and input must have same shapes'
            loss = nn.MSELoss()(input, target)
            loss.backward()


    def updatePlanningBuffer(self, model, state):
        model['plan_buffer'].append(state)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        return random.choices(model['plan_buffer'], k=1)

    def isTerminal(self, state):
        diff = np.abs(np.multiply(self.goal, state.cpu().numpy())).sum()
        return diff <= 2



    def getTransitionFromPlanningBuffer(self, n):
        if len(self.planning_transition_buffer) < n:
            n = len(self.planning_transition_buffer)
        return random.choices(self.planning_transition_buffer, k=n)

    def update_planning_transition_buffer(self, transition):
        self.planning_transition_buffer.append(transition)
        if len(self.planning_transition_buffer) > self.planning_transition_buffer_size:
            self.removeFromPlanningTransitionBuffer()

    def removeFromPlanningTransitionBuffer(self):
        self.planning_transition_buffer.pop(0)
















class BackwardDynaAgent(BaseDynaAgent):
    name = 'BackwardDynaAgent'

    def __init__(self, params):
        super(BackwardDynaAgent, self).__init__(params)

        self.model_batch_counter = 0
        self.model = {'backward': dict(network=None,
                                       step_size=params['model_stepsize'],
                                       layers_type=['fc'],
                                       layers_features=[32],
                                       action_layer_number=2,
                                       batch_size=4,
                                       batch_counter = [],
                                       training=True,
                                       halluc_steps=2,
                                       plan_number=3,
                                       plan_horizon=1,
                                       plan_buffer_size=10,
                                       plan_buffer=[])}
        self.num_models = 3
        self.counter = {}

        self.true_model = params['true_bw_model']
        self.error_network = dict(network=params['model'],
                                       step_size=0.01,
                                       layers_type=['fc'],
                                       layers_features=[32],
                                       action_layer_number=2,
                                       batch_size=4,
                                       batch_counter=None,
                                       training=True,
                                       halluc_steps=2)

        self.planning_transition_buffer = []
        self.planning_transition_buffer_size = 10


    def initModel(self, state):
        if self.model['backward']['network'] is not None:
            return
        self.model['backward']['network'] = []
        for i in range(self.num_models):
            net = STModel.StateTransitionModel(
                    state.shape,
                    len(self.action_list),
                    self.model['backward']['layers_type'],
                    self.model['backward']['layers_features'],
                    self.model['backward']['action_layer_number']).to(self.device)
            self.model['backward']['network'].append(net)
            self.model['backward']['batch_counter'].append(0)


        # if self.error_network['network'] is not None:
        #     return
        # self.error_network['network'] = \
        #     ModelError(
        #         state.shape,
        #         len(self.action_list),
        #         self.error_network['layers_type'],
        #         self.error_network['layers_features'],
        #         self.error_network['action_layer_number'])
        # self.error_network['network'].to(self.device)


    def trainModel(self, terminal=False):
        if len(self.transition_buffer) < self.model['backward']['batch_size']:
            return
        transition_batch = self.getTransitionFromBuffer(n=self.model['backward']['batch_size'])
        for i, data in enumerate(transition_batch):
            prev_state, prev_action, reward, state, action, terminal, t, _ = data
            pred_y = []
            if state is None:
                continue
            for i in range(self.num_models):
                    pred_y.append( torch.dist(
                        self.rolloutWithModel(prev_state, prev_action, self.model['backward'], i),
                        state.float()) )
            # model_index = np.argmin(pred_y)
            pos = tuple(prev_state.tolist()[0])
            net_index = pred_y.index(min(pred_y))
            if ((pos), tuple(prev_action)) not in self.counter:
                self.counter[(pos), tuple(prev_action)] = [0, 0, 0]
            self.counter[(pos), tuple(prev_action)][net_index] += 1
            if self.model['backward']['training'] is True:
                self._calculateGradients(self.model['backward'], prev_state, prev_action, state, net_index, terminal=terminal)
                self.model['backward']['batch_counter'][net_index] += 1
            # if self.error_network['training']:
            #     self._calculateErrorGradients(self.error_network, prev_state, prev_action, state)

        for i in range(self.num_models):
            if self.model['backward']['batch_counter'][i] >= self.model['backward']['batch_size']:
                step_size = self.model['backward']['step_size'] / self.model['backward']['batch_counter'][i]
                self.updateNetworkWeights(self.model['backward']['network'][i], step_size)
                self.model['backward']['batch_counter'][i] = 0

    def plan(self):
        return 0
        if self._vf['q']['training']:
            if len(self.planning_transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromPlanningBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')

        with torch.no_grad():
            self.updatePlanningBuffer(self.model['backward'], self.state)
            for state in self.getStateFromPlanningBuffer(self.model['backward']):
                action = self.policy(state)
                for j in range(self.model['backward']['plan_horizon']):
                    prev_action = self.backwardRolloutPolicy(state)
                    prev_state = self.rolloutWithModel(state, prev_action, self.model['backward'])
                    reward = -1
                    terminal = self.isTerminal(state)
                    if terminal:
                        reward = 10
                    reward = torch.tensor(reward).unsqueeze(0).to(self.device)
                    x_old = prev_state.float().to(self.device)
                    x_new = state.float().to(self.device) if not terminal else None

                    error = 0
                    if self.is_using_error :
                        error = self.calculateTrueError(state, prev_action)
                    self.update_planning_transition_buffer(utils.transition(x_old, prev_action, reward,
                                                         x_new, action, terminal, self.time_step, error))
                    action = prev_action
                    state = prev_state

    def rolloutWithModel(self, state, action, model, net_index=0, h=1):
        current_state = torch.tensor(state.data.clone()).float()
        current_action = np.copy(action)
        with torch.no_grad():
            for i in range(h):
                action_onehot = torch.from_numpy(self.getActionOnehot(current_action)).unsqueeze(0).to(self.device)
                action_index = self.getActionIndex(current_action)
                if len(model['layers_type']) + 1 == model['action_layer_number']:
                    prev_state = model['network'][net_index](current_state, action_onehot)[0][:, action_index]
                else:
                    prev_state = model['network'][net_index](current_state, action_onehot)[0]
                current_action = self.backwardRolloutPolicy(prev_state)
                current_state = prev_state
        return prev_state

    def calculateError(self, state, action, error_network):
        with torch.no_grad():
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
            action_index = self.getActionIndex(action)
            if len(error_network['layers_type']) + 1 == error_network['action_layer_number']:
                error = error_network['network'](state, action_onehot)[:, action_index]
            else:
                error = error_network['network'](state, action_onehot)

        return error

    def calculateTrueError(self, state, action, h=1):
        pred_prev_state = self.rolloutWithModel(state, action, self.model['backward'], h=h)
        state = state.cpu().numpy()[0]
        for i in range(h):
            prev_state = self.true_model(state, action)
            state = prev_state
            action = self.backwardRolloutPolicy(state)
        true_prev_state = torch.from_numpy(prev_state).unsqueeze(0)
        error = torch.mean((true_prev_state - pred_prev_state) ** 2)
        return error

    def backwardRolloutPolicy(self, state):
        # with torch.no_grad:
        random_action = self.action_list[int(np.random.rand() * self.num_actions)]
        return random_action

    def _calculateGradients(self, model, state, action, next_state, net_index, h=0, terminal=False):
        # todo: add hallucination training
        if next_state is None:
            return
        state = state.float()
        next_state = next_state.float()
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)

        if len(model['layers_type']) + 1 == model['action_layer_number']:
            input = model['network'][net_index](next_state, action_onehot)[0][:, action_index]
        else:
            input = model['network'][net_index](next_state, action_onehot)[0]
        target = state

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

    def _calculateErrorGradients(self, error_network, state, action, next_state):
        if next_state is None:
            return
        state = state.float()
        next_state = next_state.float()
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)

        if len(error_network['layers_type']) + 1 == error_network['action_layer_number']:
            input = error_network['network'](next_state, action_onehot)[:, action_index]
        else:
            input = error_network['network'](next_state, action_onehot)
        pred_state = self.rolloutWithModel(next_state, action, self.model['backward'])
        target = torch.mean((state - pred_state) ** 2).unsqueeze(0).unsqueeze(0)
        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()


    def updatePlanningBuffer(self, model, state):
        model['plan_buffer'].append(state)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        number_of_samples = min(model['plan_number'], len(model['plan_buffer']))
        return random.choices(model['plan_buffer'], k=number_of_samples)

    def isTerminal(self, state):
        diff = np.abs(np.multiply(self.goal, state.cpu().numpy())).sum()
        return diff <= 2

    def getTransitionFromPlanningBuffer(self, n):
        if len(self.planning_transition_buffer) < n:
            n = len(self.planning_transition_buffer)
        return random.choices(self.planning_transition_buffer, k=n)

    def update_planning_transition_buffer(self, transition):
        self.planning_transition_buffer.append(transition)
        if len(self.planning_transition_buffer) > self.planning_transition_buffer_size:
            self.removeFromPlanningTransitionBuffer()

    def removeFromPlanningTransitionBuffer(self):
        self.planning_transition_buffer.pop(0)

class ForwardDynaAgent(BaseDynaAgent):
    name = 'ForwardDynaAgent'

    def __init__(self, params):
        super(ForwardDynaAgent, self).__init__(params)

        self.model_batch_counter = 0
        self.model = {'forward': dict(network=None,
                                       num_networks=3,
                                       step_size=params['model_stepsize'],
                                       layers_type=['fc'],
                                       layers_features=[32],
                                       action_layer_number=2,
                                       batch_size=4,
                                       batch_counter = [],
                                       training=True,
                                       halluc_steps=2,
                                       plan_number=3,
                                       plan_horizon=1,
                                       plan_buffer_size=10,
                                       plan_buffer=[])}
        self.counter = {}

        self.true_model = params['true_fw_model']
        # self.error_network = dict(network=params['model'],
        #                                step_size=0.01,
        #                                layers_type=['fc'],
        #                                layers_features=[32],
        #                                action_layer_number=2,
        #                                batch_size=4,
        #                                batch_counter=None,
        #                                training=True,
        #                                halluc_steps=2)

        self.planning_transition_buffer = []
        self.planning_transition_buffer_size = 10


    def initModel(self, state):
        if self.model['forward']['network'] is not None:
            return
        self.model['forward']['network'] = []
        for i in range(self.model['forward']['num_networks']):
            net = STModel.StateTransitionModel(
                    state.shape,
                    len(self.action_list),
                    self.model['forward']['layers_type'],
                    self.model['forward']['layers_features'],
                    self.model['forward']['action_layer_number']).to(self.device)
            self.model['forward']['network'].append(net)
            self.model['forward']['batch_counter'].append(0)


        # if self.error_network['network'] is not None:
        #     return
        # self.error_network['network'] = \
        #     ModelError(
        #         state.shape,
        #         len(self.action_list),
        #         self.error_network['layers_type'],
        #         self.error_network['layers_features'],
        #         self.error_network['action_layer_number'])
        # self.error_network['network'].to(self.device)


    def trainModel(self, terminal=False):
        if len(self.transition_buffer) < self.model['forward']['batch_size']:
            return
        transition_batch = self.getTransitionFromBuffer(n=self.model['forward']['batch_size'])
        for i, data in enumerate(transition_batch):
            prev_state, prev_action, reward, state, action, terminal, t, _ = data
            pred_y = []
            if state is None:
                continue
            for i in range(self.model['forward']['num_networks']):
                    pred_y.append( torch.dist(
                        self.rolloutWithModel(prev_state, prev_action, self.model['forward'], i),
                        state.float()) )
            # model_index = np.argmin(pred_y)
            pos = tuple(prev_state.tolist()[0])
            net_index = pred_y.index(min(pred_y))
            if ((pos), tuple(prev_action)) not in self.counter:
                self.counter[(pos), tuple(prev_action)] = self.model['forward']['num_networks'] * [0]
            self.counter[(pos), tuple(prev_action)][net_index] += 1
            if self.model['forward']['training'] is True:
                self._calculateGradients(self.model['forward'], prev_state, prev_action, state, net_index, terminal=terminal)
                self.model['forward']['batch_counter'][net_index] += 1
            # if self.error_network['training']:
            #     self._calculateErrorGradients(self.error_network, prev_state, prev_action, state)

        for i in range(self.model['forward']['num_networks']):
            if self.model['forward']['batch_counter'][i] >= self.model['forward']['batch_size']:
                step_size = self.model['forward']['step_size'] / self.model['forward']['batch_counter'][i]
                self.updateNetworkWeights(self.model['forward']['network'][i], step_size)
                self.model['forward']['batch_counter'][i] = 0

    def plan(self):
        return 0
        if self._vf['q']['training']:
            if len(self.planning_transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromPlanningBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')

        with torch.no_grad():
            self.updatePlanningBuffer(self.model['backward'], self.state)
            for state in self.getStateFromPlanningBuffer(self.model['backward']):
                action = self.policy(state)
                for j in range(self.model['backward']['plan_horizon']):
                    prev_action = self.backwardRolloutPolicy(state)
                    prev_state = self.rolloutWithModel(state, prev_action, self.model['backward'])
                    reward = -1
                    terminal = self.isTerminal(state)
                    if terminal:
                        reward = 10
                    reward = torch.tensor(reward).unsqueeze(0).to(self.device)
                    x_old = prev_state.float().to(self.device)
                    x_new = state.float().to(self.device) if not terminal else None

                    error = 0
                    if self.is_using_error :
                        error = self.calculateTrueError(state, prev_action)
                    self.update_planning_transition_buffer(utils.transition(x_old, prev_action, reward,
                                                         x_new, action, terminal, self.time_step, error))
                    action = prev_action
                    state = prev_state

    def rolloutWithModel(self, state, action, model, net_index=None, h=1):
        if net_index is not None:
            current_state = torch.tensor(state.data.clone()).float()
            current_action = np.copy(action)
            with torch.no_grad():
                for i in range(h):
                    action_onehot = torch.from_numpy(self.getActionOnehot(current_action)).unsqueeze(0).to(self.device)
                    action_index = self.getActionIndex(current_action)
                    if len(model['layers_type']) + 1 == model['action_layer_number']:
                        next_state = model['network'][net_index](current_state, action_onehot)[0][:, action_index]
                    else:
                        next_state = model['network'][net_index](current_state, action_onehot)[0]
                    current_action = self.forwardRolloutPolicy(next_state)
                    current_state = next_state
            return next_state
        else:
            raise ValueError('net index not defined')

    def calculateError(self, state, action, error_network):
        pass
        # with torch.no_grad():
        #     action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        #     action_index = self.getActionIndex(action)
        #     if len(error_network['layers_type']) + 1 == error_network['action_layer_number']:
        #         error = error_network['network'](state, action_onehot)[:, action_index]
        #     else:
        #         error = error_network['network'](state, action_onehot)
        #
        # return error

    def calculateTrueError(self, state, action, h=1):
        pass
        # pred_prev_state = self.rolloutWithModel(state, action, self.model['backward'], h=h)
        # state = state.cpu().numpy()[0]
        # for i in range(h):
        #     prev_state = self.true_model(state, action)
        #     state = prev_state
        #     action = self.backwardRolloutPolicy(state)
        # true_prev_state = torch.from_numpy(prev_state).unsqueeze(0)
        # error = torch.mean((true_prev_state - pred_prev_state) ** 2)
        # return error

    def forwardRolloutPolicy(self, state):
        return self.policy(state)

    def _calculateGradients(self, model, state, action, next_state, net_index, h=0, terminal=False):
        # todo: add hallucination training
        if next_state is None:
            return
        state = state.float()
        next_state = next_state.float()
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)

        if len(model['layers_type']) + 1 == model['action_layer_number']:
            input = model['network'][net_index](state, action_onehot)[0][:, action_index]
        else:
            input = model['network'][net_index](state, action_onehot)[0]
        target = next_state

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

    def _calculateErrorGradients(self, error_network, state, action, next_state):
        pass
        # if next_state is None:
        #     return
        # state = state.float()
        # next_state = next_state.float()
        # action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        # action_index = self.getActionIndex(action)
        #
        # if len(error_network['layers_type']) + 1 == error_network['action_layer_number']:
        #     input = error_network['network'](next_state, action_onehot)[:, action_index]
        # else:
        #     input = error_network['network'](next_state, action_onehot)
        # pred_state = self.rolloutWithModel(next_state, action, self.model['backward'])
        # target = torch.mean((state - pred_state) ** 2).unsqueeze(0).unsqueeze(0)
        # assert target.shape == input.shape, 'target and input must have same shapes'
        # loss = nn.MSELoss()(input, target)
        # loss.backward()


    def updatePlanningBuffer(self, model, state):
        model['plan_buffer'].append(state)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        number_of_samples = min(model['plan_number'], len(model['plan_buffer']))
        return random.choices(model['plan_buffer'], k=number_of_samples)

    # def isTerminal(self, state):
    #     diff = np.abs(np.multiply(self.goal, state.cpu().numpy())).sum()
    #     return diff <= 2

    def getTransitionFromPlanningBuffer(self, n):
        if len(self.planning_transition_buffer) < n:
            n = len(self.planning_transition_buffer)
        return random.choices(self.planning_transition_buffer, k=n)

    def update_planning_transition_buffer(self, transition):
        self.planning_transition_buffer.append(transition)
        if len(self.planning_transition_buffer) > self.planning_transition_buffer_size:
            self.removeFromPlanningTransitionBuffer()

    def removeFromPlanningTransitionBuffer(self):
        self.planning_transition_buffer.pop(0)
