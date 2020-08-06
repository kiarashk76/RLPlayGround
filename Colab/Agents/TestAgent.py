from typing import Dict, List, Any, Union

import torch
import torch.nn as nn
import numpy as np
import random

from Agents.ForwardDynaAgent import ForwardDynaAgent
from Agents.BackwardDynaAgent import BackwardDynaAgent
from Agents.BaseDynaAgent import BaseDynaAgent
from Networks.ModelNN.ModelError import ModelError


class TestAgent(ForwardDynaAgent):
    def __init__(self, params):
        ForwardDynaAgent.__init__(self, params)
        self.H_max = 5
        self.model_error_params = dict(step_size=0.01,
                                 layers_type=['fc'],
                                 layers_features=[64],
                                 action_layer_number=2,
                                 batch_size=4,
                                 training=True)

        self.model_error = [dict(network=None)] * (self.H_max + 1)

    def start(self, observation):
        prev_action = ForwardDynaAgent.start(self, observation)
        self.initModelError(self.prev_state)
        return prev_action

    def step(self, reward, observation):
        prev_action = ForwardDynaAgent.step(self, reward, observation)
        self.trainModelError()
        return prev_action

    def initModelError(self, state):
        for h, model_error in enumerate(self.model_error):
            if model_error['network'] is None and h != 0:
                model_error['network'] = \
                    ModelError(
                    state.shape,
                    len(self.action_list),
                    self.model_error_params['layers_type'],
                    self.model_error_params['layers_features'],
                    self.model_error_params['action_layer_number'])
                model_error['network'].to(self.device)

    def trainModelError(self):
        if len(self.transition_buffer) >= self.model_error_params['batch_size']:
            transition_batch = self.getTransitionFromBuffer(n=self.model_error_params['batch_size'])
            for transition in transition_batch:
                if self.model_error_params['training'] is True:
                    self._calculateModelErrorGradients(self.model_error, transition.prev_state,
                                             transition.prev_action, transition.state)
            for h in range(1, self.H_max + 1):
                self.updateNetworkWeights(self.model_error[h]['network'],
                                          self.model_error_params['step_size'])

    def _calculateModelErrorGradients(self, model_error, state, action, next_state):
        # state and next_state are tensor, action is numpy array
        if next_state is not None:
            # not a terminal state
            state = state.float().to(self.device)
            next_state = next_state.float().to(self.device)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
            action_index = self.getActionIndex(action)

            pred_next_state, _ = self.rolloutWithModel(state, action, self.model['forward'], h=1)
            w = torch.sum(torch.sub(next_state, pred_next_state) ** 2).unsqueeze(0).unsqueeze(0)
            for h in range(1, self.H_max + 1):
                # state transition training
                if len(self.model_error_params['layers_type']) + 1 == self.model_error_params['action_layer_number']:
                    input = model_error[h]['network'](state, action_onehot)[:, action_index]
                else:
                    input = model_error[h]['network'](state, action_onehot)
                target = self.gamma * self.getModelError(state, action, self.model_error, h-1) + w
                assert target.shape == input.shape, 'target and input must have same shapes'
                loss = nn.MSELoss()(input, target)
                loss.backward()

    def getModelError(self, state, action, model_error, h=1):
        '''
        :param state: torch -> (B, state_shape)
        :param action: numpy array
        :param model_error: model_error dictionary
        :param h: depth -> int
        :return: error
        '''
        if h == 0:
            return torch.tensor([0]).unsqueeze(0)

        current_state = torch.tensor(state.data.clone())
        current_action = np.copy(action)
        with torch.no_grad():
            action_onehot = torch.from_numpy(self.getActionOnehot(current_action)).unsqueeze(0).to(self.device)
            action_index = self.getActionIndex(current_action)
            if len(self.model_error_params['layers_type']) + 1 == self.model_error_params['action_layer_number']:
                pred_error = model_error[h]['network'](current_state, action_onehot).detach()[:, action_index]
            else:
                pred_error = model_error[h]['network'](current_state, action_onehot).detach()

        return pred_error
