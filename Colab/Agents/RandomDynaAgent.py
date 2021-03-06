import torch
import torch.nn as nn
import numpy as np
import random

import Colab.utils as utils
from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.Networks.ModelNN.StateTransitionModel as STModel

class RandomDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(RandomDynaAgent, self).__init__(params)
        self.epsilon = 1
        self.model = {'forward': dict(network=params['model'],
                                      step_size=0.1,
                                      layers_type=['fc', 'fc'],
                                      layers_features=[256, 64],
                                      action_layer_number=3,
                                      batch_size=10,
                                      halluc_steps=2,
                                      training=True,
                                      plan_steps=0,
                                      plan_horizon=5,
                                      plan_buffer_size=5,
                                      plan_buffer=[]),
                      'backward': dict(network=params['model'],
                                      step_size=0.1,
                                      layers_type=['fc', 'fc'],
                                      layers_features=[256, 64],
                                      action_layer_number=3,
                                      batch_size=10,
                                      halluc_steps=2,
                                      training=True,
                                      plan_steps=0,
                                      plan_horizon=5,
                                      plan_buffer_size=5,
                                      plan_buffer=[])}
    def initModel(self):

        if self.model['forward']['network'] is None:
            self.model['forward']['network'] = \
                STModel.StateTransitionModel(
                self.prev_state.shape,
                len(self.action_list),
                self.model['forward']['layers_type'],
                self.model['forward']['layers_features'],
                self.model['forward']['action_layer_number'])
            self.model['forward']['network'].to(self.device)

        if self.model['backward']['network'] is None:
            self.model['backward']['network'] = \
                STModel.StateTransitionModel(
                self.prev_state.shape,
                len(self.action_list),
                self.model['backward']['layers_type'],
                self.model['backward']['layers_features'],
                self.model['backward']['action_layer_number'])
            self.model['backward']['network'].to(self.device)

    def trainModel(self, terminal=False):

        sample_transitions = self.getTransitionFromBuffer(n=self.model['forward']['batch_size'])

        for sample in sample_transitions:
            state, action, reward, next_state, _ = sample
            if self.model['backward']['training']:
                self._calculateGradients(self.model['backward'], state, action, next_state, terminal=terminal, type='backward')
            if self.model['forward']['training']:
                self._calculateGradients(self.model['forward'], state, action, next_state, terminal=terminal, type='forward')

        if self.model['backward']['training']:
            step_size = self.model['backward']['step_size'] / len(sample_transitions)
            self.updateNetworkWeights(self.model['backward']['network'], step_size)
        
        if self.model['forward']['training']:
            step_size = self.model['forward']['step_size'] / len(sample_transitions)
            self.updateNetworkWeights(self.model['forward']['network'], step_size)
        

    def plan(self):
        return 0

    def getTransitionFromBuffer(self, n=1):
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.choices(self.transition_buffer, k=n)

    def rolloutWithModel(self, state, action, model, rollout_policy=None, h=1, type='forward'):
        # get numpy state and action, returns torch h future next state
        x_old = state.unsqueeze(0).to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            pred_state = model['network'](x_old, action_onehot).detach()[:,action_index]
        else:
            pred_state = model['network'](x_old, action_onehot).detach()

        if h == 1:
            return pred_state[0]#.cpu().numpy()
        else:
            action = rollout_policy(pred_state[0])
            return self.rolloutWithModel(pred_state[0], action, model, rollout_policy, h = h-1)

    def forwardRolloutPolicy(self, state):
        return self.policy(torch.from_numpy(state).unsqueeze(0))

    def _getStateFromBuffer(self, model):
        # todo: add prioritized sweeping
        while len(model['plan_buffer']) > 0:
            yield model['plan_buffer'].pop()

    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False, type='forward'):
        # todo: add hallucination training

        x_old = state.float().unsqueeze(0).to(self.device)
        x_new = next_state.float().unsqueeze(0).to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        if type == 'forward':
            if len(model['layers_type']) + 1 == model['action_layer_number']:
                input = model['network'](x_old, action_onehot)[:, action_index]
            else:
                input = model['network'](x_old, action_onehot)
            target = x_new

        elif type =='backward':
            if len(model['layers_type']) + 1 == model['action_layer_number']:
                input = model['network'](x_new, action_onehot)[:, action_index]
            else:
                input = model['network'](x_new, action_onehot)
            target = x_old

        else:
            raise ValueError("type is not defined")

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

    # def start(self, observation):
    #     self.prev_state = self.agentState(observation)
    #
    #     x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
    #     self.prev_action = self.policy(x_old)
    #
    #     self.initModel()
    #
    #     return self.prev_action
    #
    # def step(self, reward, observation):
    #     self.state = self.agentState(observation)
    #
    #     self.updateTransitionBuffer(utils.transition(self.prev_state, self.prev_action, reward,
    #                                                  self.state, self.time_step))
    #
    #     x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
    #     x_new = torch.from_numpy(self.state).unsqueeze(0)
    #
    #     self.action = self.policy(x_new)
    #
    #     self.trainModel()
    #
    #     self.prev_state = self.state
    #     self.prev_action = self.action # another option:** we can again call self.policy function **
    #
    #     return self.prev_action
    #
    # def end(self, reward):
    #     pass