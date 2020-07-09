
import torch
import torch.nn as nn
import numpy as np
import random

from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.Networks.ModelNN.StateTransitionModel as STModel

class ForwardDynaAgent2(BaseDynaAgent):
    def __init__(self, params):
        super(ForwardDynaAgent, self).__init__(params)
        # self.model_batch_counter = 0
        self.model = {'forward': dict(network=params['model'],
                                      step_size=0.1,
                                      layers_type=['fc', 'fc'],
                                      layers_features=[256, 64],
                                      action_layer_number=3,
                                      batch_size=8,
                                      halluc_steps=2,
                                      training=True,
                                      plan_steps=0,
                                      plan_horizon=3,
                                      plan_buffer_size=1,
                                      plan_buffer=[])}
        self.true_model = params['true_model']
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

    def trainModel(self, terminal=False):
        sample_transitions = self.getTransitionFromBuffer(n=self.model['forward']['batch_size'])
        for sample in sample_transitions:
            state, action, reward, next_state, _ = sample
            if self.model['forward']['training'] is True:
                self._calculateGradients(self.model['forward'], state, action, next_state, terminal=terminal)

        step_size = self.model['forward']['step_size'] / len(sample_transitions)
        self.updateNetworkWeights(self.model['forward']['network'], step_size)

    def plan(self):
        self.updatePlanningBuffer(self.model['forward'], self.prev_state)

        for state in self.getStateFromPlanningBuffer(self.model['forward']):
            prev_action = self.forwardRolloutPolicy(state)
            for j in range(self.model['forward']['plan_horizon']):
                next_state = self.rolloutWithModel(state, prev_action,
                                                   self.model['forward'],
                                                   self.forwardRolloutPolicy,
                                                   h=1)
                assert next_state.shape == self.goal.shape, 'goal and pred states have different shapes'
                reward = -1
                if np.array_equal(next_state, self.goal):
                    reward = 10
                x_old = state.float().to(self.device)
                x_new = next_state.float().to(self.device)
                action = self.forwardRolloutPolicy(next_state)

                # mb_update = self.vf['q']['network'](x_new).detach()[:, self.getActionIndex(action)]
                # mf_update = self.vf['q']['network'](torch.from_numpy(self.state).unsqueeze(0)).detach()[:, self.getActionIndex(action)]
                # print(mb_update," - ", mf_update," = ", mb_update-mf_update)
                self.updateValueFunction(reward, x_old, prev_action, x_new= x_new, action=action)
                state = next_state
                prev_action = action

    def rolloutWithModel(self, state, action, model, rollout_policy=None, h=1):        
        # get torch state and action, returns torch h future next state
        x_old = state.float().unsqueeze(0).to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            pred_state = model['network'](x_old, action_onehot).detach()[:,action_index]
        else:
            pred_state = model['network'](x_old, action_onehot).detach()

        if h == 1:
            return pred_state[0]
        else:
            action = rollout_policy(pred_state)
            return self.rolloutWithModel(pred_state[0], action, model, rollout_policy, h = h-1)

    def forwardRolloutPolicy(self, state):
        return self.policy(state)



    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training
        # state and next_state are tensor, action is numpy array
        state = state.unsqueeze(0).float().to(self.device)
        next_state = next_state.unsqueeze(0).float().to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            input = model['network'](state, action_onehot)[:, action_index]
        else:
            input = model['network'](state, action_onehot)
        target = next_state

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

    def getTransitionFromBuffer(self, n=1):
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.choices(self.transition_buffer, k=n)


    def updatePlanningBuffer(self, model, state):
        model['plan_buffer'].append(state)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        return random.choices(model['plan_buffer'], k=1)




class ForwardDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(ForwardDynaAgent, self).__init__(params)
        # self.model_batch_counter = 0
        self.model = {'forward': dict(network=params['model'],
                                      step_size=0.1,
                                      layers_type=['fc', 'fc'],
                                      layers_features=[256, 64],
                                      action_layer_number=3,
                                      batch_size=8,
                                      halluc_steps=2,
                                      training=True,
                                      plan_steps=0,
                                      plan_horizon=3,
                                      plan_buffer_size=1,
                                      plan_buffer=[])}
        self.true_model = params['true_model']
    def initModel(self):

        if self.model['forward']['network'] is None:
            self.model['forward']['network'] = \
                STModel.StateTransitionModel(
                self.getStateRepresentation(self.prev_state).shape,
                len(self.action_list),
                self.model['forward']['layers_type'],
                self.model['forward']['layers_features'],
                self.model['forward']['action_layer_number'])
            self.model['forward']['network'].to(self.device)

    def trainModel(self, terminal=False):
        sample_transitions = self.getTransitionFromBuffer(n=self.model['forward']['batch_size'])
        for sample in sample_transitions:
            state, action, reward, next_state, _ = sample
            if self.model['forward']['training'] is True:
                self._calculateGradients(self.model['forward'], state, action, next_state, terminal=terminal)

        step_size = self.model['forward']['step_size'] / len(sample_transitions)
        self.updateNetworkWeights(self.model['forward']['network'], step_size)

    def plan(self):
        self.updatePlanningBuffer(self.model['forward'], self.prev_state)

        for state in self.getStateFromPlanningBuffer(self.model['forward']):
            prev_action = self.forwardRolloutPolicy(state)
            for j in range(self.model['forward']['plan_horizon']):
                next_state = self.rolloutWithModel(state, prev_action,
                                                   self.model['forward'],
                                                   self.forwardRolloutPolicy,
                                                   h=1)
                assert next_state.shape == self.getStateRepresentation(self.goal).shape, 'goal and pred states have different shapes'
                reward = -1
                if np.array_equal(next_state, self.goal):
                    reward = 10
                x_old = state.float().to(self.device)
                x_new = next_state.float().to(self.device)
                action = self.forwardRolloutPolicy(next_state)

                # mb_update = self.vf['q']['network'](x_new).detach()[:, self.getActionIndex(action)]
                # mf_update = self.vf['q']['network'](torch.from_numpy(self.state).unsqueeze(0)).detach()[:, self.getActionIndex(action)]
                # print(mb_update," - ", mf_update," = ", mb_update-mf_update)
                self.updateValueFunction(reward, x_old, prev_action, x_new= x_new, action=action)
                state = next_state
                prev_action = action

    def rolloutWithModel(self, state, action, model, rollout_policy=None, h=1):
        # get torch state and action, returns torch h future next state
        x_old = self.getStateRepresentation(state).float().unsqueeze(0).to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            pred_state = model['network'](x_old, action_onehot).detach()[:,action_index]
        else:
            pred_state = model['network'](x_old, action_onehot).detach()

        if h == 1:
            return pred_state[0]
        else:
            action = rollout_policy(pred_state)
            return self.rolloutWithModel(pred_state[0], action, model, rollout_policy, h = h-1)

    def forwardRolloutPolicy(self, state_torch):
        # return self.policy(state)

        if np.random.rand() > self.epsilon:
            v = []
            for i, action in enumerate(self.action_list):
                if self.policy_values == 'q':
                    v.append(self.getStateActionValue(state_torch, action, type='q'))
                elif self.policy_values == 's':
                    v.append(self.getStateActionValue(state_torch, type='s'))

                elif self.policy_values == 'qs':
                    q = self.getStateActionValue(state_torch, action, type='q')
                    s = self.getStateActionValue(state_torch, type='s')
                    v.append((q + s) / 2)
                else:
                    raise ValueError('policy is not defined')
            action = self.action_list[np.argmax(v)]
        else:
            action = self.action_list[int(np.random.rand() * self.num_actions)]

        return action



    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training
        # state and next_state are tensor, action is numpy array
        state = self.getStateRepresentation(state).unsqueeze(0).float().to(self.device)
        next_state = self.getStateRepresentation(next_state).unsqueeze(0).float().to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            input = model['network'](state, action_onehot)[:, action_index]
        else:
            input = model['network'](state, action_onehot)
        target = next_state

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

    def getTransitionFromBuffer(self, n=1):
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.choices(self.transition_buffer, k=n)


    def updatePlanningBuffer(self, model, state):
        model['plan_buffer'].append(state)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        return random.choices(model['plan_buffer'], k=1)
