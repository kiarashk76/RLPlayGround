import torch
import torch.nn as nn
import numpy as np
import random

from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.Networks.ModelNN.StateTransitionModelwError as Model

class ForwardErrorDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(ForwardErrorDynaAgent, self).__init__(params)
        self.model = {'forward': dict(network=params['model'],
                                      step_size=0.1,
                                      layers_type=['fc', 'fc'],
                                      layers_features=[256, 64],
                                      action_layer_number=2,
                                      batch_size=8,
                                      halluc_steps=2,
                                      training=True,
                                      plan_steps=0,
                                      plan_horizon=1,
                                      plan_buffer_size=1,
                                      plan_buffer=[])}

    def initModel(self):
        if self.model['forward']['network'] is not None:
            return
        self.model['forward']['network'] = Model.StateTransitionModelwError(
            self.prev_state.shape,
            len(self.action_list),
            self.model['forward']['layers_type'],
            self.model['forward']['layers_features'],
            self.model['forward']['action_layer_number'])
        self.model['forward']['network'].to(self.device)

    def trainModel(self, terminal=False):
        if self.model['forward']['training'] is False:
            return
        sample_transitions = self.getTransitionFromBuffer(n=self.model['forward']['batch_size'])
        for sample in sample_transitions:
            state, action, reward, next_state, _ = sample
            self._calculateGradients(self.model['forward'], state, action, next_state, terminal=terminal)

        step_size = self.model['forward']['step_size'] / len(sample_transitions)
        self.updateNetworkWeights(self.model['forward']['network'], step_size)

    def plan(self):
        self.updatePlanningBuffer(self.model['forward'], self.prev_state)

        for state in self.getStateFromPlanningBuffer(self.model['forward']):
            prev_action = self.forwardRolloutPolicy(state)
            for j in range(self.model['forward']['plan_horizon']):
                next_state, acc = self.rolloutWithModel(state, prev_action,
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
                self.updateValueFunction(reward, x_old, prev_action, x_new= x_new, action=action)
                state = next_state
                prev_action = action

    def rolloutWithModel(self, state, action, model, rollout_policy=None, h=1):
        # get torch state and action, returns torch h future next state
        # works only for h = 1
        x_old = state.float().unsqueeze(0).to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        pred_state, acc = model['network'](x_old, action_onehot)
        pred_state = pred_state.detach()
        acc = acc.detach()
        if h == 1:
            return pred_state[0], acc

        action = rollout_policy(pred_state)
        pred_pred_state, pred_acc = self.rolloutWithModel(pred_state[0], action, model, rollout_policy, h = h-1)
        return pred_pred_state, pred_acc + acc

    def forwardRolloutPolicy(self, state): #on-policy
        return self.policy(state)

    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training
        # state and next_state are tensor, action is numpy array
        state = state.unsqueeze(0).float().to(self.device)
        next_state = next_state.unsqueeze(0).float().to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        model_next_state, model_acc = model['network'](state, action_onehot)
        x = next_state
        y = model_next_state.detach()
        acc = (np.square(x - y)).mean()
        loss1 = nn.MSELoss()(model_next_state, next_state)
        loss2 = nn.MSELoss()(model_acc[0], acc)
        loss = loss1 + loss2
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