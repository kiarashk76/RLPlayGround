import torch
import torch.nn as nn
import numpy as np
import random

from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.Networks.ModelNN.StateTransitionModelwError as Model


class BackwardErrorDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(BackwardErrorDynaAgent, self).__init__(params)
        self.model_pred_error = []
        self.model = {'backward': dict(network = params['model'],
                                       step_size = 0.1,
                                       layers_type = ['fc', 'fc'],
                                       layers_features = [256, 64],
                                       action_layer_number = 2,
                                       batch_size = 8,
                                       halluc_steps = 2,
                                       training = True,
                                       plan_steps = 0,
                                       plan_horizon = 1,
                                       plan_buffer_size = 1,
                                       plan_buffer = [])}

    def initModel(self):
        if self.model['backward']['network'] is not None:
            return
        self.model['backward']['network'] = Model.StateTransitionModelwError(
            self.prev_state.shape,
            len(self.action_list),
            self.model['backward']['layers_type'],
            self.model['backward']['layers_features'],
            self.model['backward']['action_layer_number'])
        self.model['backward']['network'].to(self.device)

    def trainModel(self, terminal = False):
        if self.model['backward']['training'] is False:
            return
        sample_transitions = self.getTransitionFromBuffer(n = self.model['backward']['batch_size'])
        for sample in sample_transitions:
            state, action, reward, next_state, _ = sample
            self._calculateGradients(self.model['backward'], next_state, action, state, terminal = terminal)
        step_size = self.model['backward']['step_size'] / len(sample_transitions)
        self.updateNetworkWeights(self.model['backward']['network'], step_size)

    def plan(self):
        print('heh')
        self.updatePlanningBuffer(self.model['backward'], self.prev_state)
        for state in self.getStateFromPlanningBuffer(self.model['backward']):
            action = self.policy(state.float().unsqueeze(0))
            for j in range(self.model['backward']['plan_horizon']):
                prev_action = self.backwardRolloutPolicy(state)
                prev_state, acc = self.rolloutWithModel(state, prev_action,
                                                        self.model['backward'],
                                                        self.backwardRolloutPolicy(state),
                                                        h = 1)
                reward = -1
                is_terminal = False
                if np.array_equal(state, self.goal):
                    reward = 10
                    is_terminal = True
                x_old = prev_state.float().to(self.device)
                x_new = state.float().to(self.device) if not is_terminal else None
                self.updateValueFunction(reward, x_old, prev_action, x_new=x_new, action=action)
                action = prev_action
                state = prev_state


    def rolloutWithModel(self, state, action, model, rollout_policy = None, h = 1):
        # get torch state and action, returns torch h future next state
        # works only for h = 1
        x_old = state.float().unsqueeze(0).to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        prev_state, acc = model['network'](x_old, action_onehot)
        prev_state = prev_state.detach()
        acc = acc.detach()
        if h == 1:
            return prev_state[0], acc
        action = rollout_policy(prev_state)
        pred_pred_state, pred_acc = self.rolloutWithModel(prev_state[0], action, model, rollout_policy, h = h - 1)
        return pred_pred_state, pred_acc + acc

    def backwardRolloutPolicy(self, state):
        random_action = self.action_list[int(np.random.rand() * self.num_actions)]
        return random_action

    def _calculateGradients(self, model, state, action, prev_state, h = 0, terminal = False):
        # todo: add hallucination training
        # state and next_state are tensor, action is numpy array
        state = state.unsqueeze(0).float().to(self.device)
        prev_state = prev_state.unsqueeze(0).float().to(self.device)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        model_prev_state, model_acc = model['network'](state, action_onehot)
        x = prev_state
        y = model_prev_state.detach()
        acc = (np.square(x - y)).mean()
        # print(acc , model_acc[0])
        self.sum_pred_model_error += model_acc[0]
        self.count_pred_model_error += 1
        loss1 = nn.MSELoss()(model_prev_state, prev_state)
        loss2 = nn.MSELoss()(model_acc[0], acc)
        loss = loss1 + loss2
        loss.backward()

    def getTransitionFromBuffer(self, n = 1):
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.choices(self.transition_buffer, k = n)

    def updatePlanningBuffer(self, model, state):
        model['plan_buffer'].append(state)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        return random.choices(model['plan_buffer'], k = 1)

    def start(self, observation):
        self.sum_pred_model_error = 0
        self.count_pred_model_error = 0
        return super(BackwardErrorDynaAgent, self).start(observation)

    def end(self, reward):
        self.model_pred_error.append(self.sum_pred_model_error / self.count_pred_model_error)
        return super(BackwardErrorDynaAgent, self).end(reward)
