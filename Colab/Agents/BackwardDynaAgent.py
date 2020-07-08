
import torch
import torch.nn as nn
import numpy as np
import random

from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.Networks.ModelNN.StateTransitionModel as STModel

class BackwardDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(BackwardDynaAgent, self).__init__(params)
        self.model_batch_counter = 0
        self.model = {'backward': dict(network=params['model'],
                                      step_size=0.1,
                                      layers_type=['fc', 'fc'],
                                      layers_features=[256, 64],
                                      action_layer_number=3,
                                      batch_size=8,
                                      halluc_steps=2,
                                      training=True,
                                      plan_steps=0,
                                      plan_horizon=100,
                                      plan_buffer_size=1,
                                      plan_buffer=[])}
        self.true_model = params['true_model']
    def initModel(self):
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
        sample_transitions = self.getTransitionFromBuffer(n=self.model['backward']['batch_size'])
        for sample in sample_transitions:
            state, action, reward, next_state, _ = sample
            if self.model['backward']['training'] is True:
                self._calculateGradients(self.model['backward'], state, action, next_state, terminal=terminal)

        step_size = self.model['backward']['step_size'] / len(sample_transitions)
        self.updateNetworkWeights(self.model['backward']['network'], step_size)

    def plan(self):
        self.updatePlanningBuffer(self.model['backward'], self.prev_state)

        for state in self.getStateFromPlanningBuffer(self.model['backward']):
            action = self.policy(state.float().unsqueeze(0))
            for j in range(self.model['backward']['plan_horizon']):
                prev_action = self.backwardRolloutPolicy(state)
                # prev_state = self.rolloutWithModel(state, prev_action,
                #                                    self.model['backward'],
                #                                    self.backwardRolloutPolicy,
                #                                    h=1)
                true_prev_state = self.true_model(state.cpu().numpy(), prev_action, type='sample')
                if true_prev_state is None:
                    break
                true_prev_state = torch.from_numpy(true_prev_state).to(self.device)

                prev_state = true_prev_state #true_prev_state
                
                assert prev_state.shape == self.goal.shape, 'goal and pred states have different shapes'
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
            pred_state = pred_state.numpy()
            action = rollout_policy(pred_state)
            return self.rolloutWithModel(pred_state[0], action, model, rollout_policy, h=h - 1)

    def backwardRolloutPolicy(self, state):
        random_action = self.action_list[int(np.random.rand() * self.num_actions)]
        return random_action
        return self.policy(state)

    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training
        state = state.float().unsqueeze(0)
        next_state = next_state.float().unsqueeze(0)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            input = model['network'](next_state, action_onehot)[:, action_index]
        else:
            input = model['network'](next_state, action_onehot)
        target = state

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

    def getTransitionFromBuffer(self, n=1):
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.choices(self.transition_buffer, k=n)

    def updatePlanningBuffer(self, model, state):
        # model['plan_buffer'].append(state)
        model['plan_buffer'].append(self.goal)
        if len(model['plan_buffer']) > model['plan_buffer_size']:
            self.removeFromPlanningBuffer(model)

    def removeFromPlanningBuffer(self, model):
        model['plan_buffer'].pop(0)

    def getStateFromPlanningBuffer(self, model):
        # todo: add prioritized sweeping
        return random.choices(model['plan_buffer'], k=1)