from Test.Agents.BaseDynaAgent import BaseDynaAgent
import Test.Networks.ModelNN.StateTransitionModel as STModel
import torch
import torch.nn as nn
import numpy as np
import random

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
                                      plan_steps=1,
                                      plan_horizon=1,
                                      plan_buffer_size=1,
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
                assert next_state[0].shape == self.goal.shape, 'goal and pred states have different shapes'
                reward = -1
                if np.array_equal(next_state[0], self.goal):
                    reward = 10
                x_old = torch.from_numpy(state).float().unsqueeze(0)
                x_new = torch.from_numpy(next_state).float()
                action = self.forwardRolloutPolicy(next_state)

                # mb_update = self.vf['q']['network'](x_new).detach()[:, self.getActionIndex(action)]
                # mf_update = self.vf['q']['network'](torch.from_numpy(self.state).unsqueeze(0)).detach()[:, self.getActionIndex(action)]
                # print(mb_update," - ", mf_update," = ", mb_update-mf_update)
                
                self.updateValueFunction(reward, x_old, x_new, prev_action=prev_action, action=action)
                state = next_state[0]
                prev_action = action

    def rolloutWithModel(self, state, action, model, rollout_policy=None, h=1):
        # get numpy state and action, returns torch h future next state
        x_old = torch.from_numpy(state).unsqueeze(0)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            pred_state = model['network'](x_old, action_onehot).detach()[:,action_index]
        else:
            pred_state = model['network'](x_old, action_onehot).detach()

        if h == 1:
            return pred_state.numpy()
        else:
            pred_state = pred_state.numpy()
            action = rollout_policy(pred_state)
            return self.rolloutWithModel(pred_state[0], action, model, rollout_policy, h=h - 1)

    def forwardRolloutPolicy(self, state):
        return self.policy(torch.from_numpy(state).unsqueeze(0))



    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training

        x_old = torch.from_numpy(state).float().unsqueeze(0)
        x_new = torch.from_numpy(next_state).float().unsqueeze(0)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            input = model['network'](x_old, action_onehot)[:, action_index]
        else:
            input = model['network'](x_old, action_onehot)
        target = x_new

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
