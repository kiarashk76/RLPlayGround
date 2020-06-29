from Test.Agents.BaseDynaAgent import BaseDynaAgent
import Test.Networks.ModelNN.StateTransitionModel as STModel
from ..Networks.ValueFunction.StateActionValueFunction import StateActionVFNN3

import torch
import torch.nn as nn
import numpy as np
import random


class Dyna2Agent(BaseDynaAgent):
    def __init__(self, params):
        super(Dyna2Agent, self).__init__(params)
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
        self.short_vf = {'q': dict(network=None,
                        layers_type=['fc'],
                        layers_features=[32],
                        action_layer_num=2, # if one more than layer numbers => we will have num of actions output
                        batch_size=1,
                        step_size=0.05,
                        training=True),
                        's': dict(network=None,
                        layers_type=['fc', 'fc'],
                        layers_features=[64, 32],
                        batch_size=1,
                        step_size=0.01,
                        training=False)}

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

        nn_state_shape = (self.short_vf['q']['batch_size'],) + self.prev_state.shape
        self.short_vf['q']['network'] = StateActionVFNN3(nn_state_shape, self.num_actions,
                                                 self.short_vf['q']['layers_type'],
                                                 self.short_vf['q']['layers_features'],
                                                 self.short_vf['q']['action_layer_num']).to(self.device)

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

                self.updateShortValueFunction(reward, x_old, x_new, prev_action=prev_action, action=action)
                state = next_state[0]
                prev_action = action

    def rolloutWithModel(self, state, action, model, rollout_policy=None, h=1):
        # get numpy state and action, returns torch h future next state
        x_old = torch.from_numpy(state).unsqueeze(0)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0)
        action_index = self.getActionIndex(action)
        if len(model['layers_type']) + 1 == model['action_layer_number']:
            pred_state = model['network'](x_old, action_onehot).detach()[:, action_index]
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

    def policy(self, state_torch):
        if np.random.rand() > self.epsilon:
            v = []
            for i, action in enumerate(self.action_list):
                if self.policy_values == 'q':
                    v.append(self.getStateActionValue(state_torch, action, type= 'q')
                             + self.getStateActionShortValue(state_torch, action, type='q'))
                elif self.policy_values == 's':
                    v.append(self.getStateActionValue(state_torch, action, type='s'))
                elif self.policy_values == 'qs':
                    q = self.getStateActionValue(state_torch, action, type='q')
                    s = self.getStateActionValue(state_torch, action, type='s')
                    v.append( (q+s) /2)
                else:
                   raise ValueError('policy is not defined')
            action = self.action_list[np.argmax(v)]
        else:
            action = self.action_list[int(np.random.rand() * self.num_actions)]

        return action

    def updateShortValueFunction(self, reward, x_old, x_new, prev_action, action):
        if x_new is not None:  # Not a terminal State
            assert x_old.shape == x_new.shape, 'x_old and x_new have different shapes'
            prev_action_index = self.getActionIndex(prev_action)
            action_index = self.getActionIndex(action)
            prev_action_onehot = torch.from_numpy(self.getActionOnehot(prev_action)).float().unsqueeze(0)
            action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0)
            if self.short_vf['q']['training']:
                if len(self.short_vf['q']['layers_type']) + 1 == self.short_vf['q']['action_layer_num']:
                    target = reward + self.gamma * (self.short_vf['q']['network'](x_new).detach()[:, action_index]
                                                    + self.getStateActionValue(x_new, action))
                    input = self.short_vf['q']['network'](x_old)[:, prev_action_index] \
                            + self.getStateActionValue(x_old, prev_action)
                else:
                    target = reward + self.gamma * (self.short_vf['q']['network'](x_new, action_onehot).detach()
                             + self.getStateActionValue(x_new, action))
                    input = self.short_vf['q']['network'](x_old, prev_action_onehot)\
                            + self.getStateActionValue(x_old, prev_action)
                assert target.shape == input.shape, 'target and input must have same shapes'
                loss = nn.MSELoss()(input, target)
                loss.backward()
                if self.batch_counter % self.short_vf['q']['batch_size'] == 0:
                    step_size = self.short_vf['q']['step_size'] / self.short_vf['q']['batch_size']
                    self.updateNetworkWeights(self.short_vf['q']['network'],
                                              step_size)

            # if self.vf['s']['training']:
            #     target = reward + self.gamma * self.vf['s']['network'][action_index](x_new).detach()
            #     input = self.vf['s']['network'][prev_action_index](x_old)
            #     assert target.shape == input.shape, 'target and input must have same shapes'
            #     loss = nn.MSELoss()(input, target)
            #     loss.backward()
            #     if self.batch_counter % self.vf['s']['batch_size'] == 0:
            #         step_size = self.vf['s']['step_size'] / self.vf['s']['batch_size']
            #         self.updateNetworkWeights(self.vf['s']['network'][prev_action_index],
            #                                   step_size)
        else:  # terminal state
            prev_action_index = self.getActionIndex(prev_action)
            prev_action_onehot = torch.from_numpy(self.getActionOnehot(prev_action)).float().unsqueeze(0)
            if self.short_vf['q']['training']:
                if len(self.short_vf['q']['layers_type']) + 1 == self.short_vf['q']['action_layer_num']:
                    target = torch.tensor(reward).unsqueeze(0)
                    input = self.short_vf['q']['network'](x_old)[:, prev_action_index]
                else:
                    target = torch.tensor(reward).unsqueeze(0).unsqueeze(0)
                    input = self.short_vf['q']['network'](x_old, prev_action_onehot)

                assert target.shape == input.shape, 'target and input must have same shapes'
                loss = nn.MSELoss()(input.float(), target.float())
                loss.backward()

                if self.batch_counter % self.short_vf['q']['batch_size'] == 0:
                    step_size = self.short_vf['q']['step_size'] / self.short_vf['q']['batch_size']
                else:
                    step_size = self.short_vf['q']['step_size'] / \
                                (self.batch_counter % self.short_vf['q']['batch_size'])

                self.updateNetworkWeights(self.short_vf['q']['network'], step_size)
            # if self.vf['s']['training']:
            #     target = torch.tensor(reward).unsqueeze(0).unsqueeze(0)
            #     input = self.vf['s']['network'][prev_action_index](x_old)
            #     assert target.shape == input.shape, 'target and input must have same shapes'
            #     loss = nn.MSELoss()(input.float(), target.float())
            #     loss.backward()
            #
            #     if self.batch_counter % self.vf['s']['batch_size'] == 0:
            #         step_size = self.vf['s']['step_size'] / self.vf['s']['batch_size']
            #     else:
            #         step_size = self.vf['s']['step_size'] / \
            #                     (self.batch_counter % self.vf['s']['batch_size'])
            #     self.updateNetworkWeights(self.vf['s']['network'][prev_action_index], step_size)

    def getStateActionShortValue(self, x, action, type='q'):
        action_index = self.getActionIndex(action)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).float().unsqueeze(0)
        if type == 'q':
            if len(self.short_vf['q']['layers_type']) + 1 == self.short_vf['q']['action_layer_num']:
                value = self.short_vf['q']['network'](x).detach()[:, action_index]
            else:
                value = self.short_vf['q']['network'](x, action_onehot).detach()

        elif type == 's':
            value = self.short_vf['s']['network'][action_index](x).detach()

        else:
            raise ValueError('state action value type is not defined')

        return value

