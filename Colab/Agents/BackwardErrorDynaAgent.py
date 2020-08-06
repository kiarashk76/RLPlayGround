import random
import numpy as np
import torch
import torch.nn as nn

import Colab.Networks.ModelNN.StateTransitionModelwError as STModel
import Colab.utils as utils
from Colab.Agents.BaseDynaAgent import BaseDynaAgent


class BackwardErrorDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(BackwardErrorDynaAgent, self).__init__(params)
        self.model_batch_counter = 0
        self.model = {'backward': dict(network=params['model'],
                                       step_size=0.01,
                                       layers_type=['fc'],
                                       layers_features=[64],
                                       action_layer_number=2,
                                       batch_size=4,
                                       batch_counter=None,
                                       training=True,
                                       halluc_steps=2,
                                       plan_steps=2,
                                       plan_number=5,
                                       plan_horizon=1,
                                       plan_buffer_size=4,
                                       plan_buffer=[])}
        self.true_model = params['true_model']
        self.planning_transition_buffer = []
        self.planning_transition_buffer_size = 20

    def initModel(self, state):
        if self.model['backward']['network'] is not None:
            return
        self.model['backward']['network'] = \
            STModel.StateTransitionModelwError(
                state.shape,
                len(self.action_list),
                self.model['backward']['layers_type'],
                self.model['backward']['layers_features'],
                self.model['backward']['action_layer_number'])
        self.model['backward']['network'].to(self.device)
        self.model['backward']['batch_counter'] = 0

    def trainModel(self, terminal=False):
        if len(self.transition_buffer) < self._vf['q']['batch_size']:
            return
        transition_batch = self.getTransitionFromBuffer(n=self.model['backward']['batch_size'])
        for i, data in enumerate(transition_batch):
            prev_state, prev_action, reward, state, action, terminal, t = data
            if self.model['backward']['training'] is True:
                self._calculateGradients(self.model['backward'], prev_state, prev_action, state, terminal=terminal)
        step_size = self.model['backward']['step_size'] / len(transition_batch)
        self.updateNetworkWeights(self.model['backward']['network'], step_size)

    def plan(self):
        self.updatePlanningBuffer(self.model['backward'], self.state)
        for state in self.getStateFromPlanningBuffer(self.model['backward']):
            action = self.policy(state)
            for j in range(self.model['backward']['plan_horizon']):
                prev_action = self.backwardRolloutPolicy(state)
                prev_state, certainty = self.rolloutWithModel(state, prev_action, self.model['backward'])
                reward = -1
                terminal = self.isTerminal(state)
                if terminal:
                    reward = 10
                reward = torch.tensor(reward).unsqueeze(0).to(self.device)
                x_old = prev_state.float().to(self.device)
                x_new = state.float().to(self.device) if not terminal else None
                self.update_planning_transition_buffer(utils.transition(x_old, prev_action, reward,
                                                                        x_new, action, terminal, self.time_step))

                if self._vf['q']['training']:
                    if len(self.planning_transition_buffer) >= self._vf['q']['batch_size']:
                        transition_batch = self.getTransitionFromPlanningBuffer(n=self._vf['q']['batch_size'])
                        self.updateValueFunction(transition_batch, 'q')

                action = prev_action
                state = prev_state

    def rolloutWithModel(self, state, action, model, h=1):
        current_state = torch.tensor(state.data.clone())
        current_action = np.copy(action)
        sum_certainty = 0
        with torch.no_grad():
            for i in range(h):
                action_onehot = torch.from_numpy(self.getActionOnehot(current_action)).unsqueeze(0).to(self.device)
                action_index = self.getActionIndex(current_action)
                if len(model['layers_type']) + 1 == model['action_layer_number']:
                    prev_state = model['network'](current_state, action_onehot)[0][:, action_index]
                    terminal = model['network'](current_state, action_onehot)[1][:, action_index]
                    certainty = model['network'](current_state, action_onehot)[2][:, action_index]
                else:
                    prev_state = model['network'](current_state, action_onehot)[0]
                    terminal = model['network'](current_state, action_onehot)[1]
                    certainty = model['network'](current_state, action_onehot)[2]
                current_action = self.backwardRolloutPolicy(prev_state)
                current_state = prev_state
                sum_certainty += certainty
        return prev_state, sum_certainty

    def backwardRolloutPolicy(self, state):
        # with torch.no_grad:
        random_action = self.action_list[int(np.random.rand() * self.num_actions)]
        return random_action

    def _calculateGradients(self, model, state, action, next_state, h=0, terminal=False):
        # todo: add hallucination training
        if next_state is None:
            return
        state = state.float().unsqueeze(0)
        next_state = next_state.float().unsqueeze(0)
        action_onehot = torch.from_numpy(self.getActionOnehot(action)).unsqueeze(0).to(self.device)
        action_index = self.getActionIndex(action)

        if len(model['layers_type']) + 1 == model['action_layer_number']:
            model_state = model['network'](next_state, action_onehot)[0][:, action_index]
            model_certainty = model['network'](next_state, action_onehot)[2][:, action_index]
        else:
            model_state = model['network'](next_state, action_onehot)[0]
            model_certainty = model['network'](next_state, action_onehot)[2]

        input = model_state
        target = state

        assert target.shape == input.shape, 'target and input must have same shapes'
        loss = nn.MSELoss()(input, target)
        loss.backward()

        input = model_certainty
        x = state
        y = model_state.detach()
        err = (np.square(x - y)).mean()
        target = torch.tensor([err]).unsqueeze(0).to(self.device)
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
        diff = np.abs(np.multiply(self.goal, state)).sum()
        return diff <= 0.8

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
