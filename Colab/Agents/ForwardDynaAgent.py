
import torch
import torch.nn as nn
import numpy as np
import random

from Colab.Agents.BaseDynaAgent import BaseDynaAgent
import Colab.Networks.ModelNN.StateTransitionModel as STModel
import Colab.utils as utils

class ForwardDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(ForwardDynaAgent, self).__init__(params)
        self.model = {'forward': dict(network=params['model'],
                                      step_size=0.01,
                                      layers_type=['fc'],
                                      layers_features=[64],
                                      action_layer_number=2,
                                      batch_size=4,
                                      halluc_steps=2,
                                      training=True,
                                      plan_horizon=3,
                                      plan_buffer_size=1,
                                      plan_buffer=[])}
        self.true_model = params['true_model']
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

    def trainModel(self):
        if len(self.transition_buffer) >= self.model['forward']['batch_size']:
            transition_batch = self.getTransitionFromBuffer(n=self.model['forward']['batch_size'])
            for transition in transition_batch:
                if self.model['forward']['training'] is True:
                    self._calculateGradients(self.model['forward'], transition.prev_state,
                                             transition.prev_action, transition.state)

            self.updateNetworkWeights(self.model['forward']['network'],
                                      self.model['forward']['step_size'])

    def plan(self):
        return 0
        with torch.no_grad():
            current_state = torch.tensor(self.prev_state.data.clone())
            true_current_state = self.prev_state.cpu().numpy()[0]
            current_action = np.copy(self.prev_action)
            for h in range(self.model['forward']['plan_horizon']):
                next_state, is_terminal = self.rolloutWithModel(current_state, current_action, self.model['forward'], 1)
                is_terminal = np.random.binomial(n=1, p=float(is_terminal.data.cpu().numpy()), size=1)
                true_next_state, _, reward = self.true_model(true_current_state, current_action)
                reward = torch.tensor(reward).unsqueeze(0).to(self.device)
                next_action = self.forwardRolloutPolicy(next_state)

                if is_terminal:
                    self.updateTransitionBuffer(utils.transition(current_state, current_action, reward,
                                                             None, None, True, self.time_step))
                else:
                    self.updateTransitionBuffer(utils.transition(current_state, current_action, reward,
                                                                 next_state, next_action, False, self.time_step))
                current_state = next_state
                current_action = next_action
                true_current_state = true_next_state

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
        return pred_state, is_terminal

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