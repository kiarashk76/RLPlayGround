from Test.Agents.BaseDynaAgent import BaseDynaAgent
import Test.Networks.ModelNN.StateTransitionModel as STModel
import torch
import torch.nn as nn
import numpy as np

class ForwardBackwardDynaAgent(BaseDynaAgent):
    def __init__(self, params):
        super(ForwardBackwardDynaAgent, self).__init__(params)
        self.model = {'forward': dict(network=None,
                                      step_size=0.05,
                                      layers_type=['fc', 'fc'],
                                      layers_features=[256, 64],
                                      action_layer_number=2,
                                      batch_size=10,
                                      halluc_steps=2,
                                      training=True,
                                      plan_steps=2,
                                      plan_horizon=5,
                                      plan_buffer_size=5,
                                      plan_buffer=[]),
                      'backward': dict(network=None,
                                       step_size=0.05,
                                       layers_type=['fc', 'fc'],
                                       layers_features=[256, 64],
                                       action_layer_number=2,
                                       batch_size=10,
                                       halluc_steps=2,
                                       training= False,
                                       plan_steps=2,
                                       plan_horizon=5,
                                       plan_buffer_size=5,
                                       plan_buffer=[])}
    def initModel(self):
        self.model_batch_counter = 0

        if self.model['forward']['network'] is None:
            self.model['forward']['network'] = \
                STModel.StateTransitionModel(
                self.prev_state.shape,
                len(self.action_list),
                self.model['forward']['layers_type'],
                self.model['forward']['layers_features'],
                self.model['forward']['action_layer_number'])
        self.model['forward']['network'].to(self.device)

    def trainModel(self, state, action, next_state):
        self.model_batch_counter += 1
        if self.model['forward']['training'] is True:
            self._trainModel(self.model['forward'], state, action, next_state)

    def plan(self):
        self.model['forward']['plan_buffer'].append(self.state)
        for state in self._getStateFromBuffer(self.model['forward']):
            state_lst = [state]
            reward_lst = []
            action_lst = []
            for j in range(1, self.model['forward']['plan_horizon']):
                action = self.forwardRolloutPolicy(state)
                next_state = self.getNextStateFromModel(state, action,
                                             self.model['forward'],
                                             self.forwardRolloutPolicy,
                                             h=j)
                state_lst.append(next_state)
                action_lst.append(action)
                assert next_state[0].shape == self.goal.shape, 'goal and pred states have different shapes'
                if np.array_equal(next_state, self.goal):
                    reward_lst.append(10)
                else:
                    reward_lst.append(-1)
            action_lst.append(self.forwardRolloutPolicy(next_state))

            for i,state in enumerate(state_lst[:-self.model['forward']['plan_steps']]):
                reward = 0
                for r in range(len(reward_lst) - self.model['forward']['plan_steps'] + i - 2, i - 1, -1):
                    reward *= self.gamma
                    reward += reward_lst[r]
                x_old = torch.from_numpy(state).float().unsqueeze(0)
                x_new = torch.from_numpy(state_lst[i+self.model['forward']['plan_steps']])
                prev_action = action_lst[i]
                action = action_lst[i+self.model['forward']['plan_steps']]
                self.updateValueFunction(reward, x_old, x_new,prev_action=prev_action, action=action)

    def getNextStateFromModel(self, state, action, model, rollout_policy=None, h=1):
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
            return self.getNextStateFromModel(pred_state[0], action, model, rollout_policy, h = h-1)

    def forwardRolloutPolicy(self, state):
        return self.policy(torch.from_numpy(state).unsqueeze(0))

    def _getStateFromBuffer(self, model):
        # todo: add prioritized sweeping
        while len(model['plan_buffer']) > 0:
            yield model['plan_buffer'].pop()

    def _trainModel(self, model, state, action, next_state, h = 0):
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

        if self.model_batch_counter % model['batch_size'] == 0:
            self.updateNetworkWeights(model['network'], model['step_size'])

