from Test.Agents.BasePlannerAgent import BasePlannerAgent
from Test.Networks.StateValueFunction import StateVFNN
from Test.Networks.ModelNN.StateTransitionModel import StateTransitionModel
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from Test.Networks.ModelNN.StateTransitionModel import preTrainForwad

class ForwardPlannerAgent(BasePlannerAgent):
    def __init__(self, params = {}):
        super(ForwardPlannerAgent, self).__init__(params)

    #reward function must change
    def plan(self):
        for i in range (len(self.buffer)):
            start_state = self.buffer[i]
            state = torch.from_numpy(start_state.copy())
            state = state.unsqueeze(0)
            rewardList = []
            for j in range(self.planning_steps):
                action = self.rollout_policy(state)
                next_state = self.model(state, torch.from_numpy(action).unsqueeze(0))
                is_terminal = np.array_equal(next_state.detach().numpy()[0], self.goal)
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                # reward = self.reward_function(next_state.detach().numpy())
                rewardList.append(reward)
                if is_terminal:
                    break
                state = next_state.detach()

            state = start_state
            state = torch.from_numpy(state).unsqueeze(0)
            action = self.rollout_policy(state)
            action_index = self.getActionIndex(action)
            next_state = self.model(state, torch.from_numpy(action).unsqueeze(0))
            next_action = self.rollout_policy(next_state)
            next_action_index = self.getActionIndex(next_action)
            target = self.q_value_function[next_action_index](next_state).detach()
            rewardListLen = len(rewardList)
            for j in range (rewardListLen):
                target *= self.gamma
                index = rewardListLen - j - 1
                target += rewardList[index]

            input = self.q_value_function[action_index](state)
            loss = nn.MSELoss()(input, target)

            loss.backward()
            self.updateWeights(action_index)

    def rollout_policy(self, state):
        # random policy
        # action = self.action_list[int(np.random.rand() * self.num_actions)]
        v = []
        for i, action in enumerate(self.action_list):
            v.append(self.q_value_function[i](state))
        action = self.action_list[np.argmax(v)]
        return action



    def trainModel(self, state, action, next_state):
        x_old = torch.from_numpy(state).unsqueeze(0).float()
        x_new = torch.from_numpy(next_state).unsqueeze(0).float()
        action = torch.from_numpy(action).unsqueeze(0).float()

        input = self.model(x_old, action)
        target = x_new

        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights()
