from Test.Agents.BasePlannerAgent import BasePlannerAgent
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

class BackwardPlannerAgent(BasePlannerAgent):
    def __init__(self, params = {}):
        super(BackwardPlannerAgent, self).__init__(params)

    #reward function must change
    def plan(self):
        for i in range (len(self.buffer)):
            next_state = self.buffer[i]
            next_state = torch.from_numpy(next_state).unsqueeze(0)
            next_action = self.policy(next_state)

            for j in range(self.planning_steps):
                action = self.rollout_policy(next_state)
                action_index = self.getActionIndex(action)
                state = self.model(next_state, torch.from_numpy(action).unsqueeze(0))
                next_action_index = self.getActionIndex(next_action)

                is_terminal = np.array_equal(next_state.detach().numpy()[0], self.goal)
                # reward = self.reward_function(next_state.detach().numpy())
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                target = reward
                if not is_terminal:
                    target += self.gamma * self.q_value_function[next_action_index](next_state).detach()
                input = self.q_value_function[action_index](state)
                loss = nn.MSELoss()(input, target)
                loss.backward()
                self.updateWeights(action_index)
                if is_terminal:
                    break
                next_state = state.detach()
                next_action = action


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
        input = self.model(x_new, action)
        target = x_old
        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights()
