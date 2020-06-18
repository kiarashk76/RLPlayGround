from Test.Agents.BasePlannerAgent import BasePlannerAgent
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

class ForwardPlannerAgent2(BasePlannerAgent):
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
                next_state = self.model(state, torch.from_numpy(action).unsqueeze(0)).detach()
                is_terminal = np.array_equal(next_state.detach().numpy()[0], self.goal)
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                # reward = self.reward_function(next_state.detach().numpy())
                rewardList.append(reward)
                if is_terminal:
                    break
                state = torch.from_numpy(np.copy(next_state))

            state = start_state
            state = torch.from_numpy(state).unsqueeze(0)
            action = self.rollout_policy(state)
            action_index = self.getActionIndex(action)
            next_state = self.model(state, torch.from_numpy(action).unsqueeze(0)).detach()
            next_action = self.rollout_policy(next_state)
            next_action_index = self.getActionIndex(next_action)
            target = self.q_value_function(next_state).detach()[0, next_action_index]
            rewardListLen = len(rewardList)
            for j in range (rewardListLen):
                target *= self.gamma
                index = rewardListLen - j - 1
                target += rewardList[index]

            input = self.q_value_function(state)[0, action_index]
            # print(target.shape, input.shape)
            loss = nn.MSELoss()(input, target)

            loss.backward()
            self.updateWeights(action_index)

    def plan2(self):
        for i in range (len(self.buffer)):
            start_state = self.buffer[i]
            state = torch.from_numpy(start_state.copy())
            state = state.unsqueeze(0)
            rewardList = []
            for j in range(self.planning_steps):
                action = self.rollout_policy(state)
                action_index = self.getActionIndex(action)
                next_state = self.model(state).detach()[:, action_index]
                is_terminal = np.array_equal(next_state.numpy()[0], self.goal)
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                # reward = self.reward_function(next_state.detach().numpy())
                rewardList.append(reward)
                if is_terminal:
                    break
                state = torch.from_numpy(np.copy(next_state))

            state = start_state
            state = torch.from_numpy(state).unsqueeze(0)
            action = self.rollout_policy(state)
            action_index = self.getActionIndex(action)

            next_state = self.model(state).detach()[:, action_index]

            next_action = self.rollout_policy(next_state)
            next_action_index = self.getActionIndex(next_action)
            target = self.q_value_function(next_state).detach()[0, next_action_index]
            rewardListLen = len(rewardList)
            for j in range (rewardListLen):
                target *= self.gamma
                index = rewardListLen - j - 1
                target += rewardList[index]

            input = self.q_value_function(state)[0, action_index]
            loss = nn.MSELoss()(input, target)

            loss.backward()
            self.updateWeights(action_index)

    def rollout_policy(self, state):
        # random policy
        # action = self.action_list[int(np.random.rand() * self.num_actions)]
        v = []
        for i, action in enumerate(self.action_list):
            v.append(self.q_value_function(state).detach()[0,i])
        action = self.action_list[np.argmax(v)]
        return action

    def trainModel(self, state, action, next_state):
        state = np.asarray(self.model_batch['prev_state'])
        next_state = np.asarray(self.model_batch['state'])
        action = np.asarray(self.model_batch['action'])

        x_old = torch.from_numpy(state).float()
        x_new = torch.from_numpy(next_state).float()
        action = torch.from_numpy(action).float()

        input = self.model(x_old, action)
        target = x_new
        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights()

    def trainModel2(self, state, action, next_state):
        x_old = torch.from_numpy(state).unsqueeze(0).float()
        x_new = torch.from_numpy(next_state).unsqueeze(0).float()
        action_index = self.getActionIndex(action)

        input = self.model(x_old)[:, action_index]  #B , num_action, state shape
        target = x_new

        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights()


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
                next_state = self.model(state, torch.from_numpy(action).unsqueeze(0)).detach()
                is_terminal = np.array_equal(next_state.detach().numpy()[0], self.goal)
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                # reward = self.reward_function(next_state.detach().numpy())
                rewardList.append(reward)
                if is_terminal:
                    break
                state = torch.from_numpy(np.copy(next_state))

            state = start_state
            state = torch.from_numpy(state).unsqueeze(0)
            action = self.rollout_policy(state)
            action_index = self.getActionIndex(action)
            next_state = self.model(state, torch.from_numpy(action).unsqueeze(0)).detach()
            next_action = self.rollout_policy(next_state)
            next_action_index = self.getActionIndex(next_action)
            target = self.q_value_function[next_action_index](next_state).detach()
            rewardListLen = len(rewardList)
            for j in range (rewardListLen):
                target *= self.gamma
                index = rewardListLen - j - 1
                target += rewardList[index]

            input = self.q_value_function[action_index](state)
            # print(target.shape, input.shape)
            loss = nn.MSELoss()(input, target)

            loss.backward()
            self.updateWeights(action_index)

    def plan2(self):
        for i in range (len(self.buffer)):
            start_state = self.buffer[i]
            state = torch.from_numpy(start_state.copy())
            state = state.unsqueeze(0)
            rewardList = []
            for j in range(self.planning_steps):
                action = self.rollout_policy(state)
                action_index = self.getActionIndex(action)
                next_state = self.model(state).detach()[:, action_index]
                is_terminal = np.array_equal(next_state.numpy()[0], self.goal)
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                # reward = self.reward_function(next_state.detach().numpy())
                rewardList.append(reward)
                if is_terminal:
                    break
                state = torch.from_numpy(np.copy(next_state))

            state = start_state
            state = torch.from_numpy(state).unsqueeze(0)
            action = self.rollout_policy(state)
            action_index = self.getActionIndex(action)

            next_state = self.model(state).detach()[:, action_index]

            next_action = self.rollout_policy(next_state)
            next_action_index = self.getActionIndex(next_action)
            target = self.q_value_function(next_state).detach()[0, next_action_index]
            rewardListLen = len(rewardList)
            for j in range (rewardListLen):
                target *= self.gamma
                index = rewardListLen - j - 1
                target += rewardList[index]

            input = self.q_value_function(state)[0, action_index]
            loss = nn.MSELoss()(input, target)

            loss.backward()
            self.updateWeights(action_index)

    def rollout_policy(self, state):
        # random policy
        # action = self.action_list[int(np.random.rand() * self.num_actions)]
        v = []
        for i, action in enumerate(self.action_list):
            v.append(self.q_value_function[i](state).detach())
        action = self.action_list[np.argmax(v)]
        return action

    def trainModel(self, state, action, next_state):

        x_old = torch.from_numpy(state).float().unsqueeze(0)
        x_new = torch.from_numpy(next_state).float().unsqueeze(0)
        action = torch.from_numpy(action).float().unsqueeze(0)

        input = self.model(x_old, action)
        target = x_new
        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights()

    def trainModel2(self, state, action, next_state):
        x_old = torch.from_numpy(state).unsqueeze(0).float()
        x_new = torch.from_numpy(next_state).unsqueeze(0).float()
        action_index = self.getActionIndex(action)

        input = self.model(x_old)[:, action_index]  #B , num_action, state shape
        target = x_new

        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights()

