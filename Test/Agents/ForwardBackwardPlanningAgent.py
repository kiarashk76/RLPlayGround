from Test.Agents.BaseAgent import BaseAgent
from Test.Networks.StateValueFunction import StateVFNN
from Test.Networks.ModelNN.StateTransitionModel import StateTransitionModel
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from Test.Networks.ModelNN.StateTransitionModel import preTrainForwad
from Test.Networks.ModelNN.StateTransitionModel import preTrainBackward

class ForwardBackwardPlannerAgent(BaseAgent):
    def __init__(self, params = {}):
        self.q_value_function = None
        self.action_list = params['action_list']
        self.actions_shape = self.action_list.shape[1]
        self.num_actions = self.action_list.shape[0]
        self.prev_state = None
        self.state = None
        self.gamma = params['gamma']
        self.step_size = params['step_size']
        self.model_step_size = params['model_step_size']
        self.epsilon = params['epsilon']
        self.batch_size = params['batch_size']
        self.vf_layers_type = ['fc']
        self.vf_layers_features = [32]

        self.fw_model_layers_type = ['fc']
        self.fw_model_layers_features = [32]

        self.bw_model_layers_type = ['fc']
        self.bw_model_layers_features = [32]

        self.greedy = False
        self.planning_steps = 2
        self.buffer_size = 1
        self.buffer = []

        if params['pre_trained_model'] :
            self.fw_state_transition_model = preTrainForwad()
            self.bw_state_transition_model = preTrainBackward()
        else:
            self.fw_state_transition_model = None
            self.bw_state_transition_model = None

        self.reward_function = params['reward_function']
        self.goal = params['goal']
        self.device = params['device']

        # default `log_dir` is "runs" - we'll be more specific here
        # self.writer = SummaryWriter('runs/')


    def start(self, observation):
        '''
        :param observation: numpy array
        :return: action:
        '''

        self.prev_state = self.agentState(observation)
        if self.q_value_function is None:
            nn_state_shape = (self.batch_size,) + self.prev_state.shape
            self.q_value_function = []
            for i in range(len(self.action_list)):
                v = StateVFNN(nn_state_shape, self.vf_layers_type, self.vf_layers_features)
                v.to(self.device)
                self.q_value_function.append(v)

        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        self.prev_action = self.policy(x_old, greedy= self.greedy)


        # initiliazing the forward and backward model
        if self.fw_state_transition_model is None:
            self.fw_state_transition_model = \
                StateTransitionModel((self.batch_size,) + self.prev_state.shape,
                                 (self.batch_size,) + self.prev_action.shape,
                                 self.fw_model_layers_type,
                                 self.fw_model_layers_features,
                                 action_layer_num=2)
            self.fw_state_transition_model.to(self.device)

        if self.bw_state_transition_model is None:
            self.bw_state_transition_model = \
                StateTransitionModel((self.batch_size,) + self.prev_state.shape,
                                 (self.batch_size,) + self.prev_action.shape,
                                 self.bw_model_layers_type,
                                 self.bw_model_layers_features,
                                 action_layer_num=2)
            self.bw_state_transition_model.to(self.device)

        return self.prev_action

    def step(self, reward, observation):

        '''
        :param reward: int
        :param observation: numpy array
        :return: action
        '''
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(self.prev_state)

        self.state = self.agentState(observation)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        x_new = torch.from_numpy(self.state).unsqueeze(0)
        self.action = self.policy(x_new, greedy= self.greedy)


        action_index = self.getActionIndex(self.action)
        prev_action_index = self.getActionIndex(self.prev_action)

        target = reward + self.gamma * self.q_value_function[action_index](x_new).detach()
        input = self.q_value_function[prev_action_index](x_old)
        loss = nn.MSELoss()(input, target)

        loss.backward()
        self.updateWeights(prev_action_index)

        self.prev_state = self.state
        self.prev_action = self.action


        self.trainForwardModel(self.prev_state, self.prev_action, self.state)
        self.trainBackwardModel(self.prev_state, self.prev_action, self.state)
        self.trainBackWardForwardConsistency(self.prev_state, self.prev_action)

        self.planForward()
        self.planBackward()


        # self.writer.add_scalar('loss', loss.item())
        # self.writer.close()

        return self.prev_action

    def end(self, reward):
        '''
        :param reward: int
        :return: none
        '''
        print("reached the goal!!!")
        target = torch.tensor(reward).unsqueeze(0).unsqueeze(0)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)

        prev_action_index = self.getActionIndex(self.prev_action)

        input = self.q_value_function[prev_action_index](x_old)
        loss = nn.MSELoss()(input.float(), target.float())
        loss.backward()
        self.updateWeights(prev_action_index)

    def policy(self, agent_state, greedy = False):
        if np.random.rand() > self.epsilon or greedy:
            v = []
            for i, action in enumerate(self.action_list):
                v.append(self.q_value_function[i](agent_state))
            action = self.action_list[np.argmax(v)]
        else:
            action = self.action_list[int(np.random.rand() * self.num_actions)]

        return action

    def agentState(self, observation):
        return np.copy(observation)

    def updateWeights(self, action_index):
        for f in self.q_value_function[action_index].parameters():
            f.data.sub_(self.step_size * f.grad.data)
        self.q_value_function[action_index].zero_grad()

    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if list(a) == list(action):
                return i
        raise ValueError("action is not defined")

    def planForward(self):
        for i in range (len(self.buffer)):
            state = self.buffer[i]
            state = torch.from_numpy(state).unsqueeze(0)
            next_state = None
            action = self.rollout_policy(state)
            # action_index = self.getActionIndex(action)
            next_action = None
            for j in range(self.planning_steps):
                next_state = self.fw_state_transition_model(state, torch.from_numpy(action).unsqueeze(0))
                next_action = self.rollout_policy(next_state)
                next_action_index = self.getActionIndex(next_action)
                is_terminal = np.array_equal(next_state.detach().numpy()[0], self.goal)
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                # reward = self.reward_function(next_state.detach().numpy())
                target = reward
                if not is_terminal:
                    target += self.gamma * self.q_value_function[next_action_index](next_state).detach()
                action_index = self.getActionIndex(action)

                input = self.q_value_function[action_index](state)
                loss = nn.MSELoss()(input, target)

                loss.backward()
                self.updateWeights(action_index)
                if is_terminal:
                    break

                state = next_state.detach()
                action = next_action

    def planBackward(self):
        for i in range (len(self.buffer)):
            next_state = self.buffer[i]
            next_state = torch.from_numpy(next_state).unsqueeze(0)
            state = None
            next_action = self.policy(next_state)
            next_action_index = self.getActionIndex(next_action)

            for j in range(self.planning_steps):
                action = self.rollout_policy(next_state)
                action_index = self.getActionIndex(action)
                state = self.bw_state_transition_model(next_state, torch.from_numpy(action).unsqueeze(0))

                is_terminal = np.array_equal(next_state.detach().numpy()[0], self.goal)
                if is_terminal:
                    reward = 3
                else:
                    reward = -1
                # reward = self.reward_function(next_state.detach().numpy())
                target = reward
                next_action_index = self.getActionIndex(next_action)

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
        # action = self.action_list[int(np.random.rand() * self.num_actions)]
        v = []
        for i, action in enumerate(self.action_list):
            v.append(self.q_value_function[i](state))
        action = self.action_list[np.argmax(v)]
        return action

    def trainForwardModel(self, state, action, next_state):
        x_old = torch.from_numpy(state).unsqueeze(0).float()
        x_new = torch.from_numpy(next_state).unsqueeze(0).float()
        input = self.fw_state_transition_model(x_old, action)
        target = x_new
        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights(self.fw_state_transition_model)

    def trainBackwardModel(self, state, action, next_state):
        x_old = torch.from_numpy(state).unsqueeze(0).float()
        x_new = torch.from_numpy(next_state).unsqueeze(0).float()
        input = self.bw_state_transition_model(x_new, action)
        target = x_old
        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights(self.bw_state_transition_model)

    def trainBackWardForwardConsistency(self, state, action):
        x_old = torch.from_numpy(state).unsqueeze(0).float()
        one_step_forward = self.fw_state_transition_model(x_old, action)
        input = self.bw_state_transition_model(one_step_forward, action)
        target = x_old
        loss = nn.MSELoss()(input, target)
        loss.backward()
        self.updateModelWeights(self.fw_state_transition_model)
        self.updateModelWeights(self.bw_state_transition_model)

    def updateModelWeights(self, model):
        for f in model.parameters():
            f.data.sub_(self.model_step_size * f.grad.data)
        model.zero_grad()