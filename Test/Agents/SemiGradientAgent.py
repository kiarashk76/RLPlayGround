from Test.Agents.BaseAgent import BaseAgent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

gamma = 1.0
step_size = 0.5
class SGAgent(BaseAgent):
    def __init__(self):
        self.V = VN(3, 1)

    def start(self, observation):

        self.prev_state = np.copy(self.agentState(observation))
        return self.policy(observation)

    def step(self, reward, observation):
        self.current_state = np.copy(self.agentState(observation))
        x_new = torch.from_numpy(self.current_state).unsqueeze(0)
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)

        # td_error = reward + gamma * self.V(x_new.float()) - self.V(x_old.float())
        # loss = 0.5 * torch.sqrt(reward + gamma * self.V(x_new.float()) - self.V(x_old.float()))
        # print(td_error)
        self.V.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(self.V(x_old.float()), reward + gamma * self.V(x_new.float()))
        loss.backward()

        # self.V(x_old.float()).backward()

        for f in self.V.parameters():
            print(f.grad.data)
            f.data.sub_(f.grad.data *step_size)

        self.prev_state = np.copy(self.current_state)

        return self.policy(observation)

    def end(self, reward):
        x_old = torch.from_numpy(self.prev_state).unsqueeze(0)
        td_error = reward - self.V(x_old.float())
        print(td_error)
        self.V.zero_grad()
        self.V(x_old.float()).backward()
        for f in self.V.parameters():
            f.data.add_(f.grad.data * td_error *step_size)

    def agentState(self, observation):
        a = np.zeros(3)
        a[observation] = 1
        return a

    def policy(self, state):
        return 1
        a = np.random.randint(4)
        if a == 0:
            return 'U'
        if a == 1:
            return 'D'
        if a == 2:
            return 'R'
        if a == 3:
            return 'L'


class VN(nn.Module):

    def __init__(self, inputs, outputs):
        super(VN, self).__init__()
        self.fc1 = nn.Linear(inputs, outputs, bias=False)

        torch.nn.init.zeros_(self.fc1.weight)

    def forward(self, x):
        self.flatten1 = torch.flatten(x)
        return self.fc1(self.flatten1)