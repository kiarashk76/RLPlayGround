from Colab.Envs.GridWorldBase import GridWorld
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import config
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class StateTransitionModel(nn.Module):
    def __init__(self, state_shape):
        super(StateTransitionModel, self).__init__()

        self.layer1 = nn.Linear(state_shape, 64)
        self.layer2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, state_shape)

        self.vlayer1 = nn.Linear(state_shape, 64)
        self.vlayer2 = nn.Linear(64, 64)
        self.var = nn.Linear(64, state_shape * state_shape)

        torch.nn.init.xavier_uniform_(self.vlayer1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.vlayer2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.layer1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.layer2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)


    def forward(self, x):
        l = torch.relu(self.layer1(x))
        l = torch.relu(self.layer2(l))

        vl = torch.relu(self.vlayer1(x))
        vl = torch.relu(self.vlayer2(vl))

        mu = self.mu(l)
        var = torch.log(1 + torch.exp(self.var(vl)))

        return mu, var

def train_model(batch_states, batch_next_states, model, step_size, mu):
    x = torch.from_numpy(batch_states).float()
    y = torch.from_numpy(batch_next_states).float()
    mu = torch.from_numpy(mu).float()
    pred, var = model(x)
    assert pred.shape == x.shape, str(pred.shape) + str(var.shape) + str(x.shape)

    var = torch.reshape(var, (-1, var.shape[1]//2, var.shape[1]//2))
    var = var + 10**-6
    # pred = torch.reshape(pred, (pred.shape)+(1,))
    y = torch.reshape(y, (y.shape) + (1,))
    mu = torch.reshape(mu, (mu.shape) + (1,))

    t1 = torch.transpose((mu - y) + 10**-6, 1, 2)
    t2 = torch.pinverse(var)
    t3 = (mu-y) + 10**-6
    term1 = torch.matmul(t1,t2)
    term1 = torch.matmul(term1, t3)
    term2 = torch.log(torch.abs(torch.det(var)))
    term1 = term1[:,0,0]
    loss = torch.mean(term1 + term2)
    loss.backward()
    if math.isnan(loss):
        print(loss)
    optimizer = optim.Adam(model.parameters(), lr=step_size)
    # optimizer = optim.SGD(model.parameters(), lr=step_size)

    optimizer.step()
    optimizer.zero_grad()

if __name__ == '__main__':
    epochs = 2000
    batch_size = 8
    env = GridWorld(params=config.empty_room_params)
    states = np.asarray(env.getAllStates('coord'))
    actions = env.getAllActions()
    model = StateTransitionModel(states.shape[1])
    for e in tqdm(range(epochs)):
        for i in range(len(states)//batch_size):
            ind = np.random.choice(len(states), batch_size)
            batch_states = states[ind]
            batch_next_states = np.zeros([batch_size, states.shape[1]], dtype=int)
            mu = np.zeros([batch_size, states.shape[1]], dtype=int)

            for count, s in enumerate(batch_states):
                next_state, is_terminal, reward = env.fullTransitionFunction(s, actions[2], 'coord')
                batch_next_states[count] = next_state

                if np.equal(s,np.array([1, 1])).all():
                    next_state, is_terminal, reward= env.fullTransitionFunction(s, actions[1], 'coord')
                    next_state = np.array([100,100])
                mu[count] = next_state

            train_model(batch_states, batch_next_states, model, 0.001, mu)

    # test
    for state in states:
        state = torch.from_numpy(state).unsqueeze(0).float()
        mu, var = model(state)
        var = torch.reshape(var, (var.shape[1] // 2, var.shape[1] // 2))
        print(state, end=',')
        print(torch.trace(var))