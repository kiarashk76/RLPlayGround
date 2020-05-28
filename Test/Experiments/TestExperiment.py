from BaseExperiment import BaseExperiment
from SemiGradientAgent import SGAgent
from GridWorld2D import GridWorld2D
from Chain import Chain
import numpy as np
import torch

def states():
    a = np.zeros((5,5))
    lst = []
    for i in range(5):
        for j in range(5):
            a[i,j] = 5
            lst.append(a)
            a = np.zeros((5,5))
    return lst

def reward(state):
    if state == (0,4):
        return 1
    else:
        return 0

def terminal(state):
    if state == (0,4):
        return True
    else:
        return False

params = {}
params['size'] = 5, 5
params['blocks'] = []
# params['blocks'] = [(0,2),(1,2),(2,2),(3,2)]
params['reward_function'] = reward
params['action_noise'] = 0.0
params['terminal_function'] = terminal


env = Chain()
env.init(3)
agent = SGAgent()
exp = BaseExperiment(agent, env)
for i in range(1000):
    print(i, " episode")
    exp.runEpisode(max_steps= 100)
print(agent.V.fc1.weight)
    # values = np.zeros(19)
    # for i in range(19):
    #     a = np.zeros(19)
    #     a[i] = 1
    #     values[i] = agent.V(torch.from_numpy(a).float())
    # print(values)

    # states = np.zeros((5, 5))
    # values = np.zeros((5, 5))
    # for i in range(5):
    #     for j in range(5):
    #         states[i, j] = 5
    #         values[i, j] = agent.V(torch.from_numpy(states).float())
    #         states = np.zeros((5, 5))
    # print(values)




