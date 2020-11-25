import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
from Envs.cartpole import CartPoleEnv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb,os

from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN4
from Colab.Experiments.CartPoleExperiment import RunExperiment as CartPole_RunExperiment, CartPoleExperiment
from Colab.Experiments.GridWorldExperiment import GridWorldExperiment

from Colab.Experiments.ExperimentObject import ExperimentObject
import config
from Colab.Envs.GridWorldBase import GridWorld

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# env = gym.make('CartPole-v1')#.unwrapped
env = GridWorld(config.four_room_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

'''
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
'''

env.start()

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1
BUFFER_SIZE = 4096

# Get number of actions from gym action space
# n_actions = env.action_space.n
n_actions = 4

# memory = ReplayMemory(BUFFER_SIZE)

steps_done = 0


action_list = config.four_room_params['actions']
episode_durations = []




#**********************************************
class agent():
    def __init__(self):
        self.policy_net = StateActionVFNN4([1, 2], n_actions,
                                      ['fc', 'fc'],
                                      [64, 64],
                                      3).to(device)
        self.target_net = StateActionVFNN4([1, 2], n_actions,
                                      ['fc', 'fc'],
                                      [64, 64],
                                      3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.transition_buffer = []
        self.transition_buffer_size = 4096
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def start(self, observation):
        self.state = torch.tensor(observation, dtype=torch.float32).view(1, -1)
        self.action = self.select_action(self.state)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return action_list[self.action.item()]
        # return self.action.item()

    def step(self, reward, observation):
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32).view(1, -1)

        transition = Transition(self.state, self.action, next_state, reward)
        self.updateTransitionBuffer(transition)

        self.state = next_state
        self.optimize_model()
        self.action = self.select_action(self.state)
        return action_list[self.action.item()]
        # return self.action.item()


    def end(self, reward):
        reward = torch.tensor([reward], device=device)
        next_state = None
        transition = Transition(self.state, self.action, next_state, reward)
        self.updateTransitionBuffer(transition)
        self.optimize_model()

    def optimize_model(self):
        if len(self.transition_buffer) < BATCH_SIZE:
            return


        transitions = self.getTransitionFromBuffer(BATCH_SIZE)
        # print(transitions)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None,
                      batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > 0.1:  # eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]],
                                device=device, dtype=torch.long)


    def getTransitionFromBuffer(self, n):
        # both model and value function are using this buffer
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.sample(self.transition_buffer, k=n)

    def updateTransitionBuffer(self, transition):
        self.transition_buffer.append(transition)
        if len(self.transition_buffer) > self.transition_buffer_size:
            self.removeFromTransitionBuffer()

    def removeFromTransitionBuffer(self):
        self.transition_buffer.pop(0)

num_runs = config.num_runs
num_episode = config.num_episode
max_step_each_episode = config.max_step_each_episode
num_steps_run_list = np.zeros([num_runs, num_episode], dtype=np.int)

for r in range(num_runs):
    print("starting runtime ", r + 1)
    env = GridWorld()

    # initializing the agent
    agent = agent()

    # initialize experiment
    experiment = GridWorldExperiment(agent, env, device)

    for e in range(num_episode):
        print("starting episode ", e + 1)
        experiment.runEpisode(max_step_each_episode)
        num_steps_run_list[r, e] = experiment.num_steps
        if e % 100 == 0:
            mean = np.mean(num_steps_run_list, axis=0)
            plt.plot(mean[0:e])
            plt.show()
exit(0)
#**********************************************