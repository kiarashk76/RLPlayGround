from Test.Envs.GridWorldBase import GridWorld
from collections import namedtuple
import csv
from torch.utils.data import Dataset, DataLoader
import random


def data_store(train_test_split = 0.5):
    transition = namedtuple('transition', ['state', 'action', 'next_state', 'reward'])

    empty_room_params_3d = {'size': (3, 3), 'init_state': (2, 0), 'state_mode': 'full_obs',
                            'obstacles_pos': [],
                            'rewards_pos': [(0, 2)], 'rewards_value': [1],
                            'terminals_pos': [(0, 2)], 'termination_probs': [1],
                            'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                            'neighbour_distance': 0,
                            'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
                            'transition_randomness': 0.0,
                            'window_size': (255, 255),
                            'aging_reward': -1
                            }
    env = GridWorld(params=empty_room_params_3d)
    all_states = env.getAllStates(state_type='full_obs')
    random.shuffle(all_states)
    train_states = all_states[0 : int(len(all_states) * train_test_split)]
    train_states = all_states

    test_state = all_states[int(len(all_states) * train_test_split) : ]
    all_actions = env.getAllActions()
    train_list = []
    test_list = []

    for state in train_states:
        for action in all_actions:
            next_state = env.transitionFunction(state, action, state_type='full_obs')
            reward = env.rewardFunction(next_state, state_type='full_obs')
            t = transition(state, action, next_state, reward)
            train_list.append(t)

    for state in test_state:
        for action in all_actions:
            next_state = env.transitionFunction(state, action, state_type='full_obs')
            reward = env.rewardFunction(next_state, state_type='full_obs')
            t = transition(state, action, next_state, reward)
            test_list.append(t)
    return train_list, test_list

    # with open('3dTransition_train.csv', 'w') as f:
    #     w = csv.writer(f)
    #     w.writerows([(t.state, t.action, t.next_state, t.reward) for t in transition_list])

def data_loader():
    with open('3dTransition.csv', 'r') as f:
        r = csv.reader(f)
        for row in r:
            state, action, next_state, reward = row
            yield state, action, next_state, reward

