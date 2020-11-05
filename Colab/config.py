from Colab.Agents.BaseDynaAgent import BaseDynaAgent
from Colab.Agents.RandomDynaAgent import RandomDynaAgent
# from Colab.Agents.ForwardDynaAgent import ForwardDynaAgent
# from Colab.Agents.BackwardDynaAgent import BackwardDynaAgent
from Colab.Agents.TestAgent import BackwardDynaAgent, ForwardDynaAgent
from Colab.Envs.GridWorldRooms import GridWorldRooms

# # experiment
num_runs = 1
num_episode = 50
max_step_each_episode = 50

# # environment
# empty room parameters
_n = 4
empty_room_params = \
    {'size': (_n, _n), 'init_state':(_n-1, 0), 'state_mode': 'coord', #init_state (_n-1, 0)
    'obstacles_pos': [],
    'rewards_pos': [(0, _n-1)], 'rewards_value': [1000],
    'terminals_pos': [(0, _n-1)], 'termination_probs': [1],
    'actions': [(0, -1), (-1, 0), (0, 1) , (1, 0)], # L, U, R, D
    'neighbour_distance': 0,
    'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
    'transition_randomness': 0.0,
    'window_size': (600, 600),
    'aging_reward': -10,
    }

n_room_params = \
    {'init_state': 'random' , 'state_mode': 'coord', #init_state (_n-1, 0)
    'house_shape': (2,2), 'rooms_shape': (3,3),
    'obstacles_pos': [],
    'rewards_value': [1],
    'termination_probs': [1],
    'actions': [(0, -1), (-1, 0), (0, 1) , (1, 0)], # L, U, R, D
    'neighbour_distance': 0,
    'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
    'transition_randomness': 0.0,
    'window_size': (900, 900),
    'aging_reward': -1,
    }

# four room parameters
four_room_params = \
    {'size': (7, 7), 'init_state': (6, 0), 'state_mode': 'coord',
    'obstacles_pos': [(3, 0), (3, 2), (0, 3), (2, 3), (3, 3), (4, 3), (6, 3), (3, 4), (3, 6)],
    'rewards_pos': [(0, 6),(1,3),(3,1),(5,3),(3,5)], 'rewards_value': [10,1,1,1,1],
    'terminals_pos': [(0, 6)], 'termination_probs': [1],
    'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)], # R, D, L, U
    'neighbour_distance': 0,
    'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
    'transition_randomness': 0.0,
    'window_size': (900, 900),
    'aging_reward': -1
    }


