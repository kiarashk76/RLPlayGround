from Test.Envs.GridWorldBase import GridWorld

def GridWorldND(n, params={}):
    empty_room_params_nd = {'size': (n, n), 'init_state': (n-1, 0), 'state_mode': 'full_obs',
                        'obstacles_pos': [],
                        'rewards_pos': [(0, n-1)], 'rewards_value': [0],
                        'terminals_pos': [(0, n-1)], 'termination_probs': [1],
                        'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                        'neighbour_distance': 0,
                        'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
                        'transition_randomness': 0.0,
                        'window_size': (255, 255),
                        'aging_reward': -1
                        }
    if 'transition_randomness' in params:
        empty_room_params_nd['transition_randomness'] = params['transition_randomness']
    if 'state_mode' in params:
        empty_room_params_nd['state_mode'] = params['state_mode']
    if 'rewards_value' in params:
        empty_room_params_nd['rewards_value'] = params['rewards_value']
    if 'init_state' in params:
        if params['init_state'] == 'random':
            empty_room_params_nd['init_state'] = 'random'
        else:
            raise ValueError('initial state not defined')

    return GridWorld(params=empty_room_params_nd)
