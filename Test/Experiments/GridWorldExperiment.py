from BaseExperiment import BaseExperiment
from GridWorldBase import GridWorld
from SemiGradTDAgent2 import SemiGradientTD
import numpy as np
import matplotlib.pyplot as plt
import torch

class GridWorldExperiment(BaseExperiment):
    def __init__(self, agent, env, params=None):
        if params is None:
            params = {'render': True}
        self._render_on = params['render']
        self.num_steps_lst = []
        super().__init__(agent, env)

    def start(self):
        self.num_steps = 0
        s = self.environment.start()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        return (obs, self.last_action)

    def step(self):
        (reward, s, term) = self.environment.step(self.last_action)
        obs = self.observationChannel(s)
        self.total_reward += reward
        if self._render_on and self.num_episodes > 4999:
            self.environment.render()
        if term:
            self.agent.end(reward)
            roat = (reward, obs, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, obs)
            roat = (reward, obs, self.last_action, term)

        self.recordTrajectory(roat[1], roat[2], roat[0], roat[3])
        return roat

    def runEpisode(self, max_steps=0):
        if self.num_episodes > 300:
            self.agent.greedy = True
            if self.num_steps > 100:
                print("hi")
                pass
            # printing each state action value
            # for state, coord in zip(env.getAllStates(state_type= 'full_obs'),
            #                         env.getAllStates(state_type= 'coord')):
                # print("state: ", coord)
                # state = torch.from_numpy(state).unsqueeze(0)
                # for action in env.getAllActions():
                #     action_index = self.agent.getActionIndex(action)
                #     print(action, self.agent.q_value_function[action_index](state))

                # if coord == (2, 0):
                #     state = torch.from_numpy(state).unsqueeze(0)
                #     for action in env.getAllActions():
                #         action_index = self.agent.getActionIndex(action)
                #         print(action, self.agent.q_value_function[action_index](state))

        is_terminal = False
        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        self.num_episodes += 1
        self.num_steps_lst.append(self.num_steps)
        print("num steps: ", self.num_steps)
        return is_terminal

    def observationChannel(self, s):
        return s

    def recordTrajectory(self, s, a, r, t):
        pass

    def agentPolicyEachState(self):
        all_states = self.environment.getAllStates()

        for state in all_states:
            agent_state = self.agent.agentState(state)
            policy = self.agent.policy(agent_state, greedy= True)
            print(policy)

    def draw_num_steps(self):
        # x axis values
        x = range(self.num_episodes)
        # corresponding y axis values
        y = self.num_steps_lst

        # plotting the points
        plt.plot(x, y)

        # naming the x axis
        plt.xlabel('episode number')
        # naming the y axis
        plt.ylabel('number of steps')

        # giving a title to my graph
        plt.title('Learning')

        # function to show the plot
        plt.show()

if __name__ == '__main__':
    # 0,6
    four_room_params = {'size': (7, 7), 'init_state': (6,0), 'state_mode': 'full_obs',
              'obstacles_pos': [(3,0),(3,2),(0,3),(2,3),(3,3),(4,3),(6,3),(3,4),(3,6)],
              'rewards_pos': [(0, 6)], 'rewards_value': [3],
              'terminals_pos': [(0, 6)], 'termination_probs': [1],
              'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
              'neighbour_distance': 0,
              'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
              'transition_randomness': 0.0,
              'window_size': (255, 255),
              'aging_reward': -1
              }
    empty_room_params_3d = {'size': (3, 3), 'init_state': (2, 0), 'state_mode': 'full_obs',
                        'obstacles_pos': [],
                        'rewards_pos': [(0, 2)], 'rewards_value': [0],
                        'terminals_pos': [(0, 2)], 'termination_probs': [1],
                        'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                        'neighbour_distance': 0,
                        'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
                        'transition_randomness': 0.0,
                        'window_size': (255, 255),
                        'aging_reward': -1
                        }
    empty_room_params_2d = {'size': (2, 2), 'init_state': (1, 0), 'state_mode': 'full_obs',
                         'obstacles_pos': [],
                         'rewards_pos': [(0, 1)], 'rewards_value': [0],
                         'terminals_pos': [(0, 1)], 'termination_probs': [1],
                         'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                         'neighbour_distance': 0,
                         'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
                         'transition_randomness': 0.0,
                         'window_size': (255, 255),
                         'aging_reward': -1
                         }

    env = GridWorld(params=empty_room_params_3d)
    agent = SemiGradientTD({'action_list': np.asarray(env.getAllActions()),
                            'gamma':1.0, 'step_size':0.01, 'epsilon': 0.1,
                            'batch_size': 1,})
    experiment = GridWorldExperiment(agent, env)



    # print(env.getAllStates())
    for i in range(500):
        print("starting episode ", i+1)
        experiment.runEpisode(200)
    experiment.draw_num_steps()