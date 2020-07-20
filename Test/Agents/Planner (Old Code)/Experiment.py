from ..Experiments.BaseExperiment import BaseExperiment
from ..Envs.GridWorldNbN import GridWorldND
from ..Envs.GridWorldBase import GridWorld
from ..Agents.ForwardPlannerAgent import ForwardPlannerAgent
from ..Agents.SemiGradTDAgent import SemiGradientTD
import numpy as np
import matplotlib.pyplot as plt
import torch

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class GridWorldExperiment(BaseExperiment):
    def __init__(self, agent, env, params=None):
        if params is None:
            params = {'render': False}
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
        if self._render_on and self.num_episodes > 0:
            self.environment.render(values= None)
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
            agent_state = self.agent.getStateRepresentation(state)
            policy = self.agent.policy(agent_state, greedy= True)
            print(policy)

    def draw_num_steps(self):
        # x axis values
        x = range(self.num_episodes)
        # corresponding y axis values
        y = self.num_steps_lst

        plt.ylim(0, 200)

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

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def calculateValues(self):
        states = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        values = {}
        for s in states:
            pos = self.environment.stateToPos(s)
            for a in actions:
                s_torch = torch.from_numpy(s).unsqueeze(0)
                a_torch = torch.from_numpy(a).unsqueeze(0)
                values[(pos), tuple(a)] = round(
                    self.agent.q_value_function(s_torch, a_torch).detach().item(),3)
        return values

    def calculateModelError(self):
        sum = 0.0
        cnt = 0.0
        states = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        for s in states:
            for a in actions:
                action_index = self.agent.getActionIndex(a)
                true_state = self.environment.transitionFunction(s, a)
                pred_state = self.agent.state_transition_model[action_index](torch.from_numpy(s).unsqueeze(0))
                pred_state = pred_state[0].detach().numpy()
                mse = (np.square(true_state - pred_state)).mean()
                sum += mse
                cnt += 1
        avg = sum / cnt
        print(avg)


class RunExperiment():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assuming that we are on a CUDA machine, this should print a CUDA device:

        print(self.device)

    def draw_num_steps(self, num_steps_list, name = ''):
        # x axis values
        x = range(len(num_steps_list))
        # corresponding y axis values
        y = num_steps_list

        plt.ylim(0, 200)

        # plotting the points
        plt.plot(x, y)

        # naming the x axis
        plt.xlabel('episode number')
        # naming the y axis
        plt.ylabel('number of steps')

        # giving a title to my graph
        plt.title('Learning')

        # function to show the plot
        plt.savefig(name)
        plt.show()

    def run_experiment(self):
        # 0,6
        four_room_params = {'size': (7, 7), 'init_state': (6, 0), 'state_mode': 'full_obs',
                            'obstacles_pos': [(3, 0), (3, 2), (0, 3), (2, 3), (3, 3), (4, 3), (6, 3), (3, 4), (3, 6)],
                            'rewards_pos': [(0, 6)], 'rewards_value': [3],
                            'terminals_pos': [(0, 6)], 'termination_probs': [1],
                            'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)], # R, D, L, U
                            'neighbour_distance': 0,
                            'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
                            'transition_randomness': 0.0,
                            'window_size': (900, 900),
                            'aging_reward': -1
                            }
        env_size = 4
        num_runs = 1
        num_episode = 100
        max_step_each_episode = 200

        num_steps_list = np.zeros([num_runs, num_episode], dtype = np.int)

        for r in range(num_runs):
            env = GridWorldND(n = env_size, params = {'transition_randomness': 0.0})
            # env = GridWorld(params = four_room_params)
            reward_function = env.rewardFunction
            goal = env.posToState((0, env_size - 1), state_type = 'full_obs')

            # agent = SemiGradientTD({'action_list': np.asarray(env.getAllActions()),
            #                        'gamma':1.0, 'step_size': 0.01, 'epsilon': 0.0,
            #                         'batch_size': 1,
            #                         'device': self.device})

            agent = ForwardPlannerAgent({'action_list': np.asarray(env.getAllActions()),
                                    'gamma':1.0, 'step_size':0.01, 'epsilon': 0.1,
                                    'batch_size': 1, 'reward_function': reward_function,
                                    'goal': goal, 'model_step_size': 0.05, 'model': None,
                                    'device': self.device})

            # agent = BackwardPlannerAgent({'action_list': np.asarray(env.getAllActions()),
            #                               'gamma': 1.0, 'step_size': 0.01, 'epsilon': 0.1,
            #                               'batch_size': 1, 'reward_function': reward_function,
            #                               'goal': goal, 'model_step_size': 0.05, 'model': None,
            #                               'device': self.device})

            # agent = ForwardBackwardPlannerAgent({'action_list': np.asarray(env.getAllActions()),
            #                               'gamma': 1.0, 'step_size': 0.01, 'epsilon': 0.1,
            #                               'batch_size': 1, 'reward_function': reward_function,
            #                               'goal': goal, 'model_step_size': 0.05,
            #                               'pre_trained_model': False,
            #                               'device': self.device})

            experiment = GridWorldExperiment(agent, env)
            for e in range(num_episode):
                print("starting episode ", e + 1)
                experiment.runEpisode(max_step_each_episode)
                print(experiment.count_parameters(agent.model))

                # experiment.calculateModelError()
                num_steps_list[r, e] = experiment.num_steps
            # experiment.draw_num_steps()
        mean_steps = np.mean(num_steps_list, axis = 0)

        np.save('backward.npy', num_steps_list)

        self.draw_num_steps(mean_steps)