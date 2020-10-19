import numpy as np
import torch
import os
import itertools
import Colab.utils as utils, Colab.config as config
import random
import matplotlib.pyplot as plt

from Colab.Experiments.BaseExperiment import BaseExperiment
from Colab.Envs.mountain_car import MountainCarEnv

# from Colab.Agents.BaseDynaAgent import BaseDynaAgent
# from Colab.Agents.RandomDynaAgent import RandomDynaAgent
# from Colab.Agents.ForwardDynaAgent import ForwardDynaAgent
# from Colab.Agents.BackwardDynaAgent import BackwardDynaAgent
# from Colab.Agents.TestAgent import TestAgent

from Colab.Networks.ModelNN.StateTransitionModel import preTrainBackward, preTrainForward
from Colab.Datasets.TransitionDataGrid import data_store

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class MountainCarExperiment(BaseExperiment):
    def __init__(self, agent, env, device, params=None):
        if params is None:
            params = {'render': False}
        super().__init__(agent, env)

        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.num_samples = 0
        self.device = device

    def start(self):
        self.num_steps = 0
        s = self.environment.reset()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        return (obs, self.last_action)

    def step(self):
        (s, reward, term, _) = self.environment.step(self.last_action)
        self.num_samples += 1
        obs = self.observationChannel(s)
        self.total_reward += reward
        if self._render_on and self.num_episodes >= 0:
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
        is_terminal = False
        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        self.num_episodes += 1
        self.num_steps_to_goal_list.append(self.num_steps)
        print("num steps: ", self.num_steps)
        return is_terminal

    def observationChannel(self, s):
        return np.asarray(s)

    def recordTrajectory(self, s, a, r, t):
        pass

class RunExperiment():
    def __init__(self):
        gpu_counts = torch.cuda.device_count()
        # self.device = torch.device("cuda:"+str(random.randint(0, gpu_counts-1)) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)

    def run_experiment(self, experiment_object_list):
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        self.num_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)

        for i, obj in enumerate(experiment_object_list):
            for r in range(num_runs):
                print("starting runtime ", r + 1)
                env = MountainCarEnv()

                # initializing the agent
                agent = obj.agent_class({'action_list': np.asarray(range(env.action_space.n)),
                                         'gamma': 1.0,
                                         'epsilon': 0.0,
                                         'max_stepsize': obj.vf_step_size,
                                         'model_stepsize': obj.model_step_size,
                                         'reward_function': None,
                                         'goal': None,
                                         'device': self.device,
                                         'model': None,
                                         'true_bw_model': None,
                                         'true_fw_model': None,
                                         'c': obj.c})

                model_type = obj.model['type']
                if model_type is not None:
                    agent.model[model_type]['num_networks'] = obj.model['num_networks']
                    agent.model[model_type]['layers_type'] = obj.model['layers_type']
                    agent.model[model_type]['layers_features'] = obj.model['layers_features']

                # initialize experiment
                experiment = MountainCarExperiment(agent, env, self.device)

                for e in range(num_episode):
                    print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    self.num_steps_run_list[i, r, e] = experiment.num_steps

        with open('num_steps_run_list.npy', 'wb') as f:
            np.save(f, self.num_steps_run_list)
        self.show_num_steps_plot()

    def pretrain_model(self, model_type, env):
        pass

    def show_num_steps_plot(self):
        if False:
            for a in range(self.num_steps_run_list.shape[0]):
                agent_name = self.agents[a].name
                for r in range(self.num_steps_run_list.shape[1]):
                    utils.draw_plot(range(len(self.num_steps_run_list[a, r])), self.num_steps_run_list[a, r],
                                    xlabel='episode_num', ylabel='num_steps', show=True,
                                    label=agent_name, title='run_number ' + str(r + 1))
        if False:
            for r in range(self.num_steps_run_list.shape[1]):
                for a in range(self.num_steps_run_list.shape[0]):
                    agent_name = self.agents[a].name
                    utils.draw_plot(range(len(self.num_steps_run_list[a, r])), self.num_steps_run_list[a, r],
                                    xlabel='episode_num', ylabel='num_steps', show=False,
                                    label=agent_name, title='run_number ' + str(r + 1))
                plt.show()

        if False:
            color = ['blue', 'orange', 'green']
            for a in range(self.num_steps_run_list.shape[0]):
                agent_name = self.agents[a].name
                average_num_steps_run = np.mean(self.num_steps_run_list[a], axis=0)
                std_err_num_steps_run = np.std(self.num_steps_run_list[a], axis=0)
                AUC = sum(average_num_steps_run)
                print("AUC:", AUC, agent_name)
                utils.draw_plot(range(len(average_num_steps_run)), average_num_steps_run,
                                std_error=std_err_num_steps_run,
                                xlabel='episode_num', ylabel='num_steps', show=False,
                                label=agent_name + str(a), title='average over runs',
                                sub_plot_num='4' + '1' + str(a + 1), color=color[a])

                utils.draw_plot(range(len(average_num_steps_run)), average_num_steps_run,
                                std_error=std_err_num_steps_run,
                                xlabel='episode_num', ylabel='num_steps', show=False,
                                label=agent_name + str(a), title='average over runs',
                                sub_plot_num=414)

            # plt.savefig('')
            plt.show()