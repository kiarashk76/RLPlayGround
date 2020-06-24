from ..Experiments.BaseExperiment import BaseExperiment
from ..Envs.GridWorldBase import GridWorld
from ..Agents.BaseDynaAgent import BaseDynaAgent
from ..Agents.RandomDynaAgent import RandomDynaAgent
from ..Agents.ForwardDynaAgent import ForwardDynaAgent
from ..Agents.BackwardDynaAgent import BackwardDynaAgent
from ..Networks.ModelNN.StateTransitionModel import preTrainBackward, preTrainForward
from .. import utils,config
import numpy as np
import torch

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class GridWorldExperiment(BaseExperiment):
    def __init__(self, agent, env, params=None):
        if   params is None:
            params = {'render': True}
        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.visit_counts = self.createVisitCounts(env)
        self.num_samples = 0
        super().__init__(agent, env)

    def start(self):
        self.num_steps = 0
        s = self.environment.start()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        return (obs, self.last_action)

    def step(self):
        (reward, s, term) = self.environment.step(self.last_action)
        self.num_samples += 1
        obs = self.observationChannel(s)
        self.total_reward += reward
        if self._render_on and self.num_episodes > 95:
            self.environment.render(values= self.calculateModelError(self.agent.model['forward'],
                                                                     self.environment.transitionFunctionBackward)[1])
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
        return s

    def recordTrajectory(self, s, a, r, t):
        self.updateVisitCounts(s, a)

    def agentPolicyEachState(self):
        all_states = self.environment.getAllStates()

        for state in all_states:
            agent_state = self.agent.agentState(state)
            policy = self.agent.policy(agent_state, greedy= True)
            print(policy)

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
                values[(pos), tuple(a)] = round(
                    self.agent.vf['q']['network'](s_torch).detach()[:,self.agent.getActionIndex(a)].item(),3)
        return values

    def calculateModelError(self, model, true_transition_function):
        sum = 0.0
        cnt = 0.0
        states = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        errors = {} # np.zeros([len(states), len(actions)])
        for i, s in enumerate(states):
            for j, a in enumerate(actions):
                true_state = true_transition_function(s, a)
                pos = self.environment.stateToPos(s)
                pred_state = self.agent.rolloutWithModel(s, a, model)
                mse = (np.square(true_state - pred_state)).mean()

                if ((pos), tuple(a)) in self.visit_counts:
                    errors[(pos), tuple(a)] = round(mse, 3), self.visit_counts[(pos), tuple(a)]
                else:
                    errors[(pos), tuple(a)] = round(mse, 3), 0

                sum += mse
                cnt += 1
        avg = sum / cnt
        return avg, errors

    def updateVisitCounts(self, s, a):
        if a is None:
            return 0
        pos = self.environment.stateToPos(s)
        self.visit_counts[(pos), tuple(a)] += 1

    def createVisitCounts(self, env):
        visit_count = {}
        for state in env.getAllStates():
            for action in env.getAllActions():
                pos = env.stateToPos(state)
                visit_count[(pos, tuple(action))] = 0
        return visit_count



class RunExperiment():
    def __init__(self, model_type='forward',
                 pre_trained=True, use_pre_trained=False,
                 show_pre_trained_error_grid=False,
                 show_values_grid=False,
                 show_model_error_grid=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        if model_type != 'forward' and model_type != 'backward' and model_type != 'free':
            raise ValueError("model type not defined")
        self.model_type = model_type
        self.pre_trained = pre_trained
        self.use_pre_trained = use_pre_trained
        self.show_pre_trained_error_grid = show_pre_trained_error_grid
        self.show_values_grid = show_values_grid
        self.show_model_error_grid = show_model_error_grid
        print(self.device)


    def run_experiment(self):
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        num_steps_run_list = np.zeros([num_runs, num_episode], dtype = np.int)
        model_error_run_list = []
        model_error_run_num_samples = []
        pre_trained_plot_y_run_list = []
        pre_trained_plot_x_run_list = []
        for r in range(num_runs):
            env = GridWorld(params=config.empty_room_params)
            reward_function = env.rewardFunction
            goal = env.posToState((0, config._n - 1), state_type = 'full_obs')

            if self.pre_trained:
                if self.model_type == 'forward':
                    pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainForward(
                        env)
                    pre_trained_plot_y_run_list.append(pre_trained_plot_y)
                    pre_trained_plot_x_run_list.append(pre_trained_plot_x)
                elif self.model_type == 'backward':
                    pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainBackward(
                        env)
                    pre_trained_plot_y_run_list.append(pre_trained_plot_y)
                    pre_trained_plot_x_run_list.append(pre_trained_plot_x)

            if self.show_pre_trained_error_grid and self.pre_trained:
                agent = RandomDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                         'gamma': 1.0, 'epsilon': 1.0,
                                         'reward_function': reward_function,
                                         'goal': goal,
                                         'device': self.device,
                                         'model': pre_trained_model,
                                         'training': False})
                experiment = GridWorldExperiment(agent, env)
                experiment.visit_counts = pre_trained_visit_counts
                if self.model_type == 'forward':
                    utils.draw_grid((config._n, config._n), (900, 900),
                                    state_action_values=experiment.calculateModelError(agent.model['forward'],
                                                                                       env.transitionFunction)[1],
                                    all_actions=env.getAllActions())
                elif self.model_type == 'backward':
                    utils.draw_grid((config._n, config._n), (900, 900),
                                    state_action_values=experiment.calculateModelError(agent.model['backward'],
                                                                                       env.transitionFunction)[1],
                                    all_actions=env.getAllActions())

            if self.model_type == 'forward':
                if self.use_pre_trained:
                    agent = ForwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                              'gamma': 1.0, 'epsilon': 0.01,
                                              'reward_function': reward_function,
                                              'goal': goal,
                                              'device': self.device,
                                              'model': pre_trained_model})
                else:
                    agent = ForwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                              'gamma': 1.0, 'epsilon': 0.01,
                                              'reward_function': reward_function,
                                              'goal': goal,
                                              'device': self.device,
                                              'model': None})
            elif self.model_type == 'backward':
                if self.use_pre_trained:
                    agent = BackwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                               'gamma': 1.0, 'epsilon': 0.01,
                                               'reward_function': reward_function,
                                               'goal': goal,
                                               'device': self.device,
                                               'model': pre_trained_model})
                else:
                    agent = BackwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                              'gamma': 1.0, 'epsilon': 0.01,
                                              'reward_function': reward_function,
                                              'goal': goal,
                                              'device': self.device,
                                              'model': None})
            elif self.model_type == 'free':
                agent = BaseDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                        'gamma':1.0, 'epsilon': 0.1,
                                        'reward_function': reward_function,
                                        'goal': goal,
                                        'device': self.device})

            experiment = GridWorldExperiment(agent, env)
            model_error_list = []
            model_error_num_samples = []

            for e in range(num_episode):

                print("starting episode ", e + 1)
                experiment.runEpisode(max_step_each_episode)

                if self.model_type == 'forward':
                    model_error = experiment.calculateModelError(agent.model['forward'], env.transitionFunction)[0]
                elif self.model_type == 'backward':
                    model_error = experiment.calculateModelError(agent.model['backward'], env.transitionFunctionBackward)[0]

                if self.model_type != 'free':
                    print("model error: ", model_error)
                    model_error_list.append(model_error)
                    model_error_num_samples.append(experiment.num_samples)

                num_steps_run_list[r, e] = experiment.num_steps

            model_error_run_list.append(model_error_list)
            model_error_run_num_samples.append(model_error_num_samples)

            if self.show_model_error_grid :
                if self.model_type == 'forward':
                    utils.draw_grid((config._n, config._n), (900, 900),
                                    state_action_values=experiment.calculateModelError(agent.model['forward'],
                                                                                       env.transitionFunctionBackward)[1],
                                    all_actions=env.getAllActions())
                elif self.model_type == 'backward':
                    utils.draw_grid((config._n, config._n), (900, 900),
                                    state_action_values=experiment.calculateModelError(agent.model['backward'],
                                                                                       env.transitionFunctionBackward)[1],
                                    all_actions=env.getAllActions())
            if self.show_values_grid:
                utils.draw_grid((config._n, config._n), (900, 900),
                                state_action_values=experiment.calculateValues(),
                                all_actions=env.getAllActions())


        mean_steps_list = np.mean(num_steps_run_list, axis=0)

        model_error_run_list = np.asarray(model_error_run_list)
        model_error_run_num_samples = np.asarray(model_error_run_num_samples)
        mean_model_error_list = np.mean(model_error_run_list, axis=0)
        mean_model_error_num_samples = np.mean(model_error_run_num_samples, axis=0)

        if self.pre_trained:
            pre_trained_plot_y_run_list = np.asarray(pre_trained_plot_y_run_list)
            pre_trained_plot_x_run_list = np.asarray(pre_trained_plot_x_run_list)
            mean_pre_trained_plot_y_run = np.mean(pre_trained_plot_y_run_list, axis=0)
            mean_pre_trained_plot_x_run = np.mean(pre_trained_plot_x_run_list, axis=0)
            utils.draw_plot(mean_pre_trained_plot_x_run, mean_pre_trained_plot_y_run, xlabel='num_samples', ylabel='model_error')

        utils.draw_plot(mean_model_error_num_samples, mean_model_error_list, xlabel='num_samples', ylabel='model_error', show=True)
        utils.draw_plot(range(len(mean_steps_list)), mean_steps_list, xlabel='episode_num', ylabel='num_steps', show=True)
