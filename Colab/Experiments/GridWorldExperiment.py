import numpy as np
import torch
import os
import itertools
import Colab.utils as utils, Colab.config as config
import random
import matplotlib.pyplot as plt

from Colab.Experiments.BaseExperiment import BaseExperiment
from Colab.Envs.GridWorldBase import GridWorld
# from Colab.Agents.BaseDynaAgent import BaseDynaAgent
from Colab.Agents.ForwardErrorDynaAgent import ForwardErrorDynaAgent as BaseDynaAgent

from Colab.Agents.RandomDynaAgent import RandomDynaAgent
from Colab.Agents.ForwardDynaAgent import ForwardDynaAgent
from Colab.Agents.BackwardDynaAgent import BackwardDynaAgent
from Colab.Networks.ModelNN.StateTransitionModel import preTrainBackward, preTrainForward
from Colab.Datasets.TransitionDataGrid import data_store

os.environ['KMP_DUPLICATE_LIB_OK']='True'





class GridWorldExperiment(BaseExperiment):
    def __init__(self, agent, env, device, params=None):
        if   params is None:
            params = {'render': False}
        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.visit_counts = self.createVisitCounts(env)
        self.num_samples = 0
        self.device = device
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
        if self._render_on and self.num_episodes > 0:
            # self.environment.render()
            self.environment.render(values=self.calculateValues())
            # self.environment.render(values= self.calculateModelError(self.agent.model['backward'],
            #                                                          self.environment.transitionFunction)[1])
            # self.environment.render(values=self.calculateModelError(self.agent.model['backward'],
            #                                                         self.environment.transitionFunctionBackward)[1])
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
                s_torch = self.agent.getStateRepresentation(torch.from_numpy(s))
                values[(pos), tuple(a)] = round(self.agent.getStateActionValue(s_torch, a).item(), 3)
                    # self.agent.vf['q']['network'](s_torch).detach()[:,self.agent.getActionIndex(a)].item(),3)
        return values

    def calculateModelError(self, model, true_transition_function):
        sum = 0.0
        cnt = 0.0
        states = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        errors = {} # np.zeros([len(states), len(actions)])
        for i, s in enumerate(states):
            state_torch = torch.from_numpy(s).to(self.device)
            for j, a in enumerate(actions):
                true_state = torch.from_numpy(true_transition_function(s, a)).to(self.device)
                pos = self.environment.stateToPos(s)
                pred_state = self.agent.rolloutWithModel(state_torch, a, model)
                
                assert pred_state.shape == true_state.shape, 'pred_state and true_state have different shapes'
                mse = torch.mean((true_state - pred_state) ** 2)

                if ((pos), tuple(a)) in self.visit_counts:
                    errors[(pos), tuple(a)] = round(mse, 3), self.visit_counts[(pos), tuple(a)]
                else:
                    errors[(pos), tuple(a)] = round(mse, 3), 0

                sum += mse
                cnt += 1
        avg = sum / cnt
        return avg, errors

    def calculateModelError2(self, model, true_transition_function, n=1):
        sum = 0.0
        cnt = 0.0
        states = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        for i, s in enumerate(states):
            state_torch = torch.from_numpy(s).to(self.device)
            for j, a in enumerate(actions):
                true_state = torch.from_numpy(true_transition_function(s, a)).to(self.device)
                pred_state = self.agent.rolloutWithModel(state_torch, a, model)
                assert pred_state.shape == true_state.shape, 'pred_state and true_state have different shapes'

                mse = torch.mean((true_state - pred_state) ** 2)

                sum += mse
                cnt += 1
        avg = sum / cnt
        return avg
    
    def calculateModelErrorNStep(self, model, true_transition_function, vf=None, n=1):
        sum = 0.0
        sum_value = 0.0
        cnt = 0.0
        states = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        action_combination = random.choices([p for p in itertools.product(actions, repeat=n)], k=4)
        for i, s in enumerate(states):
            state_torch = torch.from_numpy(s).to(self.device)
            for action_seq in enumerate(action_combination):
                pred_state = state_torch
                true_state = s
                for a in action_seq[1]:
                    true_state = true_transition_function(true_state, a)
                    pred_state = self.agent.rolloutWithModel(pred_state, a, model)
                    assert pred_state.shape == true_state.shape, 'pred_state and true_state have different shapes'
                true_state = torch.from_numpy(true_state).to(self.device)
                v_true_state = vf(true_state)
                v_pred_state = vf(pred_state)
                
                mse_value = torch.mean((v_true_state - v_pred_state) ** 2)
                mse = torch.mean((true_state - pred_state) ** 2)

                sum += mse
                sum_value += mse_value
                cnt += 1
              
        avg = sum / cnt   
        avg_value = sum_value / cnt     
        return avg, avg_value

    def calculateModelErrorWithData(self, model, test_data, type='forward', true_transition_function=None):
        sum = 0.0
        for data in test_data:
            state, action, next_state, reward = data
            if type == 'forward':
                next_state = torch.from_numpy(next_state).to(self.device)
                true_state = torch.from_numpy(true_transition_function(state, action)).to(self.device)
                # if not torch.all(torch.eq(next_state, true_state):
                #   print("an")
                pred_state = self.agent.rolloutWithModel(torch.from_numpy(state).to(self.device), action, model)
                assert pred_state.shape == next_state.shape, 'pred_state and true_state have different shapes'
                err = torch.mean((next_state - pred_state) ** 2)

            elif type == 'backward':
                state = torch.from_numpy(state).to(self.device)
                pred_state = self.agent.rolloutWithModel(torch.from_numpy(next_state).to(self.device), action, model)
                assert pred_state.shape == next_state.shape, 'pred_state and true_state have different shapes'
                err = torch.mean((state - pred_state) ** 2)
            else:
                raise ValueError('type is not defined')
            sum += err
        mse = sum / len(test_data)
        return mse

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
    def __init__(self, random_agent=[False, False],
                 model_type=['free'],
                 pre_trained=[False, False], use_pre_trained=[False, False],
                 show_pre_trained_error_grid=[False, False],
                 show_values_grid=[False, False],
                 show_model_error_grid=[False, False]):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # if model_type != 'forward' and model_type != 'backward' and model_type != 'free':
        #     raise ValueError("model type not defined")
        self.model_type = model_type
        self.pre_trained = pre_trained
        self.use_pre_trained = use_pre_trained
        self.show_pre_trained_error_grid = show_pre_trained_error_grid
        self.show_values_grid = show_values_grid
        self.show_model_error_grid = show_model_error_grid
        self.random_agent = random_agent

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)


    def run_experiment(self):
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        

        mean_steps_list = []
        mean_model_error_list = []
        mean_model_error_num_samples = []
        for i in range(len(self.model_type)):
          num_steps_run_list = np.zeros([num_runs, num_episode], dtype = np.int)
          model_error_run_list = []
          model_val_error_run_list = []
          model_error_run_num_samples = []
          pre_trained_plot_y_run_list = []
          pre_trained_plot_x_run_list = []
          
          for r in range(num_runs):
            env = GridWorld(params=config.empty_room_params)
            reward_function = env.rewardFunction
            goal = env.posToState((0, config._n - 1), state_type = 'full_obs')
            train, test = data_store(env)
            # Pre-train the model
            if self.pre_trained[i]:
                if self.model_type[i] == 'forward':
                    pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainForward(
                        env, self.device)
                    pre_trained_plot_y_run_list.append(pre_trained_plot_y)
                    pre_trained_plot_x_run_list.append(pre_trained_plot_x)
                elif self.model_type[i] == 'backward':
                    pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainBackward(
                        env, self.device)
                    pre_trained_plot_y_run_list.append(pre_trained_plot_y)
                    pre_trained_plot_x_run_list.append(pre_trained_plot_x)

            # if self.show_pre_trained_error_grid and self.pre_trained:
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
                                    all_actions=env.getAllActions(),
                                    obstacles_pos=env.get_obstacles_pos())
                elif self.model_type == 'backward':
                    utils.draw_grid((config._n, config._n), (900, 900),
                                    state_action_values=experiment.calculateModelError(agent.model['backward'],
                                                                                       env.transitionFunction)[1],
                                    all_actions=env.getAllActions(),
                                    obstacles_pos=env.get_obstacles_pos())

            if self.random_agent[i]:
                agent = RandomDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                           'gamma': 1.0, 'epsilon': 0.01,
                                           'reward_function': reward_function,
                                           'goal': goal,
                                           'device': self.device,
                                           'model': None})
            else:
                if self.model_type[i] == 'forward':
                    if self.use_pre_trained[i]:
                        agent = ForwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                                  'gamma': 1.0, 'epsilon': 0.01,
                                                  'reward_function': reward_function,
                                                  'goal': goal,
                                                  'device': self.device,
                                                  'model': pre_trained_model,
                                                  'true_model': env.transitionFunction})
                    else:
                        agent = ForwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                                  'gamma': 1.0, 'epsilon': 0.01,
                                                  'reward_function': reward_function,
                                                  'goal': goal,
                                                  'device': self.device,
                                                  'model': None,
                                                  'true_model': env.transitionFunction})
                elif self.model_type[i] == 'backward':
                    if self.use_pre_trained[i]:
                        agent = BackwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                                   'gamma': 1.0, 'epsilon': 0.01,
                                                   'reward_function': reward_function,
                                                   'goal': goal,
                                                   'device': self.device,
                                                   'model': pre_trained_model,
                                                   'true_model': env.transitionFunctionBackward})
                    else:
                        agent = BackwardDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                                  'gamma': 1.0, 'epsilon': 0.01,
                                                  'reward_function': reward_function,
                                                  'goal': goal,
                                                  'device': self.device,
                                                  'model': None,
                                                  'true_model': env.transitionFunctionBackward})
                elif self.model_type[i] == 'free':
                    agent = BaseDynaAgent({'action_list': np.asarray(env.getAllActions()),
                                            'gamma':1.0, 'epsilon': 0.01,
                                            'reward_function': reward_function,
                                            'goal': goal,
                                            'model': None,
                                            'device': self.device})

            experiment = GridWorldExperiment(agent, env, self.device)
            model_error_list = []
            model_val_error_list = []
            model_error_num_samples = []

            for e in range(num_episode):

                print("starting episode ", e + 1)
                experiment.runEpisode(max_step_each_episode)

                if self.model_type[i] == 'forward':
                    model_error = experiment.calculateModelErrorWithData(agent.model['forward'], test, type='forward', true_transition_function= env.transitionFunction)
                    # model_error, model_val_error = experiment.calculateModelErrorNStep(agent.model['forward'], env.transitionFunction, vf=agent.getStateActionValue)
                    # model_error2 = experiment.calculateModelErrorNStep(agent.model['forward'], env.transitionFunction, n=10)

                elif self.model_type[i] == 'backward':
                    model_error = experiment.calculateModelErrorWithData(agent.model['backward'], test, type='backward')
                    # model_error = experiment.calculateModelError(agent.model['backward'], env.transitionFunctionBackward)[0]

                if self.model_type[i] != 'free':
                    print("model error: ", model_error)
                    model_error_list.append(model_error)
                    # model_val_error_list.append(model_val_error)
                    model_error_num_samples.append(experiment.num_samples)

                num_steps_run_list[r, e] = experiment.num_steps

            model_error_run_list.append(model_error_list)
            # model_val_error_run_list.append(model_val_error_list)
            model_error_run_num_samples.append(model_error_num_samples)
            
            utils.draw_plot(model_error_num_samples, model_error_list,
                          xlabel='num_samples', ylabel='model_error', show=True,
                          label=self.model_type[i], title=self.model_type[i])
            
            utils.draw_plot(range(len(num_steps_run_list[r])), num_steps_run_list[r],
                          xlabel='episode_num', ylabel='num_steps', show=True,
                          label=self.model_type[i], title=self.model_type[i])

            utils.draw_plot(range(len(experiment.agent.model_pred_error)), experiment.agent.model_pred_error,show=True)

            if self.show_model_error_grid[i] :
                if self.model_type[i] == 'forward':
                    utils.draw_grid((config._n, config._n), (900, 900),
                                    state_action_values=experiment.calculateModelError(agent.model['forward'],
                                                                                       env.transitionFunctionBackward)[1],
                                    all_actions=env.getAllActions(),
                                    obstacles_pos=env.get_obstacles_pos())
                    
                elif self.model_type[i] == 'backward':
                    utils.draw_grid((config._n, config._n), (900, 900),
                                    state_action_values=experiment.calculateModelError(agent.model['backward'],
                                                                                       env.transitionFunctionBackward)[1],
                                    all_actions=env.getAllActions(),
                                    obstacles_pos=env.get_obstacles_pos())
            if self.show_values_grid[i]:
                utils.draw_grid((config._n, config._n), (900, 900),
                                state_action_values=experiment.calculateValues(),
                                all_actions=env.getAllActions(),
                                obstacles_pos=env.get_obstacles_pos())


          mean_steps_list.append(np.mean(num_steps_run_list, axis=0))

          model_error_run_list = np.asarray(model_error_run_list)
          
          
          # model_val_error_run_list = np.asarray(model_val_error_run_list)
          model_error_run_num_samples = np.asarray(model_error_run_num_samples)
          mean_model_error_list.append(np.mean(model_error_run_list, axis=0))
          # mean_model_val_error_list = np.mean(model_val_error_run_list, axis=0)
          mean_model_error_num_samples.append(np.mean(model_error_run_num_samples, axis=0))

          if self.pre_trained[i]:
              pre_trained_plot_y_run_list = np.asarray(pre_trained_plot_y_run_list)
              pre_trained_plot_x_run_list = np.asarray(pre_trained_plot_x_run_list)
              mean_pre_trained_plot_y_run = np.mean(pre_trained_plot_y_run_list, axis=0)
              mean_pre_trained_plot_x_run = np.mean(pre_trained_plot_x_run_list, axis=0)
              utils.draw_plot(mean_pre_trained_plot_x_run, mean_pre_trained_plot_y_run,
                              xlabel='num_samples', ylabel='model_error',
                              label='pre_train_model', title=self.model_type)

          # utils.draw_plot(mean_model_error_num_samples, mean_model_val_error_list,
          #                 xlabel='num_samples', ylabel='model_val_error', show=False,
          #                 label='agent_model', title=self.model_type)
        
        for i in range(len(self.model_type)):
          utils.draw_plot(mean_model_error_num_samples[i], mean_model_error_list[i],
                          xlabel='num_samples', ylabel='model_error', show=False,
                          label=self.model_type[i], title=self.model_type)
        plt.show()
        for i in range(len(self.model_type)):
          utils.draw_plot(range(len(mean_steps_list[i])), mean_steps_list[i], xlabel='episode_num', ylabel='num_steps',
                          show=False, title=self.model_type, label=self.model_type[i])
        plt.show()