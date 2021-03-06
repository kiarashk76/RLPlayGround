import numpy as np
import torch
import os
import itertools
import Colab.utils as utils, Colab.config as config
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from Colab.Experiments.BaseExperiment import BaseExperiment
from Colab.Envs.GridWorldBase import GridWorld
from Colab.Envs.GridWorldRooms import GridWorldRooms
# from Colab.Agents.ForwardErrorDynaAgent import ForwardErrorDynaAgent as ForwardDynaAgent
from Colab.Envs.mountain_car import MountainCarEnv

# from Colab.Agents.BaseDynaAgent import BaseDynaAgent
# from Colab.Agents.RandomDynaAgent import RandomDynaAgent
# from Colab.Agents.ForwardDynaAgent import ForwardDynaAgent
# from Colab.Agents.BackwardDynaAgent import BackwardDynaAgent
# from Colab.Agents.TestAgent import TestAgent

from Colab.Networks.ModelNN.StateTransitionModel import preTrainBackward, preTrainForward
from Colab.Datasets.TransitionDataGrid import data_store

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class GridWorldExperiment(BaseExperiment):
    def __init__(self, agent, env, device, params=None):
        if params is None:
            params = {'render': False}
        super().__init__(agent, env)

        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.visit_counts = self.createVisitCounts(env)
        self.num_samples = 0
        self.device = device

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

        if self._render_on and self.num_episodes >= 0:
            self.environment.render()
            # self.environment.render(values=self.calculateValues())
            # self.environment.render (values= self.modelErrorCalculatedByAgent(self.agent.model_error))
            # self.environment.render(values= self.calculateModelError(self.agent.model['forward'],
            #                                                          self.environment.transitionFunction)[1])
            # self.environment.render(values= self.calculateModelError(self.agent.model['backward'],
            #                                                          self.environment.transitionFunction)[1])
            # self.environment.render(values=self.calculateModelError(self.agent.model['backward'],
            #                                                         self.environment.transitionFunctionBackward)[1])

            # train, test = data_store(self.environment)
            # self.environment.render(values=self.calculateModelErrorError(self.agent.model['backward'],
            #                                                              test)[1])
        if term:
            self.agent.end(reward)
            roat = (reward, obs, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, obs)
            roat = (reward, obs, self.last_action, term)

        self.recordTrajectory(roat[1], roat[2], roat[0], roat[3])
        return roat

    def runEpisode(self, max_steps=0, print_steps=True):
        is_terminal = False
        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        # while ((max_steps == 0) or (self.num_steps < max_steps)):
        #     rl_step_result = self.step()
        #     is_terminal = rl_step_result[3]
        #     if is_terminal:
        #         self.start()

        self.num_episodes += 1
        self.num_steps_to_goal_list.append(self.num_steps)
        if print_steps:
            print("num steps: ", self.num_steps)
        return is_terminal

    def observationChannel(self, s):
        return np.asarray(s)

    def recordTrajectory(self, s, a, r, t):
        self.updateVisitCounts(s, a)

    def agentPolicyEachState(self):
        all_states = self.environment.getAllStates()

        for state in all_states:
            agent_state = self.agent.getStateRepresentation(state)
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
                s_torch = self.agent.getStateRepresentation(s)
                values[(pos), tuple(a)] = round(self.agent.getStateActionValue(s_torch, a, vf_type='q').item(), 3)
                    # self.agent.vf['q']['network'](s_torch).detach()[:,self.agent.getActionIndex(a)].item(),3)
        return values

    def modelErrorCalculatedByAgent(self, model_error):
        states = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        errors = {}
        for s in states:
            pos = self.environment.stateToPos(s)
            for a in actions:
                s_torch = self.agent.getStateRepresentation(s)
                errors[(pos), tuple(a)] = round(self.agent.getModelError(s_torch, a, model_error).item(), 3)
        return errors

    def calculateModelError2(self, model, true_transition_function):
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

    def calculateModelError(self, model, true_transition_function):
        sum = 0.0
        cnt = 0.0
        obss = self.environment.getAllStates()
        actions = self.environment.getAllActions()
        errors = {}  # np.zeros([len(states), len(actions)])
        for i, obs in enumerate(obss):
            state = self.agent.getStateRepresentation(obs)
            for j, a in enumerate(actions):
                true_state = self.agent.getStateRepresentation(true_transition_function(obs, a))
                pos = self.environment.stateToPos(obs)
                pred_state = self.agent.rolloutWithModel(state, a, model)[0]
                is_terminal = self.agent.rolloutWithModel(state, a, model)[1]

                assert pred_state.shape == true_state.shape, 'pred_state and true_state have different shapes'
                mse = torch.mean((true_state - pred_state) ** 2)
                if ((pos), tuple(a)) in self.visit_counts:
                    errors[(pos), tuple(a)] = round(float(is_terminal.data.cpu().numpy()),3), \
                                              round(float(mse.data.cpu().numpy()), 3), \
                                              self.visit_counts[(pos), tuple(a)]
                else:
                    errors[(pos), tuple(a)] = round(float(is_terminal.data.cpu().numpy()),3), \
                                              round(float(mse.data.cpu().numpy()), 3), \
                                              0

                sum += mse
                cnt += 1
        avg = sum / cnt
        return avg, errors

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
            obs, action, next_obs, reward = data
            if type == 'forward':
                next_state = self.agent.getStateRepresentation(next_obs)
                state = self.agent.getStateRepresentation(obs)
                true_state = self.agent.getStateRepresentation(true_transition_function(obs, action, state_type='coord'))

                err = np.inf
                for i in range(self.agent.model[type]['num_networks']):
                    pred_state = self.agent.rolloutWithModel(state, action, model, net_index=i)
                    assert pred_state.shape == next_state.shape, 'pred_state and true_state have different shapes'
                    temp_err = torch.mean((next_state - pred_state) ** 2)
                    if temp_err < err:
                        err = temp_err

            elif type == 'backward':
                state = self.agent.getStateRepresentation(obs)
                next_state = self.agent.getStateRepresentation(next_obs)
                prev_state = self.agent.rolloutWithModel(next_state, action, model)

                # assert pred_state.shape == next_obs.shape, 'pred_state and true_state have different shapes'
                err = torch.mean((state - prev_state) ** 2)

            else:
                raise ValueError('type is not defined')
            sum += err
        mse = sum / len(test_data)
        return mse
    def calculateModelErrorError(self, model, test_data, type='backward', true_transition_function=None):
        sum = 0.0
        errors = {}
        for data in test_data:
            obs, action, next_obs, reward = data
            if type == 'forward':
                raise NotImplementedError
                next_state = self.agent.getStateRepresentation(next_obs)
                state = self.agent.getStateRepresentation(obs)
                true_state = self.agent.getStateRepresentation(
                    true_transition_function(obs, action, state_type='coord'))
                pred_state = self.agent.rolloutWithModel(state, action, model)[0]

                assert pred_state.shape == next_state.shape, 'pred_state and true_state have different shapes'
                err = torch.mean((next_state - pred_state) ** 2)

            elif type == 'backward':
                state = self.agent.getStateRepresentation(obs)
                next_state = self.agent.getStateRepresentation(next_obs)
                prev_state = self.agent.rolloutWithModel(next_state, action, model)
                # assert pred_state.shape == next_obs.shape, 'pred_state and true_state have different shapes'
                true_err = torch.mean((state - prev_state) ** 2)
                pred_err = self.agent.calculateError(next_state, action, self.agent.error_network)
                err = (true_err - pred_err) ** 2
            else:
                raise ValueError('type is not defined')
            sum += err

            pos = self.environment.stateToPos(obs)
            if ((pos), tuple(action)) in self.visit_counts:
                errors[(pos), tuple(action)] = round(float(err.data.cpu().numpy()), 3), \
                                          self.visit_counts[(pos), tuple(action)]
            else:
                errors[(pos), tuple(action)] = round(float(err.data.cpu().numpy()), 3), \
                                          0

        mse = sum / len(test_data)
        return mse, errors

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
    def __init__(self):
        gpu_counts = torch.cuda.device_count()
        # self.device = torch.device("cuda:"+str(random.randint(0, gpu_counts-1)) if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.show_pre_trained_error_grid = config.show_pre_trained_error_grid
        # self.show_values_grid = config.show_values_grid
        # self.show_model_error_grid = config.show_model_error_grid

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)

    def run_experiment(self, experiment_object_list):
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        self.num_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        # ****
        self.simulation_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        self.consistency = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        # ****
        self.model_error_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.float)
        self.agent_model_error_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.float)
        self.model_error_samples = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)

        for i, obj in tqdm(enumerate(experiment_object_list)):
            pre_trained_plot_y_run_list = []
            pre_trained_plot_x_run_list = []
            for r in range(num_runs):
                print("starting runtime ", r+1)
                # env = GridWorld(params=config.empty_room_params)
                env = GridWorldRooms(params=config.n_room_params)

                train, test = data_store(env)
                reward_function = env.rewardFunction
                goal = np.asarray(env.posToState((0, config._n - 1), state_type='coord'))

                # Pre-train the model
                pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = \
                    self.pretrain_model(obj.pre_trained, env)
                pre_trained_plot_y_run_list.append(pre_trained_plot_y)
                pre_trained_plot_x_run_list.append(pre_trained_plot_x)

                # initializing the agent
                agent = obj.agent_class({'action_list': np.asarray(env.getAllActions()),
                                       'gamma': 0.99, 'epsilon': 0.1,
                                       'max_stepsize': obj.vf_step_size,
                                       'model_stepsize': obj.model_step_size,
                                       'reward_function': reward_function,
                                       'goal': goal,
                                       'device': self.device,
                                       'model': pre_trained_model,
                                       'true_bw_model': env.transitionFunctionBackward,
                                       'true_fw_model': env.fullTransitionFunction,
                                       'c': obj.c,
                                       'num_iteration': obj.num_iteration,
                                       'simulation_depth': obj.simulation_depth,
                                       'num_simulation': obj.num_simulation,})

                model_type = obj.model['type']
                if model_type is not None:
                    agent.model[model_type]['num_networks'] = obj.model['num_networks']
                    agent.model[model_type]['layers_type'] = obj.model['layers_type']
                    agent.model[model_type]['layers_features'] = obj.model['layers_features']

                #initialize experiment
                experiment = GridWorldExperiment(agent, env, self.device)

                for e in range(num_episode):
                    print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    self.num_steps_run_list[i, r, e] = experiment.num_steps


                    if agent.name == 'DQNMCTSAgent':
                        self.simulation_steps_run_list[i, r, e] = self.simulate_dqn(agent.policy, agent.true_model,
                                                                                    env.start(), env.getAllActions())
                        self.consistency[i, r, e] = agent.action_consistency / experiment.num_steps
                    if e % 100 == 0:
                        mean = np.mean(self.num_steps_run_list[0], axis=0)
                        plt.plot(mean[0:e])
                        plt.show()
                    if agent.name != 'BaseDynaAgent' and agent.name != 'BaseMCTSAgent' and agent.name != 'DQNMCTSAgent':
                        model_type = list(agent.model.keys())[0]
                        # agent_model_error = experiment.calculateModelErrorError(agent.model[model_type],
                        #                                     test,
                        #                                     type=str(model_type),
                        #                                     true_transition_function=env.transitionFunction)[0]

                        model_error = experiment.calculateModelErrorWithData(agent.model[model_type],
                                                                         test,
                                                                         type=str(model_type),
                                                                         true_transition_function=env.transitionFunction)
                        self.model_error_list[i, r, e] = model_error
                        # self.agent_model_error_list[agent_counter, r, e] = agent_model_error
                        self.model_error_samples[i, r, e] = experiment.num_samples


                # *********
                # model_type = list(agent.model.keys())[0]
                # utils.draw_grid((config._n, config._n), (900, 900),
                #                 state_action_values=experiment.calculateModelErrorError(agent.model[model_type],
                #                                             test,
                #                                             type=str(model_type),
                #                                             true_transition_function=env.transitionFunction)[1],
                #                 all_actions=env.getAllActions(),
                #                 obstacles_pos=env.get_obstacles_pos())
                # *********

        self.show_model_error_plot()
        # self.show_agent_model_error_plot()
        # with open('sim_num_steps_run_list.npy', 'wb') as f:
        #     np.save(f, self.simulation_steps_run_list)
        with open('num_steps_run_list_random.npy', 'wb') as f:
            np.save(f, self.num_steps_run_list)
        with open('model_error_run4.npy', 'wb') as f:
            np.save(f, self.model_error_list)
            np.save(f, self.model_error_samples)
        self.show_num_steps_plot()

    def calculate_model_error(self, agent, env):
        error = {}
        for s in env.getAllStates(state_type='coord'):
            state = agent.getStateRepresentation(s)
            for a in env.getAllActions():
                # true_next_state = env.transitionFunctionBackward(s, a)
                true_next_state = env.transitionFunction(s, a)

                distance_pred = []
                for i in range(agent.model['forward']['num_networks']):
                    distance_pred.append(torch.dist(agent.rolloutWithModel(state, a, agent.model['forward'], net_index=i),
                                                    torch.from_numpy(true_next_state).float()))
                # print(distance_pred)
                pos = env.stateToPos(s, state_type='coord')
                model_index = distance_pred.index(min(distance_pred))
                if ((pos),tuple(a)) not in error:
                    # error[(pos), tuple(a)] = round(distance_pred[0].item(),3), round(distance_pred[1].item(),3), round(distance_pred[2].item(),3)
                    # error[(pos), tuple(a)] = str(model_index), round(distance_pred[model_index].item(), 3)
                    try:
                        error[(pos), tuple(a)] = str(round(distance_pred[model_index].item(), 3)) + "\n" + \
                                                 str(agent.counter[(pos), tuple(a)])

                        # error[(pos), tuple(a)] = str(model_index) + "\n" + \
                        #                          str(agent.counter[(pos), tuple(a)])
                    except:
                        error[(pos), tuple(a)] = str(round(distance_pred[model_index].item(), 3)) + "\n" + \
                                                 str(0)
                        # error[(pos), tuple(a)] = str(model_index) + "\n" + \
                        #                          str((0, 0, 0))

        utils.draw_grid((env.grid_size[0], env.grid_size[1]), (900, 900),
                        state_action_values=error,
                        all_actions=env.getAllActions(),
                        obstacles_pos=env.get_obstacles_pos())

    def pretrain_model(self, model_type, env):
        if model_type == 'forward':
            pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainForward(
                env, self.device)
        elif model_type == 'backward':
            pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x = preTrainBackward(
                env, self.device)
        elif model_type is None:
            return None, None, None, None
        else:
            raise ValueError("model type not defined")

        return pre_trained_model, pre_trained_visit_counts, pre_trained_plot_y, pre_trained_plot_x

    def show_num_steps_plot(self):
        if False:
            for a in range(self.num_steps_run_list.shape[0]):
                agent_name = self.agents[a].name
                for r in range(self.num_steps_run_list.shape[1]):
                    utils.draw_plot(range(len(self.num_steps_run_list[a,r])), self.num_steps_run_list[a,r],
                            xlabel='episode_num', ylabel='num_steps', show=True,
                            label=agent_name, title='run_number '+str(r+1))
        if False:
            for r in range(self.num_steps_run_list.shape[1]):
                for a in range(self.num_steps_run_list.shape[0]):
                    agent_name = self.agents[a].name
                    utils.draw_plot(range(len(self.num_steps_run_list[a,r])), self.num_steps_run_list[a,r],
                            xlabel='episode_num', ylabel='num_steps', show=False,
                            label=agent_name, title='run_number '+str(r+1))
                plt.show()

        if False:
            color=['blue','orange','green']
            for a in range(self.num_steps_run_list.shape[0]):
                agent_name = self.agents[a].name
                average_num_steps_run = np.mean(self.num_steps_run_list[a], axis=0)
                std_err_num_steps_run = np.std(self.num_steps_run_list[a], axis=0)
                AUC = sum(average_num_steps_run)
                print("AUC:", AUC, agent_name)
                utils.draw_plot(range(len(average_num_steps_run)), average_num_steps_run,
                        std_error = std_err_num_steps_run,
                        xlabel='episode_num', ylabel='num_steps', show=False,
                        label=agent_name + str(a), title= 'average over runs',
                        sub_plot_num='4'+'1' + str(a+1), color=color[a])

                utils.draw_plot(range(len(average_num_steps_run)), average_num_steps_run,
                                std_error=std_err_num_steps_run,
                                xlabel='episode_num', ylabel='num_steps', show=False,
                                label=agent_name + str(a), title='average over runs',
                                sub_plot_num=414)

            # plt.savefig('')
            plt.show()

    def show_model_error_plot(self):

        if False:
            for a in range(self.model_error_list.shape[0]):
                agent_name = self.agents[a].name
                for r in range(self.model_error_list.shape[1]):
                    utils.draw_plot(range(len(self.model_error_samples[a,r])), self.model_error_list[a,r],
                            xlabel='num_samples', ylabel='model_error', show=True,
                            label=agent_name, title='run_number '+str(r+1))
        if False:
            for r in range(self.model_error_list.shape[1]):
                for a in range(self.model_error_list.shape[0]):
                    agent_name = self.agents[a].name
                    utils.draw_plot(range(len(self.model_error_list[a,r])), self.model_error_list[a,r],
                            xlabel='num_samples', ylabel='model_error', show=False,
                            label=agent_name, title='run_number '+str(r+1))
                plt.show()

        if False:
            color=['blue','orange','green']
            for a in range(self.model_error_list.shape[0]):
                agent_name = self.agents[a].name
                average_model_error_run = np.mean(self.model_error_list[a], axis=0)
                std_err_model_error_run = np.std(self.model_error_list[a], axis=0)
                AUC = sum(average_model_error_run)
                print("AUC:", AUC, agent_name)
                utils.draw_plot(range(len(average_model_error_run)), average_model_error_run,
                        std_error = std_err_model_error_run,
                        xlabel='num_samples', ylabel='model_error', show=False,
                        label=agent_name + str(a), title= 'average over runs',
                        sub_plot_num='4'+'1' + str(a+1), color=color[a])

                utils.draw_plot(range(len(average_model_error_run)), average_model_error_run,
                                std_error=std_err_model_error_run,
                                xlabel='num_samples', ylabel='model_error', show=False,
                                label=agent_name + str(a), title='average over runs',
                                sub_plot_num=414)

            # plt.savefig('')
            plt.show()

    def show_agent_model_error_plot(self):
        for a in range(self.agent_model_error_list.shape[0]):
            agent_name = self.agents[a].name
            for r in range(self.agent_model_error_list.shape[1]):
                utils.draw_plot(range(len(self.model_error_samples[a, r])), self.agent_model_error_list[a, r],
                                xlabel='num_samples', ylabel='agent_model_error', show=True,
                                label=agent_name, title='run_number ' + str(r + 1))

    def simulate_dqn(self, policy, model, init_state, action_list):
        num_steps = 0
        is_terminal = False
        state = init_state
        while num_steps < config.max_step_each_episode and not is_terminal:
            num_steps += 1
            torch_state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            action_ind = policy(torch_state).item()
            action = action_list[action_ind]
            next_state, is_terminal, reward = model(state, action)
            state = next_state
        return num_steps


    #
    # def calculate_model_error(self):
    #     if self.model_type[i] == 'forward':
    #         model_error = experiment.calculateModelErrorWithData(agent.model['forward'], test,
    #                                                              type='forward',
    #                                                              true_transition_function=env.transitionFunction)
    #         model_error_list.append(model_error)
    #         # model_error, model_val_error = experiment.calculateModelErrorNStep(agent.model['forward'], env.transitionFunction, vf=agent.getStateActionValue)
    #         # model_error2 = experiment.calculateModelErrorNStep(agent.model['forward'], env.transitionFunction, n=10)
    #
    #     elif self.model_type[i] == 'backward':
    #         model_error = experiment.calculateModelErrorWithData(agent.model['backward'], test,
    #                                                              type='backward',
    #                                                              true_transition_function=env.transitionFunctionBackward)
    #         model_error_list.append(model_error)
    #         # model_error = experiment.calculateModelErrorWithData(agent.model['backward'], test, type='backward')
    #         # model_error = experiment.calculateModelError(agent.model['backward'], env.transitionFunctionBackward)[0]
    #
    # def show_model_error_grid(self):
    #     # if self.show_pre_trained_error_grid and self.pre_trained:
    #     agent = RandomDynaAgent({'action_list': np.asarray(env.getAllActions()),
    #                              'gamma': 1.0, 'epsilon': 1.0,
    #                              'reward_function': reward_function,
    #                              'goal': goal,
    #                              'device': self.device,
    #                              'model': pre_trained_model,
    #                              'training': False})
    #     experiment = GridWorldExperiment(agent, env, self.device)
    #     experiment.visit_counts = pre_trained_visit_counts
    #     if self.model_type == 'forward':
    #         utils.draw_grid((config._n, config._n), (900, 900),
    #                         state_action_values=experiment.calculateModelError(agent.model['forward'],
    #                                                                            env.transitionFunction)[1],
    #                         all_actions=env.getAllActions(),
    #                         obstacles_pos=env.get_obstacles_pos())
    #     elif self.model_type == 'backward':
    #         utils.draw_grid((config._n, config._n), (900, 900),
    #                         state_action_values=experiment.calculateModelError(agent.model['backward'],
    #                                                                            env.transitionFunction)[1],
    #                         all_actions=env.getAllActions(),
    #                          obstacles_pos=env.get_obstacles_pos())