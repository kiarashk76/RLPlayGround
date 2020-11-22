'''
train a DQN with MCTS
See if DQN agrees with DQN (saving the best path)
At what step DQN starts to work with MCTS
'''

from Colab.Experiments.GridWorldExperiment import RunExperiment as GridWorld_RunExperiment
from Colab.Experiments.MountainCarExperiment import RunExperiment as MountainCar_RunExperiment
from Colab.Experiments.CartPoleExperiment import RunExperiment as CartPole_RunExperiment

from Colab.Agents.BaseDynaAgent import BaseDynaAgent
from Colab.Agents.RandomDynaAgent import RandomDynaAgent
# from Colab.Agents.ForwardDynaAgent import ForwardDynaAgent
# from Colab.Agents.BackwardDynaAgent import BackwardDynaAgent
from Colab.Agents.TestAgent import BackwardDynaAgent, ForwardDynaAgent
from Colab.Envs.GridWorldRooms import GridWorldRooms
from Colab.Experiments.ExperimentObject import ExperimentObject
# from Colab.Agents.BaseMCTSAgent import BaseMCTSAgent
from Colab.Agents.MCTSAgent import BaseMCTSAgent
from Colab.Agents.UCBMCSTAgent import UCBMCTSAgent


if __name__ == '__main__':
    # experiment = RunExperiment()
    # experiment.run_experiment()

    # s_vf = [2 ** -6, 2 ** -5, 2 ** -7]
    # s_md = [2 ** -5, 2 ** -10, 2 ** -9]

    # agent_class_list = [UCBMCTSAgent]
    agent_class_list = [BaseMCTSAgent]


    show_pre_trained_error_grid = [False, False],
    show_values_grid = [False, False],
    show_model_error_grid = [False, False]

    s_vf_list = [0.001]
    s_md_list = [2 ** -9]

    c_list = [2**0.5]
    num_iteration_list = [10]
    simulation_depth_list = [100]
    num_simulation_list = [5]


    # model_list = [{'type':'forward', 'num_networks':1, 'layers_type':['fc'], 'layers_features':[128]},
    #               {'type': 'forward', 'num_networks': 2, 'layers_type': ['fc'], 'layers_features': [64]},
    #               {'type': 'forward', 'num_networks': 4, 'layers_type': ['fc'], 'layers_features': [32]}
    #               ]

    model_list = [{'type':None, 'num_networks':1, 'layers_type':['fc'], 'layers_features':[128]}]

    experiment = GridWorld_RunExperiment()

    experiment_object_list = []
    for agent_class in agent_class_list:
        for s_vf in s_vf_list:
            for model in model_list:
                for s_md in s_md_list:
                    for c in c_list:
                        for num_iteration in num_iteration_list:
                            for simulation_depth in simulation_depth_list:
                                for num_simulation in num_simulation_list:
                                    params = {'pre_trained': None,
                                              'vf_step_size': s_vf,
                                              'model': model,
                                              'model_step_size': s_md,
                                              'c': c,
                                              'num_iteration': num_iteration,
                                              'simulation_depth': simulation_depth,
                                              'num_simulation': num_simulation}
                                    obj = ExperimentObject(agent_class, params)
                                    experiment_object_list.append(obj)

    experiment.run_experiment(experiment_object_list)

'''
step_size: 1
AUC: 3193.2999999999993
AUC: 2999.600000000001

step_size: 0.5
AUC: 3116.2000000000007
AUC: 3155.7000000000003

step_size: 0.25
AUC: 2390.2
AUC: 2986.3

step_size: 0.125
AUC: 2271.899999999999
AUC: 2447.4000000000005

step_size: 0.0625
AUC: 892.9000000000003
AUC: 1165.4000000000003

step_size: 0.03125
AUC: 402.99999999999966
AUC: 552.7

step_size: 0.015625
AUC: 393.30000000000007
AUC: 1390.2

step_size: 0.0078125
AUC: 448.4999999999999
AUC: 577.4000000000003

step_size: 0.00390625
AUC: 349.49999999999994
AUC: 894.7000000000004

step_size: 0.001953125
AUC: 645.5999999999995
AUC: 855.6999999999997
'''