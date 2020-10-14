from Colab.Experiments.GridWorldExperiment import RunExperiment
from Colab.Agents.BaseDynaAgent import BaseDynaAgent
from Colab.Agents.RandomDynaAgent import RandomDynaAgent
# from Colab.Agents.ForwardDynaAgent import ForwardDynaAgent
# from Colab.Agents.BackwardDynaAgent import BackwardDynaAgent
from Colab.Agents.TestAgent import BackwardDynaAgent, ForwardDynaAgent
from Colab.Envs.GridWorldRooms import GridWorldRooms
from Colab.Experiments.ExperimentObject import ExperimentObject
from Colab.Agents.BaseMCTSAgent import BaseMCTSAgent
from Colab.Agents.UCBMCSTAgent import UCBMCTSAgent


if __name__ == '__main__':

    agent_class_list = [UCBMCTSAgent]
    # agent_class_list = [BaseDynaAgent]


    show_pre_trained_error_grid = [False, False],
    show_values_grid = [False, False],
    show_model_error_grid = [False, False]

    # value function step size
    s_vf_list = [2 ** -i for i in range(2, 11)]

    # model step size
    s_md_list = [2 ** -9]


    #mcts parameters
    c_list = [1.0]
    num_iteration_list = [10]
    simulation_depth_list = [1, 5, 10, 20, 50, 100]
    num_simulation_list = [1]

    # model_list = [{'type':'forward', 'num_networks':1, 'layers_type':['fc'], 'layers_features':[128]},
    #               {'type': 'forward', 'num_networks': 2, 'layers_type': ['fc'], 'layers_features': [64]},
    #               {'type': 'forward', 'num_networks': 4, 'layers_type': ['fc'], 'layers_features': [32]}
    #               ]

    model_list = [{'type':'forward', 'num_networks':1, 'layers_type':['fc'], 'layers_features':[128]}]

    experiment = RunExperiment()

    experiment_object_list = []
    for agent_class in agent_class_list:
        for s_vf in s_vf_list:
            for model in model_list:
                for s_md in s_md_list:
                    for c in c_list:
                        for num_iteration in num_iteration_list:
                            for simulation_depth in simulation_depth_list:
                                for num_simulation in num_simulation_list:
                                    params = {'pre_trained':None,
                                              'vf_step_size':s_vf,
                                              'model':model,
                                              'model_step_size':s_md,
                                              'c': c,
                                              'num_iteration': num_iteration,
                                              'simulation_depth': simulation_depth,
                                              'num_simulation': num_simulation}
                                    obj = ExperimentObject(agent_class, params)
                                    experiment_object_list.append(obj)


    experiment.run_experiment(experiment_object_list)
