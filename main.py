from Colab.Experiments.GridWorldExperiment import RunExperiment2, TestExperiment
if __name__ == '__main__':
    # experiment = RunExperiment()
    # experiment.run_experiment()

    # stepsize = [2 ** -8, 2 ** -5]
    # experiment = RunExperiment2()
    # experiment.run_experiment(stepsize)

    parameter_sweep = True
    if parameter_sweep:
        vf_stepsize = [2 ** -i for i in range(10)]
        model_stepsize = [2 ** -i for i in range(10)]
        for c1, s_vf in enumerate(vf_stepsize):
            for c2, s_md in enumerate(model_stepsize):
                experiment = RunExperiment2()
                experiment.run_experiment(s_vf, s_md)

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