from Colab.Experiments.GridWorldExperiment import RunExperiment2, TestExperiment
if __name__ == '__main__':
    # experiment = RunExperiment()
    # experiment.run_experiment()
    stepsize = [2 ** -i for i in range(10)]
    for counter, a in enumerate(stepsize):
        print(counter, "step size")
        experiment = RunExperiment2()
        experiment.run_experiment(a)

