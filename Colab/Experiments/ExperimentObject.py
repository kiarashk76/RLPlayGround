class ExperimentObject:
    def __init__(self, agent_class, params={}):
        self.agent_class = agent_class
        self.pre_trained = params['pre_trained']
        self.vf_step_size = params['vf_step_size']
        self.model = params['model']
        self.model_step_size = params['model_step_size']
