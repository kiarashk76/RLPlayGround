from BaseAgent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def __init__(self, params = {}):
        self.reward_model = None
        self.state_transition_model = None
        self.action_list = params['action_list']
        self.action_size = self.action_list.shape[0]
        self.prev_state = None
        self.state = None


    def start(self, observation):
        '''
        :param observation: numpy array
        :return: action:
        '''
        self.prev_state = self.feature_vector(observation)
        self.prev_action = self.policy(self.prev_state)

        return self.prev_action

    def step(self, reward, observation):
        '''
        :param reward: int
        :param observation: numpy array
        :return: action
        '''

        self.state = self.feature_vector(observation)
        self.action = self.policy(self.state)


        self.prev_state = self.state
        self.prev_action = self.action
        return self.prev_action

    def end(self, reward):
        '''
        :param reward: int
        :return: none
        '''
        pass

    def policy(self, x):
        action = int(np.random.rand() * self.action_size)
        return self.action_list[action]

    def feature_vector(self, observation):
        return np.copy(observation)







