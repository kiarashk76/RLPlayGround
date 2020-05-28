from Test.Envs.BaseEnvironment import BaseEnvironment
import numpy as np

right = 1
left = -1
class Chain(BaseEnvironment):
    def init(self, num_states= 19):
        self.__num_states = num_states
        self.__state = num_states // 2

    def start(self, starting_state= None):
        if starting_state == None:
            self.__state = self.__num_states // 2
        else:
            self.__state = starting_state
        return self.__state

    def step(self, action):
        if self.__state not in range(self.__num_states):
            print("State out of range! Terminate and start Over")
            raise
        terminal = False
        reward = 0
        self.__state += action

        if self.__state == -1:
            terminal = True
            reward = -1
        elif self.__state == self.__num_states:
            terminal = True
            reward = 1
        return reward, self.__state, terminal

    def get_actions(self):
        return right, left

    def get_states_num(self):
        return self.__num_states

