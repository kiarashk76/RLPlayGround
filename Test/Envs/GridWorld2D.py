from BaseEnvironment import BaseEnvironment
import numpy as np

class GridWorld2D(BaseEnvironment):
    def init(self, params={}):
        self.size = params['size']
        self.blocks = params['blocks']
        self.rewards = params['reward_function']
        self.terminal = params['terminal_function']
        self.randomness = params['action_noise']
        self.grid = np.zeros([self.size[0], self.size[1]])

        self.put_blocks_on_grid()

    def start(self):
        x, y = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        while (x,y) in self.blocks:
            x, y = np.random.randint(0, self.size[0]) , np.random.randint(0, self.size[1])
        x, y = 0, 0
        self.state = x, y
        self.grid[self.state] = 5
        return self.grid

    def step(self, action):
        '''
        :param action: R, L, U, D
        :return: New State
        '''
        self.grid[self.state] = 0
        if action == 'R' :
            if self.state[1] + 1 < self.size[1]:
                if self.grid[self.state[0], self.state[1] + 1] != 1:
                    if np.random.binomial(1, self.randomness):
                        pass
                    else:
                        self.state= self.state[0], self.state[1] + 1

        elif action == 'L' :
            if self.state[1] - 1 > -1:
                if self.grid[self.state[0], self.state[1] - 1] != 1:
                    if np.random.binomial(1, self.randomness):
                        pass
                    else:
                        self.state= self.state[0], self.state[1] - 1

        elif action == 'U' :
            if self.state[0] - 1 > -1:
                if self.grid[self.state[0] - 1, self.state[1]] != 1:
                    if np.random.binomial(1, self.randomness):
                        pass
                    else:
                        self.state = self.state[0] - 1, self.state[1]

        elif action == 'D' :
            if self.state[0] + 1 < self.size[0]:
                if self.grid[self.state[0] + 1, self.state[1]] != 1:
                    if np.random.binomial(1, self.randomness):
                        pass
                    else:
                        self.state = self.state[0] + 1, self.state[1]

        reward = self.rewards(self.state)
        is_terminal = self.terminal(self.state)
        self.grid[self.state] = 5

        return reward, self.grid, is_terminal

    def put_blocks_on_grid(self):
        for block in self.blocks:
            self.grid[block] = 1

    def show(self):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                print(self.grid[x,y], end="  ")
            print("")






'''------------------'''

# def reward(state):
#     if state == (0,4):
#         return 1
#     else:
#         return 0
#
# def terminal(state):
#     if state == (0,4):
#         return True
#     else:
#         return False
#
# params = {}
# params['size'] = 5, 5
# params['blocks'] = [(0,2),(1,2),(2,2),(3,2)]
# params['reward_function'] = reward
# params['action_noise'] = 0.0
# params['terminal_function'] = terminal
#
#
# w = GridWorld2D()
# w.init(params)
# state = w.start()
# t = False
# while not t:
#     w.show()
#     act = input()
#     r, state, t = w.step(act)
#     print(r)



