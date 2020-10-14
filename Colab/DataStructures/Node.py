import numpy as np

class Node:
    def __init__(self, state, par=None, val=0, from_par_reward=0, from_root_reward=0):
        self.state = state
        self.par = par
        self.children = []
        self.is_expanded = False
        self.val = val #state value function
        self.from_par_reward = from_par_reward
        self.from_root_reward = from_root_reward
        self.back_prop_type = 0 #0: average, 1: max
        if self.back_prop_type == 0:
            self.search_val = val
        else:
            self.search_val = val
        self.search_count = 1

    def expand(self, model, action_list, agent):
        non_terminal_children = []
        for action in action_list:
            child_state, is_terminal, reward = model(self.state, action)
            child_from_root_reward = self.from_root_reward + reward
            if is_terminal:
                child = Node(None, self, 0, reward, child_from_root_reward)
                child.search_val = 0
            else:
                child_val = agent.getStateValue(child_state)
                child = Node(child_state, self, child_val, reward, child_from_root_reward)
                non_terminal_children.append(child)
            self.children.append(child)
        self.is_expanded = True
        rand = int(np.random.rand() * len(non_terminal_children))
        return non_terminal_children[rand]

    def get_mcts_val(self):
        #change
        return self.search_val + self.from_root_reward
        # return self.from_root_reward