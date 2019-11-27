# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:42:08 2019

@author: morel
"""
import numpy as np
import matplotlib.pyplot as plt

class Grid:    # Environment
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]
    
    def set(self, rewards, actions):
        # rewards should be a dictionary of: (i, j): r, or: (row, col): reward
        # actions should be a dictionary of: (i, j): A or (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions
        
    def set_state(self, s):
        # The state s is the location of the player in the grid: s = (i, j)
        self.i = s[0]
        self.j = s[1]
        
    def current_state(self):
        return (self.i, self.j)
    
    def is_terminal(self, s):
        # A terminal state won't be in the actions dictionary (since it won't have any associated action)
        return s not in self.actions
    
    def move(self, action):
        # check if legal move first
        # Possible actions: U/D/L/R
        if action in self.actions[self.current_state()]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'D':
                self.j -= 1
        # return reward (if any)
        return self.rewards.get(self.current_state(), 0)
    
    def undo_move(self, action):
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'D':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert(self.current_state() in self.all_states())
        
    def game_over(self):
        # The game is over if we are in a state where no action is possible
        return self.current_state() not in self.actions
    
    def all_states(self):
        # Cast to a set to avoid repetition in states
        return set(list(self.rewards.keys()) + list(self.actions.keys()))
    

def standard_grid():
    # Define a grid that describes the reward for arriving at each state
    #     and possible actions at each state
    # The grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .   .   .   1
    # .   x   .  -1
    # s   .   .   .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0,0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('D', 'L', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('U', 'L', 'R'),
        (2, 3): ('U', 'L')
    }
    g.set(rewards, actions)
    return g
