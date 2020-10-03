# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 13:05:22
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-10-03 11:44:39

import itertools
import random
import copy
import json

class Player(object):
    """
    docstring for Player
    assume always x-player
    """

    def __init__(self, epsilon, alpha):
        self.epsilon = epsilon      # exploration rate
        self.alpha = alpha          # learning rate

        self.training_mode()

        self.states = {}            # dictionary that holds all the board states:values
        for state in generate_states('-xo'):
            if state_impossible(state):
                # do not store this state in the dictionary
                pass
            elif self.has_won(state):
                self.states[''.join(state)] = 1
            elif self.has_lost(state):
                self.states[''.join(state)] = 0
            else:
                self.states[''.join(state)] = 0.5
        

    def choose_action(self, board):
        # board         string of 9 elements '-', 'x' or 'o'

        # free positions
        free = [index for index, element in enumerate(list(board)) if element == '-']

        # exploration
        if random.random() < self.epsilon and self.is_training:
            # choose random action
            index = random.choice(free)
            # reduce exploration rate
            self.epsilon = max(0.999*self.epsilon, 0.05)
            return index

        # exploitation
        else:
            # next possible states
            next_states = []
            for i in free:
                next_board = copy.copy(list(board))
                next_board[i] = 'x'
                next_states.append(''.join(next_board))
            # get value for each possible next state
            next_values = [self.states.get(state) for state in next_states]
            # search which state maximizes the next values
            if self.is_training:
                index = random.choices(free, weights=next_values)[0]
            else:
                index = free[next_values.index(max(next_values))]
            
            return index


    def playing_mode(self):
        self.is_training = False


    def training_mode(self):
        self.is_training = True


    def update_values(self, board_old, board_new):
        # board         string of 9 elements '-', 'x' or 'o'
        
        val_old = self.states[board_old]
        val_new = self.states[board_new]

        self.states[board_old] -= self.alpha * ( val_new - 1 + val_old )
        
        
    def save_policy(self, file_name):
        file = open(file_name, "w")
        json.dump(self.states, file)
        file.close()
        
        
    def load_policy(self, file_name):
        file = open(file_name, "r")
        self.states = json.load(file)
        file.close()
        
    
    def reduce_alpha(self):
        self.alpha = max(0.999*self.alpha, 0.001)
        
        
    def set_alpha(self, a):
        self.alpha = a

        
    def has_won(self, board):
        # board         string of 9 elements '-', 'x' or 'o'
        
        won = False

        # positions with my sign
        pos_x = [index for index, element in enumerate(list(board)) if element == 'x']

        for w in win_comb:
            if set(w).issubset(set(pos_x)): 
                won = True
                break

        return won


    def has_lost(self, board):
    # board         string of 9 elements '-', 'x' or 'o'

        lost = False

        # positions with opponent's sign
        pos_o = [index for index, element in enumerate(list(board)) if element == 'o']

        for w in win_comb:
            if set(w).issubset(set(pos_o)): 
                lost = True
                break

        return lost


# winning combinations
win_comb = [ [0,1,2], 
             [3,4,5],
             [6,7,8],
             [0,3,6],
             [1,4,7],
             [3,5,8],
             [0,4,8],
             [2,4,6] ]


def generate_states(s):
    yield from itertools.product(*([s] * 9)) 


def state_impossible(state):
    num_x = len([index for index, element in enumerate(list(state)) if element == 'x'])
    num_o = len([index for index, element in enumerate(list(state)) if element == 'o'])

    return True if abs(num_x-num_o)>1 else False