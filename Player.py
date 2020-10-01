# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 13:05:22
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-09-17 16:51:59

import itertools
import random
import copy

class Player(object):
    """docstring for Player"""

    # def __init__(self, x_player, epsilon, alpha):
    def __init__(self, epsilon, alpha):
        # self.x_player = x_player
        self.epsilon = epsilon      # exploration rate
        self.alpha = alpha          # learning rate

        self.states = {}
        for state in generate_states('-xo'):
            if self.has_won(state):
                self.states[''.join(state)] = 1
            elif self.has_lost(state):
                self.states[''.join(state)] = 0
            else:
                self.states[''.join(state)] = 0.5
        

    def choose_action(self, board):
        # board (list) 9 elements with '-', 'x' or 'o'

        # free positions
        free = [index for index, element in enumerate(board) if element == '-']

        if random.random() < self.epsilon: 
        # exploration
            index = random.choice(free)

            self.epsilon *= 0.99
            self.epsilon = max(self.epsilon, 0.01)

            return index

        else:
        # exploitation

            # next possible states
            next_states = []
            
            for i in free:
                next_board = copy.copy(board)

                next_board[i] = 'x'
                next_states.append( ''.join(next_board) )


            # get value for each possible next state
            next_values = [self.states.get(state) for state in next_states]

            # search which state maximizes the next values
            max_value = max(next_values)
            index = free[next_values.index(max_value)]

            return index


    def update_values(self, board_old, board_new):

        val_old = self.states[''.join(board_old)]
        val_new = self.states[''.join(board_new)]

        self.states[''.join(board_old)] += self.alpha * ( val_new - val_old )
        self.alpha *= 0.99
        self.alpha = max(self.alpha, 0.1)


    def has_won(self, board):

        # sign = 'x' if self.x_player else 'o'
        won = False

        # positions with my sign
        pos_x = [index for index, element in enumerate(list(board)) if element == 'x']

        for w in win_comb:
            if set(w).issubset(set(pos_x)): won = True

        return won


    def has_lost(self, board):

        # sign = 'o' if self.x_player else 'x'
        lost = False

        # positions with my sign
        pos_o = [index for index, element in enumerate(list(board)) if element == 'o']

        for w in win_comb:
            if set(w).issubset(set(pos_o)): lost = True

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