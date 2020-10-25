# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 13:05:22
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-10-25 15:06:32

import numpy as np
import itertools
import random
import copy
import json
from tqdm import trange

extension = '.json'

class Player(object):
    """
    reinforcement learning agent for tic-tac-toe    
    """

    def __init__(self, board=None, **kwargs):
        # sign of the agent on the board
        self.sign           = kwargs.get('sign', 'x')

        # order for weighting the probability of each action
        self.ord            = kwargs.get('order', 1)  
        self.training_mode()

        # number of training games
        self.num_train      = kwargs.get('num_train', 1000)

        # learning rate
        self.lr             = kwargs.get('lr', 0.5)
        self.lr_min         = kwargs.get('lr_min', 0.01)
        self.lr_exp         = kwargs.get('lr_exp', 0.9)
        self.lr_red_steps   = kwargs.get('lr_red_steps', 1000)

        # dict with the state-value mappings
        self.state2value    = {}
        # initial value estimate
        self.init_val       = kwargs.get('init_val', 0.5)


    def choose_action(self, board):
        # free positions
        free_pos    = board.get_free_positions()
        # next possible states
        next_states = []
        for pos in free_pos:
            next_board = copy.deepcopy(board)
            next_board.add(sign=self.sign, row=pos[0], col=pos[1])
            next_states.append(next_board.get_state())
        # get value for each possible next state
        next_values = [self.state2value.get(state, self.init_val) for state in next_states]
        # randomly select action according to weights in next_values
        return random.choices(free_pos, weights=normalize(next_values, self.p))[0]


    def train(self, board, **kwargs):
        # some random actions for exploration
        self.training_mode()
        self.num_train = kwargs.get('num_train', self.num_train)

        for n in trange(self.num_train):
            # reset board for new game
            board.reset()
            # store the board states to backpropagate value when game ends
            board_states = []
            
            # assume that the game will be a draw
            reward = 0.5
            # play until board is full or someone won
            while not board.is_full():
                # player chooses an action
                action = self.choose_action(board)
                # update board
                board.add(sign=self.sign, row=action[0], col=action[1])
                # store board state for training later
                board_states.append(board.get_state())
                # check if player won
                if board.is_won(): 
                    reward = 1
                    break
                # if nobody won yet, inverse the board
                board.inverse()

            # update the state-value approximations
            self.update_values(reward, board_states)
            
            # reduce learning rate progressively
            if n % self.lr_red_steps == 0:
                self.reduce_lr()

    
    def update_values(self, reward, board_states):
        """
        update the values of the states in board_states

            reward          (float)             reward received at the end of the game

            board_states    (list of strings)   list of board states in which agent had to choose an action
        """
        board_states.reverse()
        # value of the last board state was the reward
        self.state2value[board_states[0]] = reward
        # backpropagate value of game to update the policy
        for state_new, state_old in pairwise(board_states):
            val_old = self.state2value.get(state_old, self.init_val)
            val_new = self.state2value.get(state_new, self.init_val)

            self.state2value[state_old] = val_old + self.lr * ( 1-val_new - val_old )


    def play(self, board):
        """
        given a board, choose an action and update the board

            board   (Board)     board that the agent has to play on
        """
        # no random actions any more
        self.playing_mode()
        # player chooses an action
        action = self.choose_action(board)
        # update board
        board.add(sign=self.sign, row=action[0], col=action[1])
        return board


    def playing_mode(self):
        """
        set the order for the weight normalisation to infinity
        """
        self.p = float('inf')


    def training_mode(self):
        """
        set the order for the weight normalisation to self.ord
        """
        self.p = self.ord
        
    
    def reduce_lr(self):
        """
        reduce the learning rate
        """
        self.lr = max(self.lr_exp*self.lr, self.lr_min)
        
        
    def set_lr(self, lr):
        """
        set the learning rate
        """
        self.lr = lr


    def get_value(self, state):
        """
        return the value of a specific state

            state   (string)    string of signs that represents a board state
        """
        return self.state2value.get(state)


    def get_state2value(self):
        """
        return the dictionary that contains the state-value mappings
        """
        return self.state2value


    def save_values(self, file_name):
        """
        save the dictionary that contains the state-value mappings

            file_name   (string)    name of the file without extension
        """
        file = open(file_name+extension, "w")
        json.dump(self.state2value, file)
        file.close()
        
        
    def load_values(self, file_name):
        """
        load the dictionary that contains the state-value mappings

            file_name   (string)    name of the file without extension
        """
        file = open(file_name+extension, "r")
        self.state2value = json.load(file)
        file.close()
        
    
    def save_args(self, file_args):
        """
        save all the parameters of the player

            file_args   (string)    name of the file without extension
        """
        file_vals = file_args + '-values'
        args = {
            # sign of the agent on the board
            'sign'          : self.sign,
            # order for weighting the probability of each action
            'ord'           : self.ord,
            # number of training games
            'num_train'     : self.num_train,
            # learning rate
            'lr'            : self.lr,
            'lr_min'        : self.lr_min,
            'lr_exp'        : self.lr_exp,
            'lr_red_steps'  : self.lr_red_steps,
            # initial value estimate
            'init_val'      : self.init_val,
            # file name with values
            'file_vals'     : file_vals
        }
        # save arguments
        file = open(file_args+extension, "w")
        json.dump(args, file)
        file.close()
        # dict with the state-value mappings
        self.save_values(file_vals)


    def load_args(self, file_args):
        """
        load all the parameters of the player

            file_args   (string)    name of the file without extension
        """
        file = open(file_args+extension, "r")
        args = json.load(file)
        file.close()

        # sign of the agent on the board
        self.sign           = args.get('sign')
        # order for weighting the probability of each action
        self.ord            = args.get('ord')
        # number of training games
        self.num_train      = args.get('num_train')
        # learning rate
        self.lr             = args.get('lr')
        self.lr_min         = args.get('lr_min')
        self.lr_exp         = args.get('lr_exp')
        self.lr_red_steps   = args.get('lr_red_steps')
        # initial value estimate
        self.init_val       = args.get('init_val')
        # dict with the state-value mappings
        self.load_values(args.get('file_vals'))

# end class Player


def normalize(values, ord):
    """
    normalize a list of values according to norm or order 'ord'

        values  (list of float) list of values to normalize

        ord     (int)           order of the norm
    """
    if ord == float('inf'):
        return [ 1 if v==max(values) else 0 for v in values ]
    else:
        n = sum( [ abs(v)**ord for v in values ] )
        return [ abs(v)**ord/n for v in values ]


def pairwise(iterable):
    """
    iterate over a list pairwise

        iterable    (iterable element) list of values to iterate over
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)