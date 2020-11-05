# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 13:05:22
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-10-31 13:11:17


import copy
import json
import math
import random
from datetime import datetime
from tqdm import trange
from utils import *

EXTENSION = '.json'
VALUE_FOLDER = 'data/values/'
PLAYER_FOLDER = 'data/player/'
# CONVERGENCE_FOLDER = 'data/convergence/'
# value of rewards
RWD_WIN = 1
RWD_DRAW = .5
RWD_LOOSE = 0

class Player(object):
    """
    reinforcement learning agent for tic-tac-toe    
    """

    def __init__(self, board=None, **kwargs):
        """
        explanation   
        """
        # sign of the agent on the board
        self.sign           = kwargs.get('sign', 'x')

        # order for weighting the probability of each action
        self.set_order(kwargs.get('order', 1))
        self.ord_max         = kwargs.get('ord_max', 10)
        self.ord_exp         = kwargs.get('ord_exp', 2)
        self.ord_incr_steps  = kwargs.get('ord_incr_steps', 1000)

        # number of training games
        self.num_train      = kwargs.get('num_train', 1000)

        # learning rate
        self.set_lr(kwargs.get('lr', 0.5))
        self.lr_min         = kwargs.get('lr_min', 0.01)
        self.lr_exp         = kwargs.get('lr_exp', 0.9)
        self.lr_red_steps   = kwargs.get('lr_red_steps', 1000)

        # dict with the state-value mappings
        self.state2value    = {}
        # initial value estimate
        self.init_val       = kwargs.get('init_val', 0.5)

        # name of player used for saving parameters
        self.name           = datetime.now().strftime("%y%m%d-%H%M")

        # function to inverse rewards
        self.inv_reward     = getattr(self, 'inv_reward_01') if RWD_LOOSE == 0 else getattr(self, 'inv_reward_m11')


    def train(self, board, **kwargs):
        """
        explanation

            board

            algorithm   (string)    td, az, q, sarsa

            num_train 

            store_values

            store_convergence
        """
        # training algorithm for updating the values
        train_algo = kwargs.get('algorithm', 'td')
        # number of training games
        self.num_train = kwargs.get('num_train', self.num_train)
        # whether to store self.state2val after training or not
        store_values = kwargs.get('store_values', False)
        # whether to store convergence info or not
        store_convergence = kwargs.get('store_convergence', False)

        if store_convergence:
            # load values towards which will converge
            self.load_conv_val()
            # store convergence data (mean and std error)
            conv_data = np.zeros((self.num_train,2))
            for n in trange(self.num_train):
                self.train_step(board, n, train_algo, store_convergence)
                # compute convergence
                conv_data[n,0], conv_data[n,1] = self.compute_convergence() 

            # store convergence data
            np.savetxt(PLAYER_FOLDER+self.name+'-convergence.txt', conv_data, fmt='%.3e')

        else:
            for n in trange(self.num_train):
                self.train_step(board, n, train_algo, store_convergence)      

        # store learned values of states
        if store_values:
            self.save_args(PLAYER_FOLDER+self.name)


    def train_step(self, board, game_num, train_algo, store_convergence):
        """
        explanation   
        """
        # reset board for new game
        board.reset()
        # store the board states
        board_states = []
        
        # assume that the game will be a draw
        reward = RWD_DRAW
        # play until board is full or someone won
        while not board.is_full():
            # choose action and update the board
            board = self.play(board)
            # store board state for training later
            board_states.append(board.get_state())
            # check if player won
            if board.is_won(): 
                reward = RWD_WIN
                break
            # if nobody won yet, inverse the board
            board.inverse()

        # update the state-value approximations
        # self.update_values(reward, board_states)
        getattr(self, train_algo.lower() + '_update')(reward, board_states)
        
        # reduce learning rate progressively
        if (game_num+1) % self.lr_red_steps == 0:
            self.reduce_lr()

        # increase order progressively
        if (game_num+1) % self.ord_incr_steps == 0:
            self.increase_order(store_convergence)
    

    def choose_action(self, board):
        """
        explanation   
        """
        # free board positions
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
        return random.choices(free_pos, weights=normalize(next_values, self.ord))[0]


    def compute_convergence(self):
        """
        compute incremental mean and standard deviation
        """
        count = 0
        mean  = 0
        var   = 0

        for state in self.conv_vals:
            # compute error on that state
            curr = self.state2value.get(state, 0.5)
            conv = self.conv_vals.get(state)
            err  = abs(conv-curr)
            # update mean and variance
            mean_prev = mean
            count += 1
            mean  += (err-mean_prev)/count
            var   += (err-mean_prev)*(err-mean)
        # return mean and standard deviation
        return mean, math.sqrt(var/count)


    def td_update(self, reward, board_states):
        """
        explain
        """
        board_states.reverse()
        # value of the last board state was the reward
        self.state2value[board_states[0]] = reward
        # backpropagate value of game to update the policy
        for state_new, state_old in pairwise(board_states):
            val_old = self.state2value.get(state_old, self.init_val)
            val_new = self.state2value.get(state_new, self.init_val)

            self.state2value[state_old] = val_old + self.lr * ( self.inv_reward(val_new) - val_old )


    def az_update(self, reward, board_states):
        """
        explain alpha-zero update
        """
        board_states.reverse()
        # value of the last board state was the reward
        self.state2value[board_states[0]] = reward
        # update each board value towards the terminal reward
        for state in board_states[1:]:
            # get current estimate for value in that state
            val = self.state2value.get(state, self.init_val)
            # inverse reward due to self-play
            reward = self.inv_reward(reward)
            # update estimate for value in that state
            self.state2value[state] = val + self.lr * ( reward - val )
     

    def sarsa_update():
        """
        explain
        """
        pass


    def q_update():
        """
        explain
        """
        pass


    def play(self, board):
        """
        given a board, choose an action and update the board

            board   (Board)     board that the agent has to play on
        """
        # player chooses an action
        action = self.choose_action(board)
        # update board
        board.add(sign=self.sign, row=action[0], col=action[1])
        return board


    def inv_reward_01(self, r):
        """
        explanation
        """
        return 1-r


    def inv_reward_m11(self, r):
        """
        explanation
        """
        return -r


    def set_order(self, new_ord):
        """
        explanation
        """
        self.ord = new_ord 


    def increase_order(self, store_convergence):
        """
        explanation
        """
        self.ord = min(self.ord_exp*max(self.ord, 1/self.ord_exp), self.ord_max)
        if store_convergence : self.load_conv_val()
     

    def load_conv_val(self):
        """
        load the values towards which training converges
        """
        ord2str = str(self.ord) if self.ord == float('inf') else str(int(self.ord))
        self.conv_vals = load_dict(file_name=VALUE_FOLDER+'order-'+ord2str+EXTENSION)


    # def playing_mode(self):
    #     """
    #     set the order for the weight normalisation to infinity
    #     """
    #     self.increase_order(new_val=float('inf'))


    # def training_mode(self):
    #     """
    #     set the order for the weight normalisation to self.ord
    #     """
    #     self.p = self.ord
    #     # load the values towards which training converges
    #     self.conv_vals = load_dict(file_name=VALUE_FOLDER+'order-'+str(self.ord)+EXTENSION)
    

    def set_lr(self, new_lr):
        """
        set the learning rate
        """
        self.lr = new_lr


    def reduce_lr(self):
        """
        reduce the learning rate
        """
        self.lr = max(self.lr_exp*self.lr, self.lr_min)
    

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
        save_dict(data=self.state2value, file_name=file_name+EXTENSION)

        
    def load_values(self, file_name):
        """
        load the dictionary that contains the state-value mappings

            file_name   (string)    name of the file without extension
        """
        self.state2value = load_dict(file_name=file_name+EXTENSION)
        
    
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
        save_dict(data=args, file_name=file_args+EXTENSION)
        # dict with the state-value mappings
        self.save_values(file_vals)


    def load_args(self, file_args):
        """
        load all the parameters of the player

            file_args   (string)    name of the file without extension
        """
        args = load_dict(file_name=file_args+EXTENSION)
        
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


    def get_name(self):
        return self.name

# end class Player