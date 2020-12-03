# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-12-02 12:37:05
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-12-02 12:39:27


import multiprocessing as mp
import itertools as itr
from utils import *
import math
import random
import time
from tqdm import trange

from board import Board


def choose_action_mcts_mp(board, num_sim=10**2, return_dicts=False):
    """
    explain Monte Carlo Tree Search
    """
    manager = mp.Manager()
    N = manager.dict()      # visit count
    Q = manager.dict()      # mean action value
    P = manager.dict()      # prior probability of that action, {} or could load existing dictionary
    c = 1                   # exploration/exploitation trade-off

    p = mp.Pool(processes=mp.cpu_count())
    p.starmap(mcts_simulation, itr.repeat((board, N, Q, P, c), times=num_sim))
    p.close()
    p.join()

    # next possible states
    next_states = board.get_next_states(sign='x')    
    # get count for each next state
    next_counts = [N.get(state, 0) for state in next_states]
    # randomly select action according to weights in next_counts
    action = random.choices(board.get_free_positions(), weights=normalize(next_counts, float('inf')))[0]

    return action, N, Q if return_dicts else action


def choose_action_mcts(board, num_sim=10**2, return_dicts=False):
    """
    explain Monte Carlo Tree Search
    """

    N = {}      # visit count
    Q = {}      # mean action value
    P = {}      # prior probability of that action, {} or could load existing dictionary
    c = 1       # exploration/exploitation trade-off
    
    # perform the simulations
    for n in range(num_sim):
        mcts_simulation(board, N, Q, P, c)

    # next possible states
    next_states = board.get_next_states(sign='x')    
    # get count for each next state
    next_counts = [N.get(state, 0) for state in next_states]
    # randomly select action according to weights in next_counts
    action = random.choices(board.get_free_positions(), weights=normalize(next_counts, float('inf')))[0]

    return action, N, Q if return_dicts else action


def mcts_simulation(board, N, Q, P, c):
    """
    explain: select, expand and evaluate, backup
    """
    # play on a copy of the board
    board_cpy = copy.deepcopy(board)
    # store all the states of this MCTS simulation
    board_states = []

    # assume that the game will be a draw
    reward = 0.5
    while not board_cpy.is_full():

        # update visit count (necessary because of self-play = inverse board)
        N[board_cpy.get_state()] = N.get(board_cpy.get_state(), 0) + 1

        # evaluate possible actions
        next_states = board_cpy.get_next_states(sign='x')
        ucb_states  = []
        for state in next_states:
            q  = Q.get(state, 0.5)
            p  = P.get(state, 1/len(next_states))
            na = N.get(state, 0)
            nb = N.get(board_cpy.get_state())
            ucb_states.append(q + c * p * math.sqrt(nb) / (1+na))

        # select action that maximizes the UCB value
        action = random.choices(board_cpy.get_free_positions(), weights=normalize(ucb_states, float('inf')))[0]
        # take action
        board_cpy.add(sign='x', row=action[0], col=action[1])
        # update visit count
        N[board_cpy.get_state()] = N.get(board_cpy.get_state(), 0) + 1
        # add board state to list of visited states
        board_states.append(board_cpy.get_state())

        # check if player won
        if board_cpy.is_won(): 
            reward = 1
            break
        # if nobody won yet, inverse the board
        board_cpy.inverse()   

    # backup
    board_states.reverse()
    # update each board value
    for state in board_states:
        q = Q.get(state, 0.5)
        n = N.get(state)
        # incremental mean formula
        Q[state] = q + (reward - q) / n
        # inverse reward due to self-play
        reward = 1-reward


if __name__ == "__main__":
    me    = 'o'
    agent = 'x'
    nrow  = 3
    ncol  = 3

    board  = Board(nrow=nrow, ncol=ncol, sign_play=[agent,me])
    board.set_state('xox-ox--o')
    board.print()

    start = time.time()
    action, N, Q = choose_action_mcts_mp(board, num_sim=10**5, return_dicts=True)
    end = time.time()
    print(f"multi-processing took {end-start} seconds")

    board.print()

    start = time.time()
    action, N, Q = choose_action_mcts(board, num_sim=10**5, return_dicts=True)
    end = time.time()
    print(f"no multi-processing took {end-start} seconds")

    board.print()