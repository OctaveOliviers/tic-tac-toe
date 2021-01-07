# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-10-25 16:22:44
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2021-01-07 21:33:36


from parameters import *

import copy
import itertools
import json
import random
# import pandas as pd
import numpy as np
import datetime as dt
from tqdm import trange

inv_reward = lambda r : -r if (RWD_LOOSE == -RWD_WIN) else 1-r

# def inv_reward_01(r):
#         """
#         explanation
#         """
#         return 1-r

# def inv_reward_m11(r):
#     """
#     explanation
#     """
#     return -r


def normalize(values, ord):
    """
    normalize a list of values according to norm or order 'ord'

        values  (list of float) list of values to normalize

        ord     (int)           order of the norm
    """
    if ord == float('inf'):
        n = sum([ 1 if v==max(values) else 0 for v in values ])
        return [ 1/n if v==max(values) else 0 for v in values ]
    elif sum(values) == 0:
        return [ 1 for v in values ]
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


def next_is_sign(cur_pos, all_pos, dir):
    """
    check if position cur_pos + dir is the array all_pos

        cur_pos (1x2 array of int)  current position on the board

        all_pos (nx2 array of int)  n positions on the board

        dir     (1x2 array of int)  the direction in which to search for
    """
    return any(np.equal(all_pos,[cur_pos[0]+dir[0],cur_pos[1]+dir[1]]).all(1))


def value_deeper(board, sign_agent, order, all_values):
    """
    compute values to which training converges
    """
    free_pos = board.get_free_positions()
    next_vals = []
    for pos in free_pos:
        # add symbol on that new position
        next_board = copy.deepcopy(board)
        next_board.add(sign=sign_agent, row=pos[0], col=pos[1])
        next_board_state = next_board.get_state()
        # if win
        if next_board.is_won(): 
            val = RWD_WIN
        # if draw
        elif next_board.is_full():
            val = RWD_DRAW
        # if game not done
        else:
            next_board.inverse()
            val = rwd_inverse(value_deeper(next_board, sign_agent, order, all_values))
        
        all_values[next_board_state] = val
        next_vals.append(val)
    # weigthed sum of all next values
    weights = normalize(next_vals, order)
    return np.dot(weights, next_vals)


def save_dict(data=None, file_name=None):
    """
    save a dictionary
        
        data        (dictionary)

        file_name   (string)    name of the file
    """
    file = open(file_name, "w")
    json.dump(data, file)
    file.close()

    # df = pd.DataFrame.from_dict(data, orient='columns')
    # print(df)
    # df.to_csv(path_or_buf=file_name, float_format='%.3e', header=False)


def load_dict(file_name=None):
    """
    load a dictionary

        file_name   (string)    name of the file
    """
    file = open(file_name, "r")
    data = json.load(file)
    file.close()
    return data    

    # df = pd.read_csv(filepath_or_buffer=file_name)
    # print(df)
    # return df.to_dict()


def monte_carlo_early_start(structure=None, transitions=None, rewards=None, prior=None, gamma=None, num_epi = 1000, len_epi = 10):
    """
    explain

        structure       numpy array of size ( num state-actions x num states )

        transitions     numpy array of size ( num states x num state-actions )

        rewards         numpy array of size ( num state-actions )

        prior           numpy array of size ( num state-actions )

        gamma           float in [0,1]
    """
    # set seed of random number generator
    np.random.seed(seed=dt.datetime.now().year)

    # extract usefull info
    num_s  = structure.shape[1]
    num_sa = structure.shape[0]

    # initialize policy matrix
    policy = np.multiply(structure, np.random.rand(num_sa, num_s))
    policy = policy / np.sum(policy, 0)
    # initialize vector of q-values
    q = np.random.rand(num_sa,)
    # number of times that each state-action is visited
    n = np.zeros(num_sa,)

    # loop until convergence
    # converged = False
    # while not converged:
    for k in trange(num_epi):
        # store the state-actions and rewards encountered in the episode
        z = []
        r = []

        # choose initial state-action pair according to weights in prior
        z.append(random.choices([i for i in range(num_sa)], weights=prior)[0])
        # store reward of initial state-action
        r.append(rewards[z[-1]])
        # update number of visits of initial state-action
        n[z[-1]] += 1

        # generate an episode from initial state
        # converged = False
        # while not converged:
        for t in range(len_epi):
            # go to new state-action
            z.append(random.choices([i for i in range(num_sa)], weights=np.matmul(policy, transitions)[:,z[-1]])[0])
            # store reward of new state-action
            r.append(rewards[z[-1]])
            # update number of visits of new state-action
            n[z[-1]] += 1

        # update q-estimates backwards
        g = 0
        z.reverse()
        r.reverse()
        for t in range(len_epi):
            # update goal value
            g = gamma*g + r[t]
            # update q-estimate with incremetnal mean formula
            # TODO update incremental mean formula because is not correct
            # problem when go several times through same state-action within same episode
            q[z[t]] += (g - q[z[t]]) / n[z[t]]

        # update policy
        policy = np.zeros((num_sa, num_s))
        for s in range(num_s):
            # find actions of that state
            sa =  np.nonzero(structure[:,s])[0]
            # maximal q value in state s
            max_q = np.max(q[sa])
            # choose the action with maximal q-value
            policy[np.where(q[sa]==max_q)[0][0], s] = 1

    return q