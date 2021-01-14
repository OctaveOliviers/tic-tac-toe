# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-10-25 16:22:44
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2021-01-14 13:34:11


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


def monte_carlo_early_start(structure=None, transitions=None, rewards=None, prior=None, discount=None, num_epi = 50, len_epi = 20):
    """
    explain

        structure       numpy array of size ( num state-actions x num states )

        transitions     numpy array of size ( num states x num state-actions )

        rewards         numpy array of size ( num state-actions )

        prior           numpy array of size ( num state-actions )

        gamma           float in [0,1]
    """
    # set seed of random number generator
    np.random.seed(seed=dt.datetime.now().day)

    # extract useful info
    num_sa,num_s = structure.shape

    # initialize vector of q-values
    q = np.random.rand(num_sa,)
    # compute policy matrix from q-values
    policy = compute_policy(S=structure, q=q)
    # number of times that each state-action is visited
    n = np.zeros(num_sa,)

    V_old, val_old = compute_potential(policy, transitions, rewards, prior, q, discount, len_epi)
    pol_old = policy
    pol_new = policy
    print(f"policy changed to {pol_old}")
    print(f"value of policy {np.sum(val_old)}")

    # loop until convergence
    for k in range(num_epi):

        # test if this value increases
        #inv_mat = np.linalg.inv( np.eye(num_sa) - discount * np.transpose(np.matmul(policy, transitions)) )
        #print(f" does this value increase ? {np.sum(np.matmul( inv_mat, rewards ))} ")
        
        V_new, val_new = compute_potential(policy, transitions, rewards, prior, q, discount, len_epi)

        if not np.array_equal(pol_old, pol_new): 
            print(f"policy changed to {policy}")
            print(f"value of policy {np.sum(val_new)}")

        pol_old = pol_new
        
        print(f"potential is V = {V_new}")
        
        # print(f"number of visits {n}")
        #print(f"potential increases? {V_old <= V_new}")
        #V_old = V_new
        

        # store the state-actions and rewards encountered in the episode
        z = []
        r = []
        # store number of visits of each state-action in this episode
        n_epi = np.zeros(num_sa,)

        # choose initial state-action pair according to weights in prior
        z.append(random.choices([i for i in range(num_sa)], weights=prior)[0])
        # store reward of initial state-action
        r.append(rewards[z[-1]])
        # update number of visits of initial state-action
        n_epi[z[-1]] += 1

        # generate an episode from initial state
        for t in range(len_epi):
            # go to new state-action
            z.append(random.choices([i for i in range(num_sa)], weights=np.matmul(policy,transitions)[:,z[-1]])[0])
            # store reward of new state-action
            r.append(rewards[z[-1]])
            # update number of visits of new state-action
            n_epi[z[-1]] += 1

        # update total number of visits of each state-action
        n += n_epi

        # update q-estimates
        update_q_values(q=q, z=z, r=r, n=n, n_epi=n_epi, gam=discount)
        # compute policy matrix from q-values
        policy = compute_policy(S=structure, q=q)

        pol_new = policy
    
    return q


def update_q_values(q=None, z=None, r=None, n=None, n_epi=None, gam=None):
    """
    explain
    """
    # extract useful info
    len_epi = len(z)

    # goal value
    g = 0
    # update backwards
    z.reverse()
    r.reverse()
    # loop over each step of the episode
    for t in range(len_epi):
        # update goal value
        g = gam*g + r[t]

        # update q-estimate with incremetnal mean formula
        n_epi[z[t]] -= 1
        q[z[t]] += (g - q[z[t]]) / (n[z[t]] - n_epi[z[t]])


def compute_policy(S=None, q=None):
    """
    explain
    """
    # extract useful info
    num_sa,num_s = S.shape

    # policy matrix
    P = np.zeros((num_sa, num_s))
    # for each state choose action with highest q-value
    for s in range(num_s):
        # find actions of that state
        sa = np.nonzero(S[:,s])[0]
        # index of maximal q value in state s
        idx_max_q = np.argmax(q[sa])
        # choose the action with maximal q-value
        P[sa[idx_max_q], s] = 1
        
    return P
    
    
def compute_potential(P, T, r, p, q, gam, len_epi):
    """
    explain
    """
    # extract useful info
    num_sa,num_s = P.shape
    
    A = np.matmul(P,T)

    diag = np.zeros(num_sa,)
    block = np.zeros((num_sa, num_sa))

    V = 0

    for l in range(len_epi+1):
        # [I + gam A.T + ... + gam^(L-l) A.T^(L-l)]
        mat_sum = np.matmul(np.linalg.inv(np.eye(num_sa)-gam*A.T), 
                           (np.eye(num_sa)-gam**(len_epi+1-l)*np.linalg.matrix_power(A.T,len_epi+1-l)))
        # A^l p
        Al_p = np.matmul(np.linalg.matrix_power(A,l), p)
        # diag(p + A p + ... + A^l p)
        diag += Al_p
        # [diag(p) [I + ... + gam^L A.T^L] + ... + diag(A^L p) [I]]
        block += np.matmul(np.diag(Al_p), mat_sum)

    V += np.matmul(q.T*diag, q) /2

    V -= np.matmul(np.matmul(r.T, block.T), q)
    
    V += np.matmul(np.matmul(r.T, block.T), np.matmul(block, r)/diag) /2

    # val = np.matmul(P.T, np.matmul(block, r)/diag)
    val = - np.matmul( np.matmul(r.T, block.T)/diag, np.matmul(block, r) ) /2

    return V, val