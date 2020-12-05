# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-10-25 16:22:44
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-12-05 14:07:33


from parameters import *

import copy
import itertools
import json
# import pandas as pd
import numpy as np


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
    # weigthed sum of al lnext values
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