# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-12-04 10:31:22
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-12-05 22:43:48

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense, Flatten

import numpy as np

from utils import *
from board import Board
from parameters import *

NROW = 3
NCOL = 3
ORDER = float('inf')

BCH_SIZ = 256


def load_data(nrow, ncol, order):

    state2value = load_dict(file_name=f"../{VALUE_FOLDER}{nrow}x{ncol}/order-{order}{EXTENSION}")
    num_states  = len(state2value)

    data_board  = np.zeros((num_states, nrow, ncol, 1))
    data_policy = np.zeros((num_states, nrow*ncol))
    data_value  = np.zeros((num_states,))

    board = Board(nrow=NROW, ncol=NCOL)

    for num, state in enumerate(state2value):

        board.set_state(state)

        # append state of the board
        data_board[num,:,:,0] = board.get_board()
        # append value of the game
        data_value[num] = state2value.get(state)
        # append policy to choose
        free_pos = [f[0]*ncol+f[1] for f in board.get_free_positions()]
        
        next_states = board.get_next_states(sign='x')
        next_values = [state2value.get(state) for state in next_states]
        # data_policy[num,:] = [1 if next_values[i]==max(next_values) else 0 for i in range(nrow*ncol)]


    return data_board, data_policy, data_value



def build_model(nrow, ncol):
    # input is board
    input = Input(shape=(nrow,ncol,1))

    # convolutional filter
    conv  = Conv2D(filters=8, kernel_size=3, strides=1, padding='same')(input)
    bnorm = BatchNormalization()(conv)
    relu  = ReLU()(bnorm)

    # residual blocks
    for l in range(2):
        # input of residual block
        inres = relu
        # first convolutional block
        conv  = Conv2D(filters=8, kernel_size=3, strides=1, padding='same')(relu)
        bnorm = BatchNormalization()(conv)
        relu  = ReLU()(bnorm)
        # second convolutional block
        conv  = Conv2D(filters=8, kernel_size=3, strides=1, padding='same')(relu)
        bnorm = BatchNormalization()(conv)
        # residual connection
        res   = Add()([inres, bnorm])
        relu  = ReLU()(res)

    # build policy head
    p_conv  = Conv2D(filters=4, kernel_size=3, strides=1, padding='same')(relu)
    p_bnorm = BatchNormalization()(p_conv)
    p_relu  = ReLU()(p_bnorm)
    p_flat  = Flatten()(p_relu)
    p_head  = Dense(nrow*ncol, activation='softmax')(p_flat)

    # build value head
    v_conv  = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(relu)
    v_bnorm = BatchNormalization()(v_conv)
    v_relu  = ReLU()(v_bnorm)
    v_flat  = Flatten()(v_relu)
    v_head  = Dense(1, activation='sigmoid')(v_flat)

    # define the model
    model = Model(inputs=input, outputs=(p_head, v_head))

    model.summary()



if __name__ == "__main__":

    # load data
    board, policy, value = load_data(NROW, NCOL, ORDER)
    # build network
    # model = build_model(NROW, NCOL)
    # train network

    # evaluqte network
