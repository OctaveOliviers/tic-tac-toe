# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 12:11:21
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-10-01 18:01:06

import random
import copy
from Board import Board
from Player import Player

# random.seed(a=2020)
# number of training matches
num_train = 1000

epsilon = 0.5
alpha = 0.5


board = Board()
player = Player(epsilon, alpha)
# board.print()

for n in range(num_train):

    for i in range(9):
        # player 'x' plays
        action = player.choose_action(board.get_state())

        board_old = copy.deepcopy(board.get_state())

        # update board
        board.update( action )

        board_new = copy.deepcopy(board.get_state())

        player.update_values(board_old, board_new)

        # check if player won
        if player.has_won(board.get_state()): break

        board.inverse()


    # board.print()

    # reset board for new game
    board.set('---------')


print(player.states)

print('epsilon : ', player.epsilon)
print('alpha : ', player.alpha)

print(player.states['xxoxo---o'])
