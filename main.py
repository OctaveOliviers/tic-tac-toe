# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 12:11:21
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-10-02 18:49:00

import random
import copy
import json
from itertools import tee
from tqdm import trange
from board import Board
from player import Player

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


epsilon = 1
alpha = 0.1

board = Board()
player = Player(epsilon, alpha)
# board.print()


print("Start training")

# number of training matches
num_train = 100000

for n in trange(num_train):

    # reset board for new game
    board.reset()

    # store the board states to backpropagate value when game ends
    board_states = [board.get_state()]

    # 1/2 of the time o-player plays first
    if random.random() < 0.5:
        # o-player starts in random location
        board.add_o(random.choice([i for i in range(9)]))
        # store board state
        board_states.append(board.get_state())

    while not board.is_full():
        # player 'x' plays
        action = player.choose_action(board.get_state())
        # update board
        board.add_x(action)
        # store board state
        board_states.append(board.get_state())
        # check if player won
        if player.has_won(board.get_state()): break
        # if nobody won yet, inverse the board
        board.inverse()

    # backpropagate value of game to update the policy
    board_states.reverse()
    for state_k, state_km1 in pairwise(board_states):
        player.update_values(state_km1, state_k)


    # board.print()

print("Finished training")
# print('   epsilon : ', player.epsilon)
# print('   alpha   : ', player.alpha)


# save player's policy
file = open("params/policy.json", "w")
json.dump(player.states, file)
file.close()


print("Start testing")
player.playing_mode()
# number of training matches
num_test = 10
num_draw = 0
for n in trange(num_test):

    # reset board for new game
    board.reset()

    # o-player starts in random corner location
    board.add_o(random.choice([0,2,6,8]))

    is_draw = True
    while not board.is_full():
        # player 'x' plays
        action = player.choose_action(board.get_state())
        # update board
        board.add_x(action)
        # check if player won
        if player.has_won(board.get_state()): 
            is_draw = False
            board.print()
            break
        # if nobody won yet, inverse the board
        board.inverse()

    if is_draw:
        num_draw += 1

print("Finished testing")
print('   number of draws : ', num_draw, " of ", num_test)