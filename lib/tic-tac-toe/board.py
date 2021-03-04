# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2020-09-17 13:04:41
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-02-07 16:39:25


import copy
import numpy as np


class Board:
    """
    Tic-Tac-Toe board
    """

    def __init__(self, nrow=None, ncol=None, **kwargs):
        """
        Constructor for class Board

            nrow        (int)       number of rows on the board
                                    should be positive

            ncol        (int)       number of columns on the board
                                    should be positive

        **kwargs
            len2win     (int)       min number of signs to align in order to win
                                    should be <= than nrow and ncol

            sign_empty  (string)    sign for an empty spot on the board
                                    by default '-'

            sign_play   (string)    signs used by the players
                                    by default ['x','o']

            num2sign    (dict)      map from the numbers in self.state to the players' signs
                                    by default use numbers 0, 1 and -1

            sign2num    (dict)      map from the players' signs to the numbers in self.state
        """
        self._nrow       = nrow
        self._ncol       = ncol
        self._state      = np.zeros((self.nrow,self.ncol), dtype=np.int8)
        self._len2win    = kwargs.get("len2win", min(nrow, ncol))
        assert self._len2win <= min(nrow, ncol)

        self._sign_empty = kwargs.get('sign_empty', '-')
        self._sign_play  = kwargs.get('sign_play', ['x','o'])
        self._num2sign   = { 0:self.sign_empty, 1:self.sign_play[0], -1:self.sign_play[1] }
        self._sign2num   = { v:k for k,v in self.num2sign.items() }


    def __getitem__(self, index):
        """
        explain
        """
        return self._state[index]


    def __setitem__(self, index, value):
        """
        explain
        """
        self._state[index] = value


    def __str__(self):
        """
        Visualize the board
        """
        for i in range(self.nrow):
            print('\n', end=" ")
            for j in range(self.ncol):
                print(self.num2sign.get(self.state[i,j]), end=" ")
        print()


    def reset(self):
        """
        Make the board empty
        """
        self._state.fill(0)


    def inverse(self):
        """
        Check whether either of the players has won
        """
        self._state *= -1


    # def add(self, sign=None, row=None, col=None):
    #     """
    #     Add a sign to the board

    #         sign    (string)    sign of a player on the board game
    #                             should be a sign in self.sign_play

    #         row     (int)       on which row should the sign be added
    #                             should be between 0 and self.nrow

    #         col     (int)       on which column should the sign be added
    #                             should be between 0 and self.ncol
    #     """
    #     assert [row,col] in self.get_free_positions()
    #     self.state[row, col] = self.sign2num.get(sign)


    # def remove(self, row=None, col=None):
    #     """
    #     explain
    #     """
    #     self.state[row, col] = self.sign2num.get(self.sign_empty)


    def sign_won(self, sign):
        """
        Check whether the player that plays 'sign' has won

            sign    (string)    sign of a player on the board game
                                should be a sign in self.sign_play
        """
        # position of the sign on the board
        pos_all = np.transpose(np.where(self.state==self.sign2num.get(sign)))
        for pos in pos_all:
            # all the search directions
            dir_all = np.array([ [1,-1], [1,0], [1,1], [0,1] ])
            for dir in dir_all:
                pos_cur = pos.copy()
                i = 1
                while Board.next_is_sign(pos_cur, pos_all, dir):
                    i += 1
                    pos_cur += dir
                    if i == self.len2win:
                        return True
        # if did not find any sequence of length self._len2win 
        return False

    
    def is_won(self):
        """
        Check whether either of the players won
        """
        return any([self.sign_won(s) for s in self._sign_play])


    def is_full(self):
        """
        Check whether there are any free positions left on the board
        """
        return True if np.count_nonzero(self._state)==self._nrow*self._ncol else False


    def is_done(self):
        """
        Check wether the game is done (someone won or board is full)
        """
        return self.is_won() or self.is_full()


    def is_turn_sign(self, sign=None):
        """
        Check whether it is the turn of the 'sign'-player to play
        """
        # opponent sign
        opp_sign = self.sign_play[1-self.sign_play.index(sign)]
        return (self.get_state().count(sign) <= self.get_state().count(opp_sign)) and (not self.is_done())


    def get_legal_moves(self):
        """
        Compute the free positions on the board
        """
        return np.transpose(np.where(self.state==self.sign2num.get(self.sign_empty)))


    def get_next_states(self, sign=None):
        """
        explain
        """
        next_states = []
        for pos in self.get_free_positions():
            self.add(sign=sign, row=pos[0], col=pos[1])
            next_states.append(self.get_state())
            self.remove(row=pos[0], col=pos[1])

        return next_states


    def get_state(self):
        """
        return a string with the signs on the board in row-major order
        """
        return ''.join([ self.num2sign.get(n) for n in self.state.flatten() ])


    def set_state(self, state):
        """
        set the board to a given state (state in row-major order)
        """
        self.state = np.reshape([ self.sign2num.get(n) for n in state ], (self.nrow, self.ncol))


    @staticmethod
    def next_is_sign(cur_pos, all_pos, dir):
        """
        check if position cur_pos + dir is in the array all_pos

            cur_pos (1x2 array of int)  current position on the board

            all_pos (nx2 array of int)  n positions on the board

            dir     (1x2 array of int)  the direction in which to search for
        """
        return any(np.equal(all_pos,[cur_pos[0]+dir[0],cur_pos[1]+dir[1]]).all(1))

# end class Board