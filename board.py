# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 13:04:41
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-10-23 23:50:13

import numpy as np
import copy

class Board:
    """
    tic-tac-toe board

    given a certain number of rows and columns, it is possible to add signs to the board,
    check if either of the players has won, compute which positions are still free to play,
    visualize the board, and invert all the signs on the board
    """

    def __init__(self, nrow=None, ncol=None, **kwargs):
        """
        constructor for class Board

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
        self.nrow       = nrow
        self.ncol       = ncol
        self.len2win    = kwargs.get("len2win", min(self.nrow, self.ncol))
        assert self.len2win <= min( self.nrow, self.ncol )

        self.sign_empty = kwargs.get('sign_empty', '-')
        self.sign_play  = kwargs.get('sign_play', ['x','o'])
        self.num2sign   = { 0:self.sign_empty, 1:self.sign_play[0], -1:self.sign_play[1] }
        self.sign2num   = { v:k for k,v in self.num2sign.items() }

        self.reset()


    def reset(self):
        """
        make the board empty
        """
        self.state = np.zeros((self.nrow,self.ncol), dtype=np.int8)


    def print(self):
        """
        visualize the board
        """
        for i in range(self.nrow):
            print('\n', end=" ")
            for j in range(self.ncol):
                print(self.num2sign.get(self.state[i,j]), end=" ")
        print()


    def inverse(self):
        """
        check whether either of the players has won
        """
        self.state *= -1


    def add(self, sign=None, row=None, col=None):
        """
        add a sign to the board

            sign    (string)    sign of a player on the board game
                                should be a sign in self.sign_play

            row     (int)       on which row should the sign be added
                                should be between 0 and self.nrow

            col     (int)       on which column should the sign be added
                                should be between 0 and self.ncol
        """
        assert [row,col] in self.get_free_positions()
        self.state[row, col] = self.sign2num.get(sign)


    def sign_won(self, sign):
        """
        check whether the player that plays 'sign' has won

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
                while next_is_sign(pos_cur, pos_all, dir):
                    i += 1
                    pos_cur += dir
                    if i == self.len2win:
                        return True
        # if did not find any sequence of length self.len2win 
        return False


    def is_won(self):
        """
        check whether either of the players has won
        """
        return any( [ self.sign_won(s) for s in self.sign_play] )


    def is_full(self):
        """
        check is there any free positions left onthe board
        """
        return True if np.count_nonzero(self.state)==self.nrow*self.ncol else False


    def get_free_positions(self):
        """
        compute the free positions on the board
        """
        return np.transpose(np.where(self.state==self.sign2num.get(self.sign_empty)))


    def get_state(self):
        """
        return a string with the signs on the board in row-major order
        """
        return ''.join([ self.num2sign.get(n) for n in self.state.flatten() ])


    def get_nrow(self):
        """
        return the number of rows on the board
        """
        return self.nrow


    def get_ncol(self):
        """
        return the number of columns on the board
        """
        return self.ncol

# end class Board


def next_is_sign(cur_pos, all_pos, dir):
    """
    check if position cur_pos + dir is the array all_pos

        cur_pos (1x2 array of int)  current position on the board

        all_pos (nx2 array of int)  n positions on the board

        dir     (1x2 array of int)  the direction in which to search for
    """
    return any(np.equal(all_pos,[cur_pos[0]+dir[0],cur_pos[1]+dir[1]]).all(1))