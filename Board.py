# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2020-09-17 13:04:41
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2020-09-17 16:51:33

class Board:
    """docstring for Board"""

    def __init__(self):
        self.set('---------')
        

    def set(self, setup):
        self.board = list(setup)


    def print(self):
        print()
        print(self.board[0:3])
        print(self.board[3:6])
        print(self.board[6::])
        print()


    def inverse(self):
        index_x = [index for index, element in enumerate(self.board) if element == 'x']
        index_o = [index for index, element in enumerate(self.board) if element == 'o']
        for i in index_x: self.board[i] = 'o'
        for i in index_o: self.board[i] = 'x'


    def update(self, index):
        self.board[index] = 'x'


    def get_state(self):
        return self.board


    def is_full(self):
        return False if [index for index, element in enumerate(self.board) if element == '-'] else True
