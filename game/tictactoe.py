# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: OctaveOliviers
# @              on: 2021-04-06T13:00:30+02:00


import numpy as np
import dm_env
from dm_env import specs

from board import Board

class TicTacToe(dm_env.Environment):
    """docstring for TicTacToe"""

    def __init__(
        self,
        nrow: int = 3,
        ncol: int = 3,
        self_play: bool = True,
        **kwargs
    ):
        """
        explain
        """
        super(TicTacToe, self).__init__()

        self._board = Board(nrow=nrow, ncol=ncol, **kwargs)

        self._self_play = self_play

        self._reset_next_step = True


    def reset(self):
        """
        Returns the first `TimeStep` of a new episode
        """
        self._reset_next_step = False
        self._board.reset()
        return dm_env.restart(self._observation())


    def step(self, action, player=1):
        """
        Updates the environment according to the action
        """
        if self._reset_next_step:
            return self.reset()

        self._board[action[0], action[1]] = player

        if self._self_play:
            self.inverse()

        # check for termination
        if not self._board.is_done():
            return dm_env.transition(reward=0., observation=self._observation())
        else:
            reward =
            return dm_env.termination(reward=reward, observation=self._observation())


    def observation_spec(self):
        """
        explain
        """
        pass


    def action_spec(self):
        """
        explain
        """
        pass


    def _observation(self):
        """
        explain
        """
        return self._board.copy()


    def inverse(self):
        """
        explain
        """
        self._board *= -1

# end class TicTacToe(dm_env.Environment)


# tic-tac-toe logic
