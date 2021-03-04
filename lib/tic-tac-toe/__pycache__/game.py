# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-02-07 11:58:46
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-02-07 15:23:48



class Game(object):
    """docstring for Tree"""
    def __init__(self, **kwargs):
        super(Game, self).__init__()
        
        self._state

        self._inv_reward = 
        self._reward_win
        self._reward_lose
        self._reward_draw
    
        self._q_init

        self.curr_player = 1

    @property
    def state(self):
        return self._state
    
    @property
    def next_states(self):
        """
        explain
        should return in same order as free_actions
        """
        return 
    
    @property
    def reward_win(self):
        return self._reward_win
    
    @property
    def reward_lose(self):
        return self._reward_lose
    
    @property
    def reward_draw(self):
        return self._reward_draw

    @property
    def q_init(self):
        return self._q_init

    def update(self, action):
        """
        explain
        """
        # add action for current player


        self.curr_player *= -1
        pass

    def inverse(self):
        pass

    def free_actions(self):
        """
        explain
        should return in same order as next_states
        """
        pass

    def state(self):
        pass

    def save_state(self):
        pass

    def restore_state(self):
        pass

    def is_done(self):
        pass

    def is_won(self):
        pass