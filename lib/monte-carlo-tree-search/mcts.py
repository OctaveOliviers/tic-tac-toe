# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @Created on: 2021-02-07 11:34:37
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-02-07 13:36:37


# import libraries
from tqdm import tqdm


def mcts(game, fun_prior, p_expl=1, num_sim=10**3, verbose=0, return_dicts=False):
    """
    explain: evaluate, expand, backup
    """
    # visit count
    N = {}
    # mean action value
    Q = {}
    # save current state of game
    game.save_state()

    # update the dictionaries
    for n in tqdm(range(num_sim), disable=(not verbose)):
        mcts_simulation(game, fun_prior, N, Q, p_expl)
        # restore game state to last saved state
        game.restore_state()

    # get count for each next state
    N_next_states = {nxt : N.get(nxt,0) for nxt in game.next_states}

    return N_next_states if not return_dicts else N_next_states, N, Q


def mcts_simulation(game, fun_prior, N, Q, p_expl):
    """
    explain: evaluate, expand, backup
    """
    # store states of MCTS simulation
    game_states = []

    # assume that the game will be a draw
    reward = game.reward_draw
    while not game.is_done():

        # update visit count (necessary because of self-play = inverse board)
        N[game.state] = N.get(game.state, 0) + 1

        # evaluate possible actions
        ucb_vals = mcts_evaluate(game, N, Q, fun_prior, p_expl)

        # take action that maximizes the UCB value
        game.update(game.free_actions()[ucb_vals.index(max(ucb_vals))])
        
        # update visit count
        N[game.state] = N.get(game.state, 0) + 1
        # add board state to list of visited states
        game_states.append(game.state)

        # check if agent won
        if game.is_won(): 
            reward = game.reward_win
            break
        # if nobody won yet, inverse the board
        game.inverse()

    # backup
    mcts_backup(reward, game_states, N, Q, game.inv_reward)       
    

def mcts_evaluate(game, N, Q, fun_prior, p_expl):
    """
    explain
    """
    ucb_vals  = []
    for state in game.next_states:
        q  = Q.get(state, game.q_init)
        p  = fun_prior(state)
        na = N.get(state, 0)
        nb = N.get(game.state)
        ucb_vals.append(q + p_expl * p * math.sqrt(nb) / (1+na))

    return ucb_vals


def mcts_backup(reward, game_states, N, Q, inv_reward):
    """
    explain
    """
    game_states.reverse()
    # update each board value
    for state in game_states:
        # incremental mean formula
        Q[state] += (reward - Q[state]) / N[state]
        # inverse reward due to self-play
        reward = inv_reward(reward)