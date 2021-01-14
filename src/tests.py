# -*- coding: utf-8 -*-
# @Author: OctaveOliviers
# @Date:   2021-01-07 21:11:19
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2021-01-14 18:13:06

from utils import *

def test_monte_carlo_early_starts():
    """
    explain
    """

    # set seed of random number generator
    #np.random.seed(seed=dt.datetime.now().day)

    # structure of markov tree
    S = np.array([[1, 0, 0, 0, 0], 
                  [1, 0, 0, 0, 0], 
                  [0, 1, 0, 0, 0], 
                  [0, 1, 0, 0, 0], 
                  [0, 0, 1, 0, 0], 
                  [0, 0, 1, 0, 0], 
                  [0, 0, 0, 1, 0], 
                  [0, 0, 0, 0, 1]] )
    # transition probabilities
    T = np.array([[0, 0, 0, 0, 1, 0, 0, 0], 
                  [1, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 1, 0, 0, 0, 0, 0], 
                  [0, 1, 0, 1, 0, 0, 1, 0], 
                  [0, 0, 0, 0, 0, 1, 0, 1]])
    # rewards of each state-action
    R = np.array([2, -10, 0, -10, 0, 10, 0, 0])
    # prior probability of starting episode in each state-action
    p = np.random.rand(8,)
    p = p/np.sum(p)
    # discount factor
    gam = 0.9

    q = monte_carlo_early_start(structure=S, transitions=T, rewards=R, prior=p, discount=gam)
    
    print(f"Computed q-values are {q}")
    print("Should be close to [ 8.1 -10 9 -10 7.29 10 0 0 ]")


def test_compute_policy():
    """
    explain
    """

    # structure of markov tree
    S = np.array([[1, 0, 0, 0, 0], 
                  [1, 0, 0, 0, 0], 
                  [0, 1, 0, 0, 0], 
                  [0, 1, 0, 0, 0], 
                  [0, 0, 1, 0, 0], 
                  [0, 0, 1, 0, 0], 
                  [0, 0, 0, 1, 0], 
                  [0, 0, 0, 0, 1]] )
    # q values
    q = np.array([])
    

    p = compute_policy(S=S, q=q)
    print(f"q is {q}")
    print(f"p is {p}")


def test_update_q_values():
    """
    explain
    """

    # set seed of random number generator
    np.random.seed(seed=dt.datetime.now().day)
    
    q = np.random.rand(4)
    z = [0,2,1,3]
    r = [0,1,10,5]
    n = np.array([1,1,1,1])
    n_epi = np.array([1,1,1,1])
    gam = .1

    print(q)
    q = update_q_values(q=q, z=z, r=r, n=n, n_epi=n_epi, gam=gam)
    print(q)
    
    
if __name__ == '__main__':
    test_monte_carlo_early_starts()
    
    
