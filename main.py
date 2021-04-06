# -*- coding: utf-8 -*-
# @Created by: OctaveOliviers
# @        on: 2021-04-06T12:43:15+02:00
# @Last modified by: OctaveOliviers
# @              on: 2021-04-06T13:00:27+02:00

"""MCTS agent that plays Tic-Tac-Toe"""

# use mcts from acme
# use tictactoe from open_spiel

from absl import app
from absl import flags

import acme
from acme.agents.tf import mcts

def make_environment() -> dm_env.Environment:
    """
    explain
    """
    return


def make_agent() -> :
    """
    explain
    """
  return


def main(_):
  env = make_environment()
  env_spec = acme.make_environment_spec(env)
  network = networks.DQNAtariNetwork(env_spec.actions.num_values)

  agent = dqn.DQN(env_spec, network)

  loop = acme.EnvironmentLoop(env, agent)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
