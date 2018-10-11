# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np


class RandomAgent(object):
    """
    An agent implementation for benchmarking other agents against. This agent
    does no learning. Just keeps picking actions from uniform random
    distribution at each timestep.
    """

    def __init__(self, num_actions):

        self.num_actions = num_actions
        print("Creating Random Agent with {} actions".format(self.num_actions))

    # Returns the agent's first action for the episode
    def begin_episode(self, observation):
        """
        Similar to the step function, except that it just receives the
        obsevation as the parameter and returns the action.
        Parameters:
        ----------
        observation (tuple):
            the first observation that is recevied on env.reset() should be
            passed to begin episode.
        Returns:
        -------
        action (integer):
            action is the discrete integer from the action space that the
            agent wants to take as the first action.
        """
        self.action = self._select_action()
        return self.action

    def step(self, reward, observation):
        """
        Records the most recent trasition into replay buffer and return's the
        agent's next action
        """
        self.action = self._select_action()
        return self.action

    def end_episode(self, reward, observation):
        """
        Records an end of episode. This method notes the reward and terminal
        observation and does not return an action unlike the step function.
        Parameters:
        ----------
        reward (float):
            The terminal reward that the agent receives at the end of an
            episode.
        observation:
            The terminal observation that is emitted by the environment at the
            end of an epiode.
        """
        pass

    def _select_action(self):
        """
        Selects a random action from the set of possible actions.
        Returns:
        -------
        action (int):
            A discrete action selected from uniform random distribution.
        """
        return np.random.randint(self.num_actions)
