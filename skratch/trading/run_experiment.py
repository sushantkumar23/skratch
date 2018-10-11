# Copyright 2018 Skratch Authors

import numpy as np
import matplotlib.pyplot as plt


class Runner(object):
    """
    Runner is a wrapper that takes an agent and an environment and conducts the
    Trading Reinforcement Learning experiment. It keep tracks of the important
    statistics of the experiment. At the conclusion of the experiment, it
    publishes important statistics and plots important statistics.
    """

    def __init__(self, agent, env):
        """
        Parameters
        ----------
        agent (object):
            An agent is an object that should have the implemented the
            following methods begin_episode(), step(), end_episode() to be
            usable.

        env (gym.Env):
            An env is standard OpenAI Gym Environment which should have the
            the standard methods implemented such as the reset(), step()
        """
        self._env = env
        self._agent = agent

    def run_experiment(self):
        """
        Runs a full experiment, stores the relevant statistics and in the end
        plots the performance of the agent against the benchmark.
        """
