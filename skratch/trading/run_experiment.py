# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for running Atari 2600 agents."""

from skratch.common import iteration_statistics
import matplotlib.pyplot as plt

import gym
import numpy as np
import tensorflow as tf


class Runner(object):
    """Object that handles running Atari 2600 experiments.

    Here we use the term 'experiment' to mean simulating interactions between the
    agent and the environment and reporting some statistics pertaining to these
    interactions.

    A simple scenario to train a DQN agent is as follows:

    ```python
    base_dir = '/tmp/simple_example'
    def create_agent(sess, environment):
      return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
    runner = Runner(base_dir, create_agent, game_name='Pong')
    runner.run()
    ```
    """

    def __init__(self,
                 agent,
                 environment):
        """Initialize the Runner object in charge of running a full experiment.

        Args:
          create_agent_fn: A function that takes as args a Tensorflow session and an
            Atari 2600 Gym environment, and returns an agent.
          create_environment_fn: A function which receives a game name and creates
            an Atari 2600 Gym environment.


        This constructor will take the following actions:
        - Initialize an environment.
        - Initialize a `tf.Session`.
        - Initialize a logger.
        - Initialize an agent.
        - Reload from the latest checkpoint, if available, and initialize the
          Checkpointer object.
        """

        self._environment = environment
        # Set up a session and initialize variables.
        #self._sess = tf.Session('',
                              #  config=tf.ConfigProto(allow_soft_placement=True))
        # self._agent = create_agent_fn(self._sess, self._environment,
        #                               summary_writer=self._summary_writer)

        self._agent = agent
        self.total_steps = 0
        self.total_reward = 0


    def run_experiment(self):
        """
        Runs a full experiment and at conclusion plots the important
        statistics.
        """

        self._run_one_episode()

        self._plot_statistics()

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """

        action = self._initialize_episode()
        done = False
        # Keep interacting until we reach a terminal state.
        while not done:
            observation, reward, done = self._run_one_step(action)
            self.num_steps += 1
            if not done:
                action = self._agent.step(reward, observation)
        self._end_episode(reward, observation)

    def _initialize_episode(self):
        """Initialization for a new episode.

        Returns:
          action: int, the initial action chosen by the agent.
        """

        self.rewards = []
        self.benchmark_rewards = []
        self.num_steps = 0
        initial_observation = self._environment.reset()
        return self._agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        """Executes a single step in the environment.

        Args:
          action: int, the action to perform in the environment.

        Returns:
          The observation, reward, and is_terminal values returned from the
            environment.
        """
        observation, reward, done, info = self._environment.step(action)
        self.rewards.append(reward)
        self.benchmark_rewards.append(info['return'])
        return observation, reward, done

    def _end_episode(self, reward, observation):
        """Finalizes an episode run.

        Args:
          reward: float, the last reward from the environment.
        """
        self._agent.end_episode(reward, observation)


    def _plot_statistics(self):

        plt.title("Performance: Agent vs Benchmark")
        plt.plot(np.cumsum(self.rewards))
        plt.plot(np.cumsum(self.benchmark_rewards))

        print("Total Steps: {}".format(self.num_steps))
        print("Total Reward: {}".format(np.sum(self.rewards)))
