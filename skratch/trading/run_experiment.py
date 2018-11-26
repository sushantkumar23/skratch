# Copyright 2018 The Skratch Authors.

"""Module defining classes and helper methods for a trading agent."""

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime


class Runner(object):
    """Object that handles running Atari 2600 experiments.

    Here we use the term 'experiment' to mean simulating interactions between
    the agent and the environment and reporting some statistics pertaining to
    these interactions.

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
                 agent_fn,
                 environment,
                 base_dir=None):
        """
        Initialize the Runner object in charge of running a full experiment.

        Args:
        agent (object):
            An agent is an object that should have the implemented the
            following methods begin_episode(), step(), end_episode() to be
            usable.

        env (gym.Env):
            An env is standard OpenAI Gym Environment which should have the
            the standard methods implemented such as the reset(), step()

        This constructor will take the following actions:
        - Initialize an environment.
        - Initialize an agent.
        """

        self.__name__ = "trading-exp0"
        self._base_dir = base_dir

        if self._base_dir is None:
            self._base_dir = "./summaries/{}-{}".format(
                self.__name__,
                datetime.datetime.strftime("%Y-%m-%dT%H:%M:%S"))

        self._environment = environment

        self._summary_writer = tf.summary.FileWriter(self._base_dir)
        self.sess = tf.Session(
            '',
            config=tf.ConfigProto(allow_soft_placement=True))
        self._summary_writer.add_graph(graph=tf.get_default_graph())
        self._agent = agent_fn(
            sess=self.sess,
            summary_writer=self._summary_writer)
        self.sess.run(tf.global_variables_initializer())

    def run_experiment(self):
        """Runs a full experiment and at conclusion plots the important
        statistics.
        """

        self._run_one_episode()
        # Plot equity curve of the agent
        self._plot_equity_curve()
        # Printing the Statistics
        self._print_statistics()

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
            if not done:
                action = self._agent.step(reward, observation)
        self._end_episode(reward, observation)

    def _initialize_episode(self):
        """Initialization for a new episode.

        Returns:
            action: int the initial action chosen by the agent.
        """
        self.index = []
        self.rewards = []
        self.benchmark_rewards = []
        self.num_steps = 0
        initial_observation = self._environment.reset()
        return self._agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        """Executes a single step in the environment.

        Args:
          action (int):
            the action to perform in the environment.

        Returns:
          The observation, reward, and is_terminal values returned from the
            environment.
        """
        observation, reward, done, info = self._environment.step(action)
        self.num_steps += 1
        self.index.append(observation[0])
        self.rewards.append(reward)
        self.benchmark_rewards.append(info['return'])

        agent_rewards = tf.Summary()
        agent_rewards.value.add(
            tag="agent_rewards",
            simple_value=np.sum(self.rewards))

        benchmark_rewards = tf.Summary()
        benchmark_rewards.value.add(
            tag="benchmark_rewards",
            simple_value=np.sum(self.benchmark_rewards))

        self._summary_writer.add_summary(agent_rewards, self.num_steps)
        self._summary_writer.add_summary(benchmark_rewards, self.num_steps)
        return observation, reward, done

    def _end_episode(self, reward, observation):
        """Finalizes an episode run.

        Args:
          reward: float, the last reward from the environment.
        """
        self._agent.end_episode(reward, observation)

    def _plot_equity_curve(self):
        """Plots the Equity Curve of Agent vs Benchmark"""

        df = pd.DataFrame(
            {
                'Agent': np.cumsum(self.rewards),
                'Benchmark': np.cumsum(self.benchmark_rewards)
            },
            index=self.index)
        df.plot(title="Performance: Agent vs Benchmark")

    def _print_statistics(self):
        """Prints out the relevant experiment statistics"""
        print("Total Steps: {}".format(self.num_steps))
        print("Total Reward: {}".format(np.sum(self.rewards)))
