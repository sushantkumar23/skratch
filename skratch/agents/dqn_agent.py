# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np
import tensorflow as tf
import pandas as pd


NORMALIZATION_WINDOW = 96
STACK_SIZE = 8


class StateTransformer(object):

    def __init__(self):


    def _get_time_features(self, timestamp):

        min = timestamp.to_pydatetime().minute
        min_sin = np.sin(min*(2.*np.pi/60))
        min_cos = np.cos(min*(2.*np.pi/60))

        hr = timestamp.to_pydatetime().hour
        hr_sin = np.sin(hr*(2.*np.pi/24))
        hr_cos = np.cos(hr*(2.*np.pi/24))

        day = timestamp.to_pydatetime().weekday()
        day_sin = np.sin(day*(2.*np.pi/7))
        day_cos = np.cos(day*(2.*np.pi/7))

        time_features = np.array(
            [min_sin, min_cos, hr_sin, hr_cos, day_sin, day_cos]
        )

        return time_features

    def _get_market_features(self):
        """Returns the market feature at the current timestep"""
        log_ret = np.log(self.time_series) - np.log(self.time_series.shift(1))
        market_features = log_ret[:STACK_SIZE]
        return market_features.values


    def iniitalise_state(self, initial_observation):
        """Initialises the time series with the first observation"""
        self.time_series = pd.Series(observation[1], index=[observation[0]])
        time_features = self._get_time_features(observation[0])
        market_features = self._get_market_features()

        self.last_states = []
        for action in range(self.num_actions):
            position_features = self.action_array[action]
            state = np.concatenate((time_features, market_features, position_features))
            self.last_states.append(state)


    def build_state(self, observation):
        new_series = pd.Series([observation[1]], index=[observation[0]])
        self.time_series.append(new_series)

    def get_experiences(self):
        time_features = self._get_time_features(observation[0])
        market_features = self._get_market_features()

        log_return = np.log(self.time_series[-1]/self.time_series[-2])
        self.next_states = []
        experiences = []

        for action in range(self.num_actions):
            position_features = self.action_array[action]
            next_state = np.concatenate((time_features, market_features, position_features))
            self.next_states.append(next_state)
            step_return = log_return * (action - 1)
            for last_action, last_state in enumerate(self.last_states):
                commission = self.spread * np.abs(action - last_action)
                reward = step_return - commission
                experiences.append((last_state, action, reward, next_state))

        self.last_states = self.next_states

        return experiences



class DQNAgent(object):
    """
    This agent is based on a deep Q-learning architecture specified in Yi 2018
    """

    def __init__(self,
                 replay_buffer_size = 1000,
                 learning_timestep = 96,
                 stack_size = 8,
                 gamma,
                 spread=0.00005
                 ):
        self.st = StateTransformer()
        self.action_array = np.identity(3)

        self._replay_buffer_size = replay_buffer_size
        self._replay = collections.deque(maxlen=self._replay_buffer_size)

    def build_state(self, observation, action, reward):
        """
        Takes the observations and builds the state spaces out of the observations.
        Parameters
        ----------
         observation (tuple):
             observation is received from the runner and the environment.
        """

        #How does time feature work in our case is it even important
        state = np.concatenate((time_features, market_features, position_features))
        state = tf.convert_to_tensor(state, dtype = tf.float32)

        return augmented_state

    def action_augmentation(self, observation):

        self.st.build_state(observation)
        experiences = self.st.get_experiences(observation)
        for experience in experiences:
            self._replay.append(experience)

    def _initialise_time_series(self, observation):
        """Initialises the time series with the first observation"""
        self.time_series = pd.Series(observation[1], index=[observation[0]])
        time_features = self._get_time_features(observation[0])
        market_features = self._get_market_features()

        self.last_states = []
        for action in range(self.num_actions):
            position_features = self.action_array[action]
            state = np.concatenate((time_features, market_features, position_features))
            self.last_states.append(state)


    def build_replay_buffer(self,state):
        """Updates the replay buffer based on the new observations"""
        self.replay_buffer = self.replay_buffer.append(state)
        self.replay_buffer = self.replay_buffer[-replay_history:]

    # Returns the agent's first action for the episode
    def begin_episode(self, initial_observation):
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
        self.st.iniitalise_state(initial_observation)

    def step(self, reward, observation):
        """
        Records the most recent transition into replay buffer and return's the
        agent's next action
        """
        self.action = self._select_action(reward, observation)
        self.state = build_state(self, observation, reward)
        build_replay_buffer(self, self.state)
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

    def _select_action(self,reward,observation):
        """
        trains the estimation and target networks by sampling from the existing replay buffer.
        Then selects a greedy action based on estimation of Q function.
        Returns:
        -------
        action taken by the agent each step
        """
        #estimation_model = keras.Sequential()
        #model.add(keras.layers.Dense(16, activation=tf.nn.elu))
        #model.add(keras.layers.Dense(16, activation=tf.nn.elu))
        #model.add(keras.layers.LSTM(


        input = sample(i)
        layer1 = tf.dense(inputs = input,units = 32,activation =
        tf.nn.elu)
        layer2 = tf.dense(inputs = layer1,units = 32,activation =
            tf.nn.elu)
        lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(1)
        initial_state = lstm.zero_state(batch_size, tf.float32)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(lstm, hidden3,initial_state=initial_state)
        output = fully_connected(lstm, 3 ,scope="outputs",activation_fn = softmax)
        sample = np.random.choice(self.replay_buffer, learning_timestep )
        for i in 1 to learning_timestep :
