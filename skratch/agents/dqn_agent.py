# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
import collections
import random
import logging


NORMALIZATION_WINDOW = 96
STACK_SIZE = 8

# Setting logging config to INFO
logging.basicConfig(level=logging.INFO)


class ReplayBuffer(object):
    """
    Replay Buffer which creates and stores the observations
    at each time step and returns the state using the observation time
    series
    """

    def __init__(
                self,
                replay_buffer_size=1000,
                num_actions=3,
                stack_size=8,
                spread=0.000008):
        """
        Initialises a Replay Buffer .with the following parameters

        The replay buffer takes care of generating the states from the
        observation
        """

        self.replay_buffer_size = replay_buffer_size
        self.num_actions = num_actions
        self.last_states = None
        self.stack_size = stack_size
        self.spread = spread

        self._buffer = collections.deque(maxlen=self.replay_buffer_size)
        self._add_count = 0

    def _get_time_features(self, timestamp):
        """Returns the time features for a time step

        Args:
            timestamp (pandas.timestamp): uses a pandas timestamp object to
            extract minute, hour and wekday

        Returns:
            time_features (np.array, shape=6): an array of cyclical time
            features generated using sin and cos encoding of the minute, hour
            and the weekday.
        """

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
        """Returns the market feature at the current timestep, which are the
        log returns of the past stack_size number of close price and volumes.
        The final features are flattened array

        Returns:
            market_features (np.array): A flattened numpy array of log returns
                of the close_price and volume with the length equal to the
                stack_size.
        """
        log_ret = np.log(self.time_series/self.time_series.shift(1))
        market_features = log_ret[-self.stack_size:].values
        market_features = market_features.flatten()
        return market_features

    def _get_position_features(self, action):
        """Returns the position feature based on the agent's last action"""
        position_features = np.zeros(3)
        position_features[action] = 1
        return position_features

    def store_iniital_observation(self, initial_observation):
        """Initialises the time series with the first observation

        Also, creates the last_states for the first time when get_experiences
        is called in subsequent steps, they assume that last_states already
        has some values.
        """

        initial_size = self.stack_size
        temp_ts = initial_observation[0]
        initial_index = []

        # Index Initialisation
        for i in range(initial_size):
            temp_ts = temp_ts - pd.Timedelta(minutes=15)
            initial_index.append(temp_ts)

        # Values Initialisation
        dtype = [('close', np.float64), ('volume', np.int64)]
        initial_values = np.array(
            [(initial_observation[1], initial_observation[2])] * initial_size,
            dtype=dtype
        )

        # Dataframe initialisation using Values + Index
        self.time_series = pd.DataFrame(initial_values, index=initial_index)
        self.time_series.sort_index(inplace=True)

        # Construct the first set of states
        self.last_states = self._construct_states(initial_observation)

    def _append_time_series(self, observation):
        """Appends a single timestep to the time series"""
        dtype = [('close', np.float64), ('volume', np.int64)]
        values = np.array(
            [(observation[1], observation[2])],
            dtype=dtype)
        new_df = pd.DataFrame(values, index=[observation[0]])
        self.time_series = pd.concat([self.time_series, new_df])

    def _construct_states(self, observation):
        """Adds the current observation to the time series and construct the
        current set of states from the series"""
        self._append_time_series(observation)
        time_features = self._get_time_features(observation[0])
        market_features = self._get_market_features()

        states = []
        for action in range(self.num_actions):
            position_features = np.zeros(self.num_actions)
            position_features[action] = 1
            state = np.concatenate(
                (time_features, market_features, position_features)
            )
            states.append(state)

        return states

    def add_observation(self, observation):
        """
        Stores the observation and creates experiences by performing action
        augumentation and stores them in the buffer

        Action augumentation: Creating more experiences from actual experiences
        which reduces the exploration as it uses the reward from the current
        timestamp and modifies it achieve 3x3 or 9 more experiences from the
        one single experience. Therefore, expediting the training of the agent.
        """

        # Append observation to time series
        self.next_states = self._construct_states(observation)
        log_return = np.log(
            self.time_series['close'][-1]/self.time_series['close'][-2])
        experiences = []
        for action, next_state in enumerate(self.next_states):
            step_return = log_return * (action - 1)
            for last_action, last_state in enumerate(self.last_states):
                commission = self.spread * np.abs(action - last_action)
                reward = step_return - commission
                experiences.append((last_state, action, reward, next_state))
        self.last_states = self.next_states

        # Add all the experiences to the replay buffer
        for experience in experiences:
            self._add_count = min(self._add_count + 1, 1000)
            self._buffer.append(experience)

    def sample(self, batch_size=96):
        """
        Returns a batch of the experiences sampled randomly from the buffer
        """
        batch = random.sample(self._buffer, batch_size)
        return batch

    def get_current_state(self, last_action):
        """Returns the current state of the agent

        Args:
            last_action (int): Takes the last action of the agent and uses that
                to find the current state of the agent from the array of
                last_states which contains the state of all possible actions.
        """
        return self.last_states[last_action]


class DQNAgent(object):
    """
    This agent is based on a deep Q-learning architecture specified in Yi 2018
    """

    def __init__(
        self,
        num_actions=3,
        stack_size=8,
        gamma=0.99,
        replay_buffer_size=2000,
        learning_rate=0.025,
        tau=0.01,
        batch_size=128,
        online_update_period=32,
        target_update_period=96
    ):
        """
        Initialises the agent and assigns values for all the hyperparameters

        Parameters:
            num_actions (int): number of actions agent can take at any state
            stack_size (int): number of past observations to use for creating
                the state
            gamma (float): discount factor with the usual RL meaning.
            replay_buffer_size (int): Size of the replay buffer for storing the
                experiences
            learning_rate (float): learning_rate to be used for the optimizer
            tau (float): tau value to be used when making soft update to the
                weights of the target_network
            batch_size (int): number of experiences to be used in one training
                step
            online_update_period (int): update period for the online network.
            target_update_period (int): update period for the target network.
        """

        # Log all parameters to the console
        logging.info('Creating {} agent with the following parameters:'.format(
            self.__class__.__name__)
        )
        logging.info("num_actions: {}".format(num_actions))
        logging.info("stack_size: {}".format(stack_size))
        logging.info("gamma: {}".format(gamma))
        logging.info("replay_buffer_size: {}".format(replay_buffer_size))
        logging.info("learning_rate: {}".format(learning_rate))
        logging.info("tau: {}".format(tau))
        logging.info("batch_size: {}".format(batch_size))
        logging.info("online_update_period: {}".format(online_update_period))
        logging.info("target_update_period: {}".format(target_update_period))

        # Initialise the variables and hyperparameters
        self.num_actions = num_actions
        self.stack_size = stack_size
        self.gamma = gamma
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.online_update_period = online_update_period
        self.target_update_period = target_update_period

        # Shape of time_features + Shape of market_features + Shape of
        # position features
        self.state_shape = (6 + (self.stack_size * 2) + 3)

        # Initiailze the ReplayBuffer
        self._replay = ReplayBuffer(
            replay_buffer_size=self.replay_buffer_size,
            num_actions=self.num_actions,
            stack_size=self.stack_size)

        # Build the online_network and target_network
        self.online_network = self._build_network(name="online")
        self.target_network = self._build_network(name="target")

        # Initiailze the internal_variables
        self.action = None
        self.total_steps = 0

    def _build_network(self, name=None):
        """Returns a standard model for training the Q-network"""
        model = Sequential(name=name)
        model.add(Dense(24, input_dim=self.state_shape, activation='elu'))
        model.add(Dense(128, activation='elu'))
        # model.add(LSTM(1, input_shape = (24,1)))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(
                    loss='mse',
                    optimizer=Adam(lr=self.learning_rate))
        return model

    def begin_episode(self, initial_observation):
        """
        Similar to the step function, except that it just receives the
        obsevation as the parameter and returns a uniformly selected random
        action.
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
        self._replay.store_iniital_observation(initial_observation)
        self.action = np.random.randint(self.num_actions)
        # self._train_step()

        return self.action

    def step(self, reward, observation):
        """
        Records the most recent transition into replay buffer and return's the
        agent's next action
        Args:
            reward (float): the reward the agent recieved for it's last action
            observation (tuple): observation is a tuple of (timestamp, close,
                volume) that the agent uses to take the next action.
        Returns:
            action (int): the action that the agent wants to take at the
                current timestep.
        """
        self.total_steps += 1

        # Add the observation to the replay buffer
        self._record_observation(observation)

        # Perform the training of the networks
        self._train_step()

        self.action = self._select_action()
        return self.action

    def end_episode(self, reward, observation):
        """
        Records an end of episode. This method notes the reward and terminal
        observation and does not return an action unlike the step function.
        Args:
            reward (float): The terminal reward that the agent receives at the
                end of an episode.
            observation (tuple): The terminal observation that is emitted by
                the environment at the end of an episode.
        """
        pass

    def _select_action(self):
        """
        Selects a greedy action based on estimation of Q function.
        Returns:
            action (int): action taken by the agent for the current state
        """
        predict_batch = np.array([self.current_state])
        action_values = self.online_network.predict(predict_batch)
        return np.argmax(action_values[0])

    def _train_step(self):
        """Runs a single training step based on the update periods of both the
        online network

        And, updates the weights from online to target network if training
        steps is a multiple of target update period.
        """

        # Online Network
        # Train online network if step is a multiple of online_update_period
        if (self.total_steps % self.online_update_period) == 0:
            if (self._replay._add_count > self.batch_size):
                minibatch = self._replay.sample(batch_size=self.batch_size)

                train_batch = []
                target_batch = []
                for (state, action, reward, next_state) in minibatch:
                    current_state_batch = np.array([state])
                    next_state_batch = np.array([next_state])

                    # Get the Q-values
                    Q_online_current = self.online_network.predict(
                        current_state_batch)[0]
                    Q_target_next = self.target_network.predict(
                        next_state_batch)[0]
                    Q_online_next = self.online_network.predict(
                        next_state_batch)[0]

                    # Double DQN uses online network for argmax and target
                    # network for value of that argmax
                    next_action = np.argmax(Q_online_next)
                    target = reward + self.gamma * Q_target_next[next_action]

                    # Make the Bellman update on target (Y) value
                    Q_online_current[action] = target
                    train_batch.append(state)
                    target_batch.append(Q_online_current)

                # Train the online network on the minibatch
                X_train = np.array(train_batch)
                y_train = np.array(target_batch)
                self.online_network.fit(X_train, y_train, epochs=1)

        # Target Network
        # Update target weights if step is a multiple of target_update_period
        if (self.total_steps % self.target_update_period) == 0:
            self._update_target_weights()

    def _update_target_weights(self):
        """
        Makes a soft update to the weight of the target_network using the
        weights of the online_network
        """

        w1 = self.online_network.get_weights()
        w2 = self.target_network.get_weights()
        for i in range(len(w2)):
            w2[i] = (1 - self.tau) * w2[i] + self.tau * w1[i]
        self.target_network.set_weights(w2)

    def _record_observation(self, observation):
        """
        Adds the obsevation to the replay buffer and gets the current state
        from the replay_buffer
        """
        self._replay.add_observation(observation)
        self.current_state = self._replay.get_current_state(
            last_action=self.action
        )
