# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np
from keras.models import model
from keras.layers import input,Dense
from keras.optimizers import Adam
import keras import backend as K
import pandas as pd
import collections
import random


NORMALIZATION_WINDOW = 96
STACK_SIZE = 8


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
                spread=0.00005):
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
        """Returns the market feature at the current timestep"""
        log_ret = np.log(self.time_series) - np.log(self.time_series.shift(1))
        market_features = log_ret[-self.stack_size:]
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
        initial_series_data = np.ones(9) * initial_observation[1]
        ts = initial_observation[0]
        initial_series_index = []
        for i in range(9):
            temp_ts = ts - pd.Timedelta(days=i)
            initial_series_index.append(temp_ts)

        self.time_series = pd.Series(
            initial_series_data, index=initial_series_index
        )
        time_features = self._get_time_features(initial_observation[0])
        market_features = self._get_market_features()

        self.last_states = []
        for action in range(self.num_actions):
            position_features = np.zeros(self.num_actions)
            position_features[action] = 1
            state = np.concatenate(
                (time_features, market_features, position_features)
            )
            self.last_states.append(state)

    def _append_time_series(self, observation):
        """Appends a single timestep to the time series"""
        new_series = pd.Series(observation[1], index=[observation[0]])
        self.time_series = pd.concat([self.time_series, new_series])

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
        self._append_time_series(observation)

        # Get the time features and market features
        time_features = self._get_time_features(observation[0])
        market_features = self._get_market_features()

        log_return = np.log(self.time_series[-1]/self.time_series[-2])
        self.next_states = []
        experiences = []

        for action in range(self.num_actions):
            position_features = np.zeros(self.num_actions)
            position_features[action] = 1
            next_state = np.concatenate(
                (time_features, market_features, position_features)
            )
            self.next_states.append(next_state)
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


    def get_current_state(self, last_action):
        """Returns the current state of the agent

        Args:
            last_action (int): Takes the last action of the agent and uses that
            to find the current state of the agent from the array of
            last_states which contains the state of all possible actions.
        """
        return self.last_states[last_action]


class PG_agent(object):
    """
    This agent is based on a deep Q-learning architecture specified in Yi 2018
    """

    def __init__(
                self,
                num_actions=3,
                gamma=0.99,
                replay_buffer_size=1000,
                learning_rate=0.00025,
                tau=0.001,
                batch_size=96,
                online_update_period=8,
                target_update_period=96):
        """
        Initialises the agent and assigns values for all the hyperparameters

        Args:
            num_actions (int): number of actions agent can take at any state
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

        # Initialise the variables and hyperparameters
        self.num_actions = num_actions
        self.gamma = gamma
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.online_update_period = online_update_period
        self.target_update_period = target_update_period
        self.gamma = gamma

        self.state_shape = 17

        # Initiailze the ReplayBuffer
        self._replay = ReplayBuffer(
            replay_buffer_size=self.replay_buffer_size,
            num_actions=self.num_actions)

        # Build the network
        self.online_network = self._build_network(name="online")

        # Initiailze the internal_variables
        self.action = None
        self.total_steps = 0

   def build_feature(self):



   def path_score(self,path):#path is a list of (state,actions) tuples



   def log_prob(self,state,action):
       P = K.log(model(state)[action])
       return P





   def path_loss(self,path):
       """
       Compute - path_score*log_prob{path}
       """
       Path_loss = K.constant(0)
       for state,action in path:
           Path_loss += log_prob(state,action)
       Path_loss *= path_score(path)
       return Path_loss







   def build_path(self):
       """
       Builds a path of length l by sampling actions from the models current
       action predictions for l steps and records the state,action sequence.
       Returns the path loss.
       """




   def sample_paths(self):
       """
       Builds n paths from a uniformly sampled starting point.
       """



   def loss(y_true, y_pred):
       l = K.constant(0)
       sample = self.sample_paths()#returns a list of self.n paths of length self.l
       for path in sample:
           l += path_loss(path) #path_loss returns a tensor
       l = l/self.n
       return l




    def build_network(self, name=None):
        inputs = Input(shape=(,))
        # model.add(LSTM(24, input_shape = state_shape))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='Adam',
              loss=self.loss)




    def select_action(self):




    # Returns the agent's first action for the episode
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
        self._train_step()

        return self.action

    def step(self, reward, observation):
        """
        Records the most recent transition into replay buffer and return's the
        agent's next action
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

    def _record_observation(self, observation):
        """
        Adds the obsevation to the replay buffer and gets the current state
        from the replay_buffer
        """
        self._replay.add_observation(observation)
        self.current_state = self._replay.get_current_state(
            last_action=self.action
        )
