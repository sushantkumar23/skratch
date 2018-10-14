# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np
import tensorflow as tf
import pandas as pd

# dt = pd.Timestamp('2018-01-06')
# series = pd.Series([1.222], index=[dt])
# series
#
# dt.to_pydatetime().weekday()
#
# dt2 = pd.Timestamp('2018-01-02')
# new_series = pd.Series([1.333], index=[dt2])
# series.append(new_series)

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

    def get_experiences()



class DQNAgent(object):
    """
    This agent is based on a deep Q-learning architecture specified in Yi 2018
    """

    def __init__(self,
                 replay_buffer_size = 1000,
                 learning_timestep = 96,
                 stack_size = 8,
                 gamma,
                 spread=0.00005,
                 action_size = 3,
                 ):
        self.st = StateTransformer()
        self.action_array = np.identity(3)
        self.action_size = action_size
        self._replay_buffer_size = replay_buffer_size
        self._replay = collections.deque(maxlen=self._replay_buffer_size)
        self.state
        self.past_state
        self.model1 = build_model()
        self.model2 = build_model()
        self.action

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
        self._add_time_series(observation)
        time_features = self._get_time_features(observation[0])
        market_features = self._get_market_features()

        log_return = np.log(self.time_series[-1]/self.time_series[-2])
        self.next_states = []

        for action in range(self.num_actions):
            position_features = self.action_array[action]
            next_state = np.concatenate((time_features, market_features, position_features))
            self.next_states.append(next_state)
            step_return = log_return * (action - 1)
            for last_action, last_state in enumerate(self.last_states):
                commission = self.spread * np.abs(action - last_action)
                reward = step_return - commission
                self._replay.append((last_state, action, reward, next_state))

        self.last_states = self.next_states

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


    def build_replay_buffer(self,reward,action):
        """Updates the replay buffer based on the new observations"""
        self.replay_buffer = self.replay_buffer.append((self.past_state,reward,self.action,self.state))
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
        self._train_model()
        self.action = self._select_action(reward, observation)
        self.past_state = self.state
        self.state = build_state(observation, reward)
        build_replay_buffer(reward)
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
        Selects a greedy action based on estimation of Q function.
        Returns:
        -------
        action taken by the agent each step
        """
        act_value = self.model1.predict(state)
        return np.argmax(act_value)


    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='elu'))
        model.add(Dense(24, activation='elu'))
        model.add(LSTM(1, input_shape = (24,1)))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='mse',
                  optimizer=Adam(lr=self.learning_rate))
        return model




    def _train_model(self):
        minibatch = random.sample(self.replay_buffer, self.learning_timestep)
        for state, action, reward, next_state in minibatch :
            Q_next = self.model2.predict(next_state)
            target = reward + self.gamma * np.amax(Q_next)
            #train network
            self.model1.fit(state, target, epochs=1)
            self.model2.fit(state, target, epochs = 1)

# To do - Preprocess data , Write a separate function for training target network
# and make the training less frequent






