# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np
import Tensorflow as tf
import pandas as pd


NORMALIZATION_WINDOW = 96
STACK_SIZE = 8


class ReplayBuffer(object):

<<<<<<< HEAD
    def __init__(self,observation,action,reward):
        self.past_observation =  self.observation
        self.observation = observation
        self.past_action = self.action
        self.action = action
        self.reward = reward
        self.past_state = self.state
        self.state = self.build_state()

    def _get_time_features(self):
        timestamp = self.observation[0]
=======
    def __init__(
                self,
                replay_buffer_size=1000,
                num_actions=3,
                stack_size=8,
                spread=0.00005):
        """Initialises a Replay Buffer which creates and stores the observations
        at each time step and returns the state using the observation time
        series

        The replay buffer takes care of generating the states from the observation
        """

        self.replay_buffer_size = replay_buffer_size
        self.num_actions = num_actions
        self.last_states = None
        self.stack_size = stack_size
        self.spread = spread

        self._buffer = collections.deque(maxlen=self._replay_buffer_size)

    def _get_time_features(self, timestamp=timestamp):
        """Returns the time features for a time step

        Args:
            timestamp (pandas.timestamp): uses a pandas timestamp object to extract
            minute, hour and wekday

        Returns:
            time_features (np.array, shape=6): an array of cyclical time
            features generated using sin and cos encoding of the minute, hour
            and the weekday.
        """
>>>>>>> f-dqn
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
<<<<<<< HEAD
        log_ret = np.log(self.observation[1]) - np.log(self.past_observation[1])
        market_features = np.append(market_features,log_ret)
        market_features = market_features[-self.stack_size:]
        return market_features

    def _get_position_features(self, action = self.action):
        """Returns the position feature based on the agent's last action"""
        position_features = np.zeros(3)
        position_features[action + 1] = 1
        return position_features


    # def build_state(self):
    #     m = _get_market_features()
    #     p = _get_position_feature()
    #     t = _get_time_features()
    #     state = np.append(m, p, t)
    #     return state



    def iniitalise_state(self, initial_observation):
        """Initialises the time series with the first observation"""
=======
        log_ret = np.log(self.time_series[1]) - np.log(self.time_series[2])
        market_features = market_features[-self.stack_size:]
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
>>>>>>> f-dqn
        self.time_series = pd.Series(observation[1], index=[observation[0]])
        time_features = self._get_time_features(observation[0])
        market_features = self._get_market_features()

        self.last_states = []
        for action in range(self.num_actions):
            position_features = self.action_array[action]
            state = np.concatenate((time_features, market_features, position_features))
            self.last_states.append(state)


<<<<<<< HEAD

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
=======
    def _append_time_series(self, observation):
        """Appends a single timestep to the time series"""
        new_series = pd.Series(observation[1], index=[observation[0]])
        self.time_series = pd.Series.concat([self.time_series, new_series])


    def store_observation(self, observation):
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
            position_features = self.action_array[action]
            next_state = np.concatenate((time_features, market_features, position_features))
            self.next_states.append(next_state)
            step_return = log_return * (action - 1)
            for last_action, last_state in enumerate(self.last_states):
                commission = self.spread * np.abs(action - last_action)
                reward = step_return - commission
                experiences.append((last_state, action, reward, next_state))

        self.last_states = self.next_states

        # Add all the experiences to the replay buffer
        for experience in experiences:
            self._buffer.append(experience)

    def sample(self, batch_size=32):
        """
        Returns a batch of the experiences sampled randomly from the buffer
        """
        batch = random.sample(self.__buffer, self.learning_timestep)
        return batch

>>>>>>> f-dqn

        return experiences


class DQNAgent(object):
    """
    This agent is based on a deep Q-learning architecture specified in Yi 2018
    """

    def __init__(self,
<<<<<<< HEAD
                 replay_buffer_size = 1000,
                 training_timestep = 96,
                 stack_size = 8,
                 gamma,
                 spread=0.00005,
                 action_size = 3,
                 learning_rate = 0.00025,
                 tau = 0.001
                 ):
        self.action_array = np.identity(3)
        self.action_size = action_size
        self._replay_buffer_size = replay_buffer_size
        self._replay = collections.deque(maxlen=self._replay_buffer_size)
        self.state
        self.past_state
        self.model1 = build_model()
        self.model2 = build_model()
        self.action
        self.step = 0
        self.training_timestep = training_timestep
        self.learning_rate = learning_rate
        self.tau = tau


    def build_replay_buffer(self):
=======
                num_actions = 3,
                gamma=0.99,
                replay_buffer_size=1000,
                learning_rate=0.00025,
                tau=0.001
                online_update_period=96,
                target_update_period=96
                ):
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
            online_update_period (int): update period for the online network.
            target_update_period (int): update period for the target network.
        """

        # Initialise the variables and hyperparameters
        self.num_actions = num_actions
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.online_update_period = online_update_period
        self.target_update_period = target_update_period

        # Initiailze the ReplayBuffer
        self.st = ReplayBuffer(
            replay_buffer_size=self.replay_buffer_size,
            num_actions=self.num_actions)

        # Create the replay buffer

        # Build the online_network and target_network
        self.online_network = self._build_network(name="online")
        self.target_network = self._build_network(name="target")

        # Initiailze the internal_variables
        self.action = None
        self.step = 0

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


    def build_replay_buffer(self,reward,action):
>>>>>>> f-dqn
        """Updates the replay buffer based on the new observations"""
        self.replay_buffer = self.replay_buffer.append(self.st.action_augmentation() )
        self.replay_buffer = self.replay_buffer[-replay_history:]


    def _build_network(self, name=None):
        """Returns a standard model for training the Q-network"""
        model = Sequential(name=name)
        model.add(Dense(24, input_dim=self.state_size, activation='elu'))
        model.add(Dense(24, activation='elu'))
       # model.add(LSTM(1, input_shape = (24,1)))
        model.add(Dense(self.num_actions, activation='sigmoid'))
        model.compile(loss='mse',
                  optimizer=Adam(lr=self.learning_rate))
        return model

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
        self._replay.store_iniital_observation(initial_observation)
        self.action = self._select_action())
        self._train_step()

        return action

    def step(self, reward, observation):
        """
        Records the most recent transition into replay buffer and return's the
        agent's next action
        """
        self.step += 1
<<<<<<< HEAD
        if self.step % self.training_timestep == 0:
            self._train_model()
        self._update_target_weights()
        self.action = self._select_action()
        self.past_state = self.state
        self.st = StateTransformer(observation,reward,self.action)
        build_replay_buffer(reward)
=======
        self.action = self._select_action()

        # Add the observation to the replay buffer
        self._replay.store_observation(observation)

        # Perform the training of the networks
        self._train_step()

>>>>>>> f-dqn
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
        act_value = self.online_network.predict(state)
        return np.argmax(act_value)


<<<<<<< HEAD
    def _build_model(self):
        """Returns a standard model for training the Q-network"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='elu'))
        model.add(Dense(24, activation='elu'))
       # model.add(LSTM(1, input_shape = (24,1)))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss='mse',
                  optimizer=Adam(lr=self.learning_rate))
        return model

    def _train_model(self):
        minibatch = random.sample(self.replay_buffer, self.learning_timestep)
        for state, action, reward, next_state in minibatch :
            Q_next = self.model2.predict(next_state)
            a = argmax(self.model1.predict(next_state))
            target = reward + self.gamma * Q_next[a]
            #greedy action wrt model1 not model 2
            #train network
            self.model1.fit(state, target, epochs=1)

    def _update_target_weights(self):
        n = len(self.model1.layers)
        w1 = model1.get_weights()
        w2 = model2.get_weights()
        for i in range(len(w2)):
            w2 = (1 - self.tau)*w2 + self.tau * w1
        self.model2.set_weights(w2)
=======
    def _train_step(self):
        """Runs a single training step based on the update periods of both the
        online network

        And, updates the weights from online to target network if training steps
        is a multiple of target update period.
        """

        # Online Network
        # Train the online network if step is a multiple of online_update_period
        if (self.step % self.online_update_period) == 0:
            self._replay.sample(size=self.batch_size)
            for state, action, reward, next_state in minibatch :
                Q_next = self.target_network.predict(next_state)
                a = argmax(self.online_network.predict(next_state))
                target = reward + self.gamma * Q_next[a]
                #greedy action wrt online_network not target_network
                #train network
                self.online_network.fit(state, target, epochs=1)

        # Target Network
        # Update the target weights if step is a multiple of target_update_period
        if (self.step % self.target_update_period) == 0:
            self._update_target_weights()

    def _update_target_weights(self):
        """
        Makes a soft update to the weight of the target_network using the
        weights of the online_network
        """

        w1 = self.online_network.get_weights()
        w2 = self.target_network.get_weights()
        for i in range(len(w2)):
            w2 = (1 - self.tau)*w2 + self.tau * w1
        self.target_network.set_weights(w2)
>>>>>>> f-dqn
