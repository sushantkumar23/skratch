# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np
import Tensorflow as tf
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
        log_ret = np.log(self.observation[1]) - np.log(self.past_observation[1])
        market_features = np.append(market_features,log_ret)
        market_features = market_features[-STACK_SIZE:]
        return market_features

    def _get_position_features(self,action = self.action):
        position_features = np.zeros(3)
        position_features[action + 1] = 1
        return position_features


    def build_state(self):
        m = _get_market_features()
        p = _get_position_feature()
        t = _get_time_features()
        state = np.append(m,p,t)
        return state



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




    def action_augmentation(self):
        experience =[]
        m = _get_market_features()
        t = _get_time_features()
        m1 = self.past_state[0:8]
        t1 = self.past_state[11:]
        for position in position_space:
            for action in action_space:
                past_state = np.append(m1,position,t1)
                p = _get_position_features(action)
                commission = self.spread * np.abs(action - self.past_action)
                step _return = (action-1)*(np.log(self.observation[0]) - np.log(self.past_observation[0]))
                reward = step_return - commission
                state = np.append(m,p,t)
                experience = [experience,(past_state,action,reward,state)]
                #expirience is a list of tuples .States are np arrays.
        return experience





#TODO - convert to series
       """
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

        self.last_states = self.next_states"""







class DQNAgent(object):
    """
    This agent is based on a deep Q-learning architecture specified in Yi 2018
    """

    def __init__(self,
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
        self.learning_rate =learning_rate
        self.tau = tau




    def build_replay_buffer(self):
        """Updates the replay buffer based on the new observations"""
        self.replay_buffer = self.replay_buffer.append(self.st.action_augmentation() )
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
        self.step += 1
        if self.step % self.training_timestep == 0:
            self._train_model()
        self._update_target_weights()
        self.action = self._select_action()
        self.past_state = self.state
        self.st = StateTransformer(observation,reward,self.action)
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
