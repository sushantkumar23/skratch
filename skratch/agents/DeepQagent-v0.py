# random_agent.py
# Copyright 2018 Skratch Authors.

import numpy as np
import tensorflow as tf




class DeepQagent-v0(object):
    """
    This agent is based on a deep Q-learning architecture specified in Yi 2018
    """

    def __init__(self,
                 replay_history = 480,
                 learning_timestep = 96,
                 stack_size = 8,
                 gamma,
                 ):
        self.state





    def build_state(self,observation,reward):
        """
        Takes the observations and builds the state spaces out of the observations.
        Parameters
        ----------
         observation (tuple):
             Observation is received from the runner and the environment.
        """

        #How does time feature work in our case is it even important
        datetime = observation[1]
        time_feature =
        close = np.log(observation[2])
        market_feature = market_feature.append(close)
        market_feature = market_feature[-stack_size:]
        position_feature = np.zeros(3)
        position_feature[self.action + 2] = 1
        state = (time_feature ,position_feature ,.market_feature )
        state = tf.convert_to_tensor(state, dtype = tf.float32)




        return augmented_state



    def build_replay_buffer(self,state):
        """Updates the replay buffer based on the new observations"""
        self.replay_buffer = self.replay_buffer.append(state)
        self.replay_buffer = self.replay_buffer[-replay_history:]





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

    def step(self, reward, observation):
        """
        Records the most recent trasition into replay buffer and return's the
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






        
























