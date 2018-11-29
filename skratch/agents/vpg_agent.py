# vpg_agent.py
# Copyright 2018 Skratch Authors.

import tensorflow as tf
import numpy as np
from scipy import stats


class VPGAgent(object):

    def __init__(
        self,
        sess,
        num_actions=3,
        stack_size=8,
        gamma=0.99,
        normalisation_window=96,
        training_period=96,
        episodes_per_epoch=8,
        steps_per_episode=72,
        pi_learning_rate=0.001,
        v_learning_rate=0.0003,
        train_v_iters=5,
        spread=0.000008,
        summary_writer=None
    ):

        self.__name__ = 'VPG'
        self.__version__ = "0.1.0"

        self.sess = sess
        self.num_actions = num_actions
        self.stack_size = stack_size
        self.gamma = gamma
        self.normalisation_window = normalisation_window
        self.training_period = training_period
        self.episodes_per_epoch = episodes_per_epoch
        self.steps_per_episode = steps_per_episode
        self.pi_learning_rate = pi_learning_rate
        self.v_learning_rate = v_learning_rate
        self.train_v_iters = train_v_iters
        self.spread = spread
        self._summary_writer = summary_writer

        self.epoch_returns = []
        self.total_steps = 0
        self.time_series = []
        self.series = []
        self.vol_series = []

        self.training_epochs = 0

        # Initialise the placeholders
        self.states_ph = tf.placeholder(
            shape=(None, self.stack_size*2 + 6 + 3),
            dtype=tf.float32)

        self._build_policy_network()
        self._build_v_network()
        self._summary_writer.add_graph(graph=tf.get_default_graph())

        self.sess.run(tf.global_variables_initializer())

    def _build_policy_network(self):

        # make core of policy network
        hidden1 = tf.layers.dense(self.states_ph, units=32,
                                  activation=tf.nn.relu)
        logits = tf.layers.dense(hidden1, units=self.num_actions,
                                 activation=None)

        # make action selection op (outputs int actions, sampled from policy)
        self.action_probs = tf.nn.softmax(logits)
        self.actions = tf.squeeze(
            tf.multinomial(logits=logits, num_samples=1), axis=1)

        # make loss function whose gradient, is policy gradient
        self.weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        action_masks = tf.one_hot(self.actions_ph, self.num_actions)
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits),
                                  axis=1)
        self.pi_loss = -tf.reduce_mean(self.weights_ph * log_probs)
        self.loss_summary = tf.summary.scalar(name='loss_summary',
                                              tensor=self.pi_loss)
        self.pi_train_op = tf.train.AdamOptimizer(
            learning_rate=self.pi_learning_rate).minimize(self.pi_loss)

    def _build_v_network(self):
        v_hidden = tf.layers.dense(self.states_ph, units=32,
                                   activation=tf.nn.relu)
        self.v_vals = tf.layers.dense(v_hidden, units=1, activation=None)
        self.ret_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v_vals)**2)
        self.v_train_op = tf.train.AdamOptimizer(
            learning_rate=self.v_learning_rate).minimize(self.v_loss)

    def train_one_epoch(self):

        self.training_epochs += 1
        # make some empty lists for logging.
        batch_states = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_q_vals = []

        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        for episode in range(self.episodes_per_epoch):
            start = (self.stack_size + 1)
            end = (len(self.series) - self.steps_per_episode) - 1
            index = np.random.randint(start, end)
            ep_rews = []            # list for rewards accrued throughout ep
            action = 1

            # collect experience by acting in environment with current policy
            for step in range(self.steps_per_episode):

                # save obs
                state = self.construct_state(index, action)
                batch_states.append(state)

                # act in the environment
                past_action = action
                action = self.sess.run(
                    self.actions,
                    {self.states_ph: np.array([state])})[0]

                index += 1
                step_return = np.log(self.series[index-1]/self.series[index-2])
                commission = self.spread * np.abs(past_action - action)
                reward = ((action - 1) * step_return) - commission
                normalised_reward = reward*1000

                next_state = self.construct_state(index, action)
                v_vals = self.sess.run(
                    self.v_vals,
                    {self.states_ph: np.array([state, next_state])})
                q_val = normalised_reward + (self.gamma * v_vals[1][0])

                advantage = q_val - v_vals[0][0]

                batch_q_vals.append(q_val)
                batch_weights.append(advantage)

                state = next_state
                # save action, reward
                batch_acts.append(action)
                ep_rews.append(reward)

            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

        # take a single policy gradient update step
        batch_loss, loss_summary, _ = self.sess.run(
            [self.loss, self.loss_summary, self.pi_train_op],
            feed_dict={
                self.states_ph: np.array(batch_states),
                self.actions_ph: np.array(batch_acts),
                self.weights_ph: stats.zscore(np.array(batch_weights))
            })

        for i in range(self.train_v_iters):
            v_loss, _ = self.sess.run(
                [self.v_loss, self.v_train_op],
                feed_dict={
                    self.states_ph: np.array(batch_states),
                    self.ret_ph: np.array(batch_q_vals)
                })

        if self._summary_writer is not None:
            return_summary = tf.Summary()
            return_summary.value.add(
                tag="return_summary",
                simple_value=np.mean(ep_rews))

            self._summary_writer.add_summary(return_summary,
                                             self.training_epochs)
            self._summary_writer.add_summary(loss_summary,
                                             self.training_epochs)

        print('epoch: \t loss: %.3f \t return: %.3f \t ep_len: %.3f'
              % (batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        self.epoch_returns.append(np.mean(batch_rets))

    def _reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def _select_action(self, state):
        action_probs = self.sess.run(
            self.action_probs,
            {self.states_ph: np.array([state])})[0]
        action = np.argmax(action_probs)
        return action

    def construct_state(self, index, action):
        series = self.series[:index]
        vol_series = self.vol_series[:index]
        time_series = self.time_series[:index]
        log_ret = np.log(np.array(series[1:])/np.array(series[:-1]))
        vol_log_ret = np.log(
            np.array(vol_series[1:])/np.array(vol_series[:-1]))
        normal_ret = stats.zscore(log_ret[-self.normalisation_window:])
        vol_normal_ret = stats.zscore(vol_log_ret[-self.normalisation_window:])

        time_features = self.get_time_features(time_series[-1])
        position_features = np.zeros(self.num_actions)
        position_features[action] = 1
        state = np.concatenate([
            time_features,
            normal_ret[-self.stack_size:],
            vol_normal_ret[-self.stack_size:],
            position_features
        ])

        # Clipping values between -10 and 10
        state = np.clip(state, -10, 10)
        return np.nan_to_num(state)

    def begin_episode(self, initial_observation):
        self.time_series.extend([initial_observation[0]] * self.stack_size)
        self.series.extend([initial_observation[1]] * self.stack_size)
        self.vol_series.extend([initial_observation[2]] * self.stack_size)

        self.action = 1
        state = self.get_state(initial_observation, self.action)
        return self._select_action(state)

    def step(self, reward, observation):
        self.total_steps += 1
        self.train_step()
        state = self.get_state(observation, self.action)
        self.action = self._select_action(state)
        return self.action

    def end_episode(self, reward, observation):
        state = self.get_state(observation, self.action)

    def get_state(self, observation, action):
        self.time_series.append(observation[0])
        self.series.append(observation[1])
        self.vol_series.append(observation[2])
        return self.construct_state(len(self.series), action)

    def train_step(self):
        if self.total_steps % self.training_period == 0:
            self.train_one_epoch()

    def get_time_features(self, timestamp):
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
