from __future__ import division
import copy
import numpy as np
import tensorflow as tf

from spyro.core import BaseAgent
from spyro.builders import build_dqn
from spyro.utils import obtain_env_information, make_env


class DQNAgent(BaseAgent):

    def __init__(self, policy, memory, use_target_network=True, double_dqn=True, dueling_dqn=True, batch_size=32,
                 tau=0.5, train_frequency=4, td_lambda=0, n_layers=2, n_neurons=512, activation="relu",
                 value_neurons=64, advantage_neurons=256, optimization="sgd", name="DQN_Agent", *args, **kwargs):

        self.name = name

        # algorithm hyperparameters
        self.policy = policy
        self.memory = memory
        self.use_target_network = use_target_network
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.batch_size = batch_size
        self.tau = tau  # soft-update of target network
        self.train_frequency = train_frequency
        self.td_lambda = td_lambda
        self.optimization = optimization

        # network hyperparameters
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.value_neurons = value_neurons
        self.advantage_neurons = advantage_neurons

        super().__init__(policy, log_prefix=self.name + "_run", *args, **kwargs)

    def _init_graph(self):
        """Initialize the Tensorflow graph."""
        build_dqn_params = {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
            "dueling": self.dueling_dqn,
            "value_neurons": self.value_neurons,
            "advantage_neurons": self.advantage_neurons
        }

        self.session = tf.Session()
        with tf.variable_scope(self.name):

            self.states_ph = tf.placeholder(tf.float64, shape=(None,) + self.obs_shape, name="states_ph")
            self.actions_ph = tf.placeholder(tf.int64, shape=(None,) + self.action_shape, name="actions_ph")
            self.rewards_ph = tf.placeholder(tf.float64, shape=(None, 1), name="rewards_ph")
            self.next_states_ph = tf.placeholder(tf.float64, shape=(None,) + self.obs_shape, name="next_states_ph")
            self.dones_ph = tf.placeholder(tf.bool, shape=(None, 1), name="dones_ph")

            with tf.variable_scope("online"):
                self.online_qvalues = build_dqn(self.states_ph, self.n_actions, **build_dqn_params)
                # self.online_qvalues = tf.identity(self.online_qvalues, name="online_qvalues")

            if self.use_target_network:
                with tf.variable_scope("target"):
                    self.target_qvalues = build_dqn(self.states_ph, self.n_actions, **build_dqn_params)

                with tf.variable_scope("target", reuse=True):
                    self.next_target_qvalues = build_dqn(self.next_states_ph, self.n_actions, **build_dqn_params)

                if self.double_dqn:
                    # when using DDQN, use the action selected by online network, but its corresponding
                    # value predicted by the target network
                    with tf.variable_scope("online", reuse=True):
                        self.next_online_qvalues = build_dqn(self.next_states_ph, self.n_actions, **build_dqn_params)

                    self.next_action = tf.argmax(self.next_online_qvalues, axis=1)
                    self.next_action_one_hot = tf.one_hot(self.next_action, self.n_actions, dtype=self.next_target_qvalues.dtype)
                    self.next_action_target_qvalue = tf.reduce_sum(self.next_target_qvalues * self.next_action_one_hot, axis=1, keepdims=True)
                    self.targets = self.rewards_ph + self.gamma * self.next_action_target_qvalue

                else:  # no double dqn, use only the target network to predict next state value
                    self.targets = self.rewards_ph + self.gamma * tf.reduce_max(self.next_target_qvalues, keepdims=True, axis=1)

            else:  # no target network, do everything with the online network
                with tf.variable_scope("online", reuse=True):
                    self.next_online_qvalues = build_dqn(self.next_states_ph, self.n_actions, **build_dqn_params)
                    self.next_state_values = tf.reduce_max(self.next_online_qvalues, axis=1, keepdims=True, name="next_state_values")
                    # self.next_online_qvalues = tf.identity(self.next_online_qvalues, name="next_online_qvalues")

                self.targets = self.rewards_ph + self.gamma * self.next_state_values

            self.target = tf.stop_gradient(tf.where(self.dones_ph, x=self.rewards_ph, y=self.targets), name="target")
            # select the predicted q-values for the taken actions
            self.action_one_hot = tf.one_hot(self.actions_ph, self.n_actions, dtype=tf.float64, name="action_one_hot")
            self.qvalues_for_actions = tf.reduce_sum(self.online_qvalues * self.action_one_hot, axis=1, keepdims=True, name="qvalues_for_actions")
            # compute loss
            self.loss = 0.5 * tf.reduce_mean(tf.squeeze(tf.square(self.qvalues_for_actions - self.target)), name="loss")

            # update operations of target model (soft or hard)
            self.online_vars = tf.trainable_variables(scope=self.name + "/online")

            if self.use_target_network:
                self.target_vars = tf.trainable_variables(scope=self.name + "/target")

                self.hard_update_ops = [
                    tf.assign(target_var, online_var)
                    for target_var, online_var in zip(self.target_vars, self.online_vars)
                ]

                if self.tau < 1:
                    self.soft_update_ops = [
                        tf.assign(target_var, self.tau * online_var + (1 - self.tau) * target_var)
                        for target_var, online_var in zip(self.target_vars, self.online_vars)
                    ]

            # minimize the loss, possibly after clipping gradients
            self.optimizer = self.tf_optimizers[self.optimization](learning_rate=self.learning_rate)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.online_vars)
            if self.gradient_clip is not None:
                a_min, a_max = self.gradient_clip
                self.grads_and_vars = [(tf.clip_by_value(grad, a_min, a_max), var) for grad, var in self.grads_and_vars]
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
       
        # write to tensorboard
        if self.log:
            self.total_reward = tf.placeholder(tf.float64, shape=())
            self.mean_reward = tf.placeholder(tf.float64, shape=())
            total_reward_summary = tf.summary.scalar("total_episode_reward", self.total_reward)
            mean_reward_summary = tf.summary.scalar("mean_episode_reward", self.mean_reward)
            self.episode_summary = tf.summary.merge([total_reward_summary, mean_reward_summary])
            self.loss_summary = tf.summary.scalar("loss", self.loss)
            self.summary_writer = tf.summary.FileWriter(self.logdir, self.session.graph)

        self.session.run(tf.global_variables_initializer())
        print("Tensorflow graph trainable variables in online scope:")
        print([v.name for v in tf.trainable_variables(scope=self.name + "/online")])
        print("Tensorflow graph trainable variables in target scope:")
        print([v.name for v in tf.trainable_variables(scope=self.name + "/target")])

    def fit(self, env_cls, total_steps=4e7, warmup_steps=10000, tmax=None, env_params=None):
        """Train the agent on a given environment."""
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.tmax = tmax
        self.env = make_env(env_cls, env_params)
        self.action_shape, self.n_actions, self.obs_shape, _ = \
                obtain_env_information(env_cls, env_params)

        self._init_graph()
        if self.use_target_network:
            self.hard_update_target_network()

        self.episode_counter = 0
        self.step_counter = 0
        self.done = True
        while self.step_counter < total_steps:
            self.run_session()
            self.episode_counter += 1

    def run_session(self):
        """Run a single episode / trial, possibly truncated by t_max."""
        self.state = np.array(self.env.reset(), dtype=np.float64)
        self.episode_step_counter = 0
        self.episode_reward = 0

        for i in range(self.tmax):

            # predict policy pi(a|s) and value V(s)
            qvalues = self.session.run(
                self.online_qvalues,
                feed_dict={self.states_ph: np.reshape(self.state, (1, -1))}
            )

            # select and perform action
            self.action = self.policy.select_action(qvalues.reshape(-1))
            new_state, self.reward, self.done, _ = self.env.step(self.action)

            # print("State: {}, qvalues: {}, action: {}, reward: {}, new_state: {}".format(self.state, qvalues, self.action, self.reward, new_state))
            # save experience
            self.memory.store(copy.copy(self.state), self.action, self.reward, new_state, self.done)

            # train and possibly soft-update the target network
            if (self.step_counter % self.train_frequency == 0) and (self.step_counter > self.warmup_steps):
                self.train()
                if self.tau < 1:
                    self.soft_update_target_network()

            # possibly hard-update the target network
            if self.use_target_network and (self.step_counter % self.tau == 0):
                self.hard_update_target_network()

            # bookkeeping
            self.step_counter += 1
            self.episode_reward += self.reward
            self.episode_step_counter += 1
            self.state = np.asarray(copy.copy(new_state), dtype=np.float64)

            # end of episode
            if self.done:
                break

        episode_summary = self.session.run(
            self.episode_summary,
            feed_dict={
                self.total_reward: self.episode_reward,
                self.mean_reward: self.episode_reward / self.episode_step_counter
            }
        )
        self.summary_writer.add_summary(episode_summary, self.episode_counter)

    def hard_update_target_network(self):
        self.session.run(self.hard_update_ops)

    def soft_update_target_network(self):
        self.session.run(self.soft_update_ops)

    def train(self):
        """Perform a train step using a batch sampled from memory."""
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)
        # reshape to minimum of two dimensions
        loss_summ, _ = self.session.run(
            [self.loss_summary, self.train_op],
            feed_dict={
                self.states_ph: states.reshape(-1, *self.obs_shape),
                self.actions_ph: actions.reshape(-1, *self.action_shape),
                self.rewards_ph: rewards.reshape(-1, 1),
                self.next_states_ph: next_states.reshape(-1, *self.obs_shape),
                self.dones_ph: dones.reshape(-1, 1)
            }
        )
        self.summary_writer.add_summary(loss_summ, self.step_counter)

    def get_config(self):
        """Return configuration of the agent as a dictionary."""
        config = {
            "name": self.name,
            "policy": self.policy.get_config(),
            "memory": self.memory.get_config(),
            "use_target_network": self.use_target_network,
            "double_dqn": self.double_dqn,
            "dueling_dqn": self.dueling_dqn,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "train_frequency": self.train_frequency,
            "td_lambda": self.td_lambda,
            "optimization": self.optimization,
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
            "value_neurons": self.value_neurons,
            "advantage_neurons": self.advantage_neurons,
            "logdir": self.logdir,
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "gamma": self.gamma
        }
        return config
