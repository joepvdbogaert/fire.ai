from __future__ import division
import copy
import numpy as np
import tensorflow as tf

from spyro.agents import DQNAgent
from spyro.builders import build_distributional_dqn
from spyro.losses import quantile_huber_loss


class QuantileRegressionDQNAgent(DQNAgent):
    """Agent that implements Quantile Regression for Distributional Reinforcement Learning.

    Parameters
    ----------
    num_atoms: int, default=51
        The number of atoms/quantiles to predict.
    kappa: int, default=1
        The parameter of Huber Loss specifying until what absolute value an exponential
        loss is used as opposed to linear.
    name: str, default='QR_DQN_Agent'
        The name of the agent. Used in logging to Tensorboard.
    *args, **kwargs: any
        DQNAgent parameters.
    """
    def __init__(self, policy, memory, num_atoms=51, kappa=1, name="QR_DQN_Agent", *args, **kwargs):
        self.name = name
        self.num_atoms = num_atoms
        self.kappa = kappa
        super().__init__(policy, memory, name=name, *args, **kwargs)

    def _init_graph(self):
        """Initialize the Tensorflow graph."""
        build_dqn_params = {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
        }

        self.session = tf.Session()
        with tf.variable_scope(self.name):

            self.states_ph = tf.placeholder(tf.float64, shape=(None,) + self.obs_shape, name="states_ph")
            self.actions_ph = tf.placeholder(tf.int64, shape=(None,) + self.action_shape, name="actions_ph")
            self.rewards_ph = tf.placeholder(tf.float64, shape=(None, 1), name="rewards_ph")
            self.next_states_ph = tf.placeholder(tf.float64, shape=(None,) + self.obs_shape, name="next_states_ph")
            self.dones_ph = tf.placeholder(tf.bool, shape=(None, 1), name="dones_ph")

            # predict quantile q-values for current action
            with tf.variable_scope("online"):
                self.online_quantile_qvalues = build_distributional_dqn(self.states_ph, self.n_actions, self.num_atoms, **build_dqn_params)
            self.online_qvalues = tf.reduce_mean(self.online_quantile_qvalues, axis=2, name="online_qvalues", keepdims=False)  # for acting

            # predict quantile q-values for the next action
            with tf.variable_scope("online", reuse=True):
                self.next_online_quantile_qvalues = build_distributional_dqn(self.next_states_ph, self.n_actions, self.num_atoms, **build_dqn_params)

            # repeat for the target network if we use one
            if self.use_target_network or self.double_dqn:

                with tf.variable_scope("target"):
                    self.target_quantile_qvalues = build_distributional_dqn(self.states_ph, self.n_actions, self.num_atoms, **build_dqn_params)
                
                with tf.variable_scope("target", reuse=True):
                    self.next_target_quantile_qvalues = build_distributional_dqn(self.next_states_ph, self.n_actions, self.num_atoms, **build_dqn_params)

                if self.double_dqn:
                    # when using DDQN, use the action selected by online network, but its corresponding
                    # value predicted by the target network
                    self.next_action = tf.argmax(tf.reduce_mean(self.next_online_quantile_qvalues, axis=2), axis=1, name="next_action")
                else:
                    # else, use target network for both action selection and value/quantile prediction 
                    self.next_action = tf.argmax(tf.reduce_mean(self.next_target_quantile_qvalues, axis=2), axis=1, name="next_action")

                self.next_action_one_hot = tf.one_hot(self.next_action, self.n_actions, dtype=self.next_target_quantile_qvalues.dtype)
                self.next_action_quantiles = tf.reduce_sum(
                    tf.multiply(tf.expand_dims(self.next_action_one_hot, axis=-1), self.next_target_quantile_qvalues),
                    axis=1, keepdims=False, name="next_action_quantiles"
                )

            else:  # no target network, do everything with the online network
                self.next_action = tf.argmax(tf.reduce_mean(self.next_online_quantile_qvalues, axis=2), axis=1, name="next_action")
                self.next_action_one_hot = tf.one_hot(self.next_action, self.n_actions, dtype=self.next_online_quantile_qvalues.dtype)
                self.next_action_quantiles = tf.reduce_sum(
                    tf.multiply(tf.expand_dims(self.next_action_one_hot, axis=-1), self.next_online_quantile_qvalues),
                    axis=1, keepdims=False, name="next_action_quantiles"
                )

            # for checking
            self.next_action_quantiles_shape = tf.shape(self.next_action_quantiles)

            # Compute target values, which are of shape [batch_size, num_atoms] in QR-DQN
            # and copy along a new axis to compute loss between all pairs of target and predicted quantiles.
            # Note that we switch axis 1 and 2 to obtain a different direction than we do for the taken action
            # quantiles. This way, we obtain all the pairs Ttheta_j and theta_i.
            self.targets = self.rewards_ph + (1. - tf.cast(self.dones_ph, tf.float64)) * self.gamma * self.next_action_quantiles  # use broadcasting
            targets_tiled = tf.tile(self.targets, [1, self.num_atoms])
            targets_reshaped = tf.transpose(tf.reshape(targets_tiled, [tf.shape(targets_tiled)[0], self.num_atoms, self.num_atoms]), perm=[0, 2, 1])

            # Select the predicted quantile-values for the taken actions and copy along a new axis as well
            # to be able to compute loss between all pairs of Ttheta_j and theta_i.
            self.action_one_hot = tf.one_hot(self.actions_ph, self.n_actions, dtype=tf.float64, name="action_one_hot")
            self.taken_action_quantiles = tf.reduce_sum(
                tf.multiply(tf.expand_dims(self.action_one_hot, axis=-1), self.online_quantile_qvalues),
                axis=1, keepdims=False, name="taken_action_quantiles"
            )
            action_quants = tf.tile(self.taken_action_quantiles, [1, self.num_atoms])
            action_quants_reshaped = tf.reshape(action_quants, [tf.shape(action_quants)[0], self.num_atoms, self.num_atoms])

            # compute errors and loss
            self.errors = tf.subtract(tf.stop_gradient(targets_reshaped), action_quants_reshaped)
            self.loss = quantile_huber_loss(self.errors, kappa=self.kappa)

            self.online_vars = tf.trainable_variables(scope=self.name + "/online")

            # update operations of target model (soft or hard)
            if self.use_target_network or self.double_dqn:
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
            loss_summary = tf.summary.scalar("loss", self.loss)
            # qvalue_summary = tf.summary.histogram("qvalues", self.online_qvalues)
            action_summary = tf.summary.scalar("prop_no_relocation",
                tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.actions_ph), 0), tf.float64))
            )
            self.step_summary = tf.summary.merge([loss_summary, action_summary])
            self.summary_writer = tf.summary.FileWriter(self.logdir, self.session.graph)

        self.session.run(tf.global_variables_initializer())

    def get_config(self):
        """Return configuration of the agent as a dictionary."""
        config = {
            "name": self.name,
            "policy": self.policy.get_config(),
            "memory": self.memory.get_config(),
            "num_atoms": self.num_atoms,
            "kappa": self.kappa,
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
