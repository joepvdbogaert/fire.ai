from __future__ import division
import copy
import numpy as np
import tensorflow as tf

from spyro.core import BaseAgent
from spyro.builders import build_dqn
from spyro.utils import obtain_env_information, make_env, progress


class DQNAgent(BaseAgent):
    """Agent that implements Deep Q-Networks (Mnih et al., 2015).

    Parameters
    ----------
    policy: an implementation of spyro.policies.BasePolicy
        The action-selection or exploration-exploitation policy. This must be an initialized
        object.
    memory: an implementation of spyro.memory.BaseMemory
        The replay buffer to store experience tuples in and sample from for training. Must
        be initialized.
    use_target_network: bool, default=True
        Whether to use a target network to predict the update targets.
    double_dqn: bool, default=True
        Whether to use Double DQN (Hasselt et al., 2016); predict the next action using the online network, but the
        value of the next action with the target network. This is used to reduce the
        maximization bias in Q-learning.
    dueling_dqn: bool, defualt=True
        Whether to use dueling network architectures as proposed by Wang et al (2015).
    batch_size: int, default=32
        The batch size to use for training the network.
    tau: float or int, default=0.02
        When ;math:`\\tau < 1`, this is the parameters of the soft-update of the target weights
        so that :math:`\\theta^- = \tau \\theta + (1 - \tau)\\theta^-`. When :math:`\\tau >= 1`
        it represents the number of steps between hard-updates of the target weights.
    train_frequency: int, default=1
        The number of steps in the environment between training steps of the network. If 1,
        trains after every step.
    td_lambda: int, optional default=0
        The number of steps to incorporate explicitly in the return calculation, i.e., the 'n'
        in the n-step Temporal Difference update.    
    n_layers: int, optional, default=3
        The number of hidden layers to use in the neural network architecture.
    n_neurons: int, optional, default=512
        The number of neurons to use per hidden layer in the neural network.
    activation: str or tensorflow callable, default="adam"
        The activation function to use for hidden layers in the neural network.
    value_neurons: int, default=64
        The number of neurons to use in the hidden layer of the value stream when
        dueling_dqn=True.
    advantage_neurons: int, default=256
        The number of neurons to use in the hidden layer of the advantage stream when
        dueling_dqn=True.
    name: str, optional, default="DQN_Agent"
        The name of the agent, which is used to define its variable scope in the Tensorflow
        graph to avoid mix-ups with other tensorflow activities.
    learning_rate: float, optional, default=1e-3
        The learning rate of the RMSProp optimizer.
    gamma: float, optional, default=0.9
        The discount factor for future rewards. At time :math:`t`, the reward :math:`r_{t + i}`
        is discounted by :math:`\\gamma^i` in the return calculation of :math:`t`, for all
        :math:`i = 1, ..., td_steps`.
    *args, **kwargs: any
        Any parameters that should be passed to spyro.agents.BaseAgent.

    References
    ---------- 
    [Hasselt et al. (2016)](https://arxiv.org/abs/1509.06461)
    [Mnih et al. (2015)](http://dx.doi.org/10.1038/nature14236)
    [Wang et al. (2015)](http://arxiv.org/abs/1511.06581)
    """
    def __init__(self, policy, memory, use_target_network=True, double_dqn=True, dueling_dqn=True, batch_size=32,
                 tau=0.02, train_frequency=1, td_lambda=0, n_layers=2, n_neurons=512, activation="relu",
                 value_neurons=64, advantage_neurons=256, optimization="adam", name="DQN_Agent", *args, **kwargs):

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
                    self.next_action_qvalue = tf.reduce_sum(self.next_target_qvalues * self.next_action_one_hot, axis=1, keepdims=True)

                else:  # no double dqn, use only the target network to predict next state value
                    self.next_action_qvalue = tf.reduce_max(self.next_target_qvalues, keepdims=True, axis=1)

            else:  # no target network, do everything with the online network
                with tf.variable_scope("online", reuse=True):
                    self.next_online_qvalues = build_dqn(self.next_states_ph, self.n_actions, **build_dqn_params)
                    self.next_action_qvalue = tf.reduce_max(self.next_online_qvalues, axis=1, keepdims=True, name="next_action_qvalue")
                    # self.next_online_qvalues = tf.identity(self.next_online_qvalues, name="next_online_qvalues")

            self.targets = self.rewards_ph + self.gamma * self.next_action_qvalue
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
            loss_summary = tf.summary.scalar("loss", self.loss)
            qvalue_summary = tf.summary.histogram("qvalues", self.online_qvalues)
            action_summary = tf.summary.scalar("prop_no_relocation",
                tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.actions_ph), 0), tf.float64))
            )
            self.step_summary = tf.summary.merge([loss_summary, qvalue_summary, action_summary])
            self.summary_writer = tf.summary.FileWriter(self.logdir, self.session.graph)

        self.session.run(tf.global_variables_initializer())

    def fit(self, env_cls, total_steps=4e7, warmup_steps=10000, tmax=None, env_params=None, restart=True):
        """Train the agent on a given environment.

        Parameters
        ----------
        env_cls: uninitialized Python class or str
            The environment to train on. If a class is provided, it must be uninitialized.
            Parameters can be passed to the environment using env_params. If a string
            is provided, this string is fed to `gym.make()` to create the environment.
        total_steps: int, optional, default=50,000
            The total number of training steps.
        warmup_steps: int, default=10000
            The number of steps to take in the environment before starting to train the
            network. This is used to build up a replay buffer with varying experiences
            to ensure uncorrelated samples from the start of training.
        tmax: int, optional, default=32
            The maximum number of steps to run in a single trial if the episode is not
            over earlier. After tmax steps, all end-of-episode tasks will be performed.
        env_params: dict, optional, default=None
            Dictionary of parameter values to pass to `env_cls` upon initialization.
        restart: boolean, default=True
            Whether to (re-)initialize the network (True) or to keep the current neural
            network parameters (False).
        """
        self.warmup_steps = warmup_steps
        self.tmax = tmax
        self.env = make_env(env_cls, env_params)
        self.action_shape, self.n_actions, self.obs_shape, _ = \
                obtain_env_information(env_cls, env_params)

        if restart:
            tf.reset_default_graph()
            self.total_steps = total_steps
            self._init_graph()
            if self.use_target_network:
                self.hard_update_target_network()

            self.episode_counter = 0
            self.step_counter = 0
            self.done = True
        else:
            try:
                self.total_steps += total_steps
            except AttributeError:
                # counters do not exist, graph must be restored from disk
                # recreate the counters like on a normal restart
                self.total_steps = total_steps
                self.step_counter = 0
                self.done = True
                self.episode_counter = 0

        while self.step_counter < total_steps:
            self.run_session()
            self.episode_counter += 1
            print("\rSteps completed: {}/{}".format(self.step_counter, self.total_steps), end="")

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
        """Sets the weights of the target network equal to the weights of the
        current online network.
        """
        self.session.run(self.hard_update_ops)

    def soft_update_target_network(self):
        """Moves the weights of the target network towards the weights of the online
        network by linearly interpolating between the two for every variable.
        """
        self.session.run(self.soft_update_ops)

    def train(self):
        """Perform a train step using a batch sampled from memory."""
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)
        # reshape to minimum of two dimensions
        step_summ, _ = self.session.run(
            [self.step_summary, self.train_op],
            feed_dict={
                self.states_ph: states.reshape(-1, *self.obs_shape),
                self.actions_ph: actions.reshape(-1, *self.action_shape),
                self.rewards_ph: rewards.reshape(-1, 1),
                self.next_states_ph: next_states.reshape(-1, *self.obs_shape),
                self.dones_ph: dones.reshape(-1, 1)
            }
        )
        self.summary_writer.add_summary(step_summ, self.step_counter)

    def evaluate(self, env_cls, n_episodes=10000, tmax=None, policy=None, env_params=None, init=False):
        """Evaluate the agent on an environemt without training.

        Parameters
        ----------
        env_cls: uninitialized Python class or str
            The environment to train on. If a class is provided, it must be uninitialized.
            Parameters can be passed to the environment using env_params. If a string
            is provided, this string is fed to `gym.make()` to create the environment.
        n_episodes: int, optional, default=10,000
            The number of episodes to run.
        tmax: int, optional, default=None
            The maximum number of steps to run in each episode. If None, set to 10,000 to
            not enforce a limit in most environments.
        policy: spyro.policies instance, default=None
            The policy to use during evaluation if it is not the same as during training.
        env_params: dict, optional, default=None
            Dictionary of parameter values to pass to `env_cls` upon initialization.
        init: boolean, default=False
            Whether to (re-)initialize the network (True) or to keep the current neural
            network parameters (False).
        """
        if policy is not None:
            self.eval_policy = policy
        else:
            self.eval_policy = self.policy

        if tmax is None:
            self.tmax = 10000
        else:
            self.tmax = tmax

        self.env = make_env(env_cls, env_params)
        self.action_shape, self.n_actions, self.obs_shape, _ = \
                obtain_env_information(env_cls, env_params)

        if init:
            tf.reset_default_graph()
            self._init_graph()

        self.episode_counter = 0
        self.step_counter = 0
        self.done = True

        self.eval_results = {
            "total_episode_reward": np.zeros(n_episodes),
            "mean_episode_reward": np.zeros(n_episodes),
            "episode_length": np.zeros(n_episodes),
        }

        for ep in range(n_episodes):
            self.state = np.asarray(self.env.reset(), dtype=np.float64)
            self.episode_step_counter = 0
            self.episode_reward = 0

            for i in range(self.tmax):

                # predict Q-values Q(s,a)
                qvalues = self.session.run(
                    self.online_qvalues,
                    feed_dict={self.states_ph: np.reshape(self.state, (1, -1))}
                )

                # select and perform action
                self.action = self.eval_policy.select_action(qvalues.reshape(-1))
                new_state, self.reward, self.done, _ = self.env.step(self.action)

                # bookkeeping
                self.step_counter += 1
                self.episode_reward += self.reward
                self.episode_step_counter += 1
                self.state = np.asarray(copy.copy(new_state), dtype=np.float64)

                # end of episode
                if self.done:
                    break

            self.eval_results["total_episode_reward"][ep] = self.episode_reward
            self.eval_results["mean_episode_reward"][ep] = self.episode_reward / self.episode_step_counter
            self.eval_results["episode_length"][ep] = self.episode_step_counter

            progress("Completed episode {}/{}".format(ep + 1, n_episodes),
                     same_line=(ep > 0), newline_end=(ep + 1 == n_episodes))

        return self.eval_results

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
