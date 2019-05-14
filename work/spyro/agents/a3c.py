from __future__ import division
import warnings
import copy
import os
import pickle
import time

import numpy as np
import tensorflow as tf
import multiprocessing as mp
import queue
import gym

from spyro.core import BaseAgent
from spyro.targets import (n_step_temporal_difference,
                           monte_carlo_discounted_mean_reward,
                           n_step_discounted_mean_reward)

from spyro.utils import find_free_numbered_subdir, make_env, obtain_env_information, get_space_shape
from spyro.builders import build_actor_critic_mlp


class A3CWorker(mp.Process):
    """Worker agent for the A3C algorithm. This class is used by the A3CAgent class
    and not useful to substantiate on its own.

    Parameters
    ----------
    env_cls: an openai.gym environment class
        The environment to train on.
    global_T: multiprocessing.Value object
        Global counter of number of performed steps.
    global_queue: multiprocessing.Queue object
        The queue to send calculated gradients to.
    global_weights: multiprocessing.RawArray object
        The shared-memory array of weights of the global network/agent.
    weight_shapes: list of tuples
        The shapes of the trainable tensorflow variables. These are used to reshape weights
        from `global_weights` to their original form, since they are flattened in the
        shared-memory array.
    policy: an implementation of spyro.policies.BasePolicy()
        The action-selection or exploration-exploitation policy. This must be an initialized
        object.
    name: str, optional, default="A3C_Global_Agent"
        The name of the agent, which is used to define its variable scope in the Tensorflow
        graph to avoid mix-ups with other tensorflow activities.
    total_steps: int, optional, default=50,000
        The total number of steps to train of all workers together.
    tmax: int, optional, default=32
        The maximum number of steps for a worker to run before calculating gradients
        and updating the global model if the episode is not over earlier.
    beta: float, optional, default=0.05
        The contribution of entropy to the total loss of the actor. Specifically, the actor's
        gradient is calculated over :math:`log \\pi_{\\theta}(a_t | s_t)A(a_t, s_t) + \\beta H`
        where :math:`H` is the entropy.
    gamma: float, optional, default=0.9
        The discount factor for future rewards. At time :math:`t`, the reward :math:`r_{t + i}`
        is discounted by :math:`\\gamma^i` in the return calculation of :math:`t`, for all
        :math:`i = 1, ..., td_steps`.
    returns: str, one of ['mc_mean', 'n_step_mean', 'n_step_td']
        The return calculation to use. For 'mc_mean', uses Monte Carlo Discounted Mean Rewards,
        for 'n_step_mean' and 'n_step_td', calculates mean or sum of first n rewards respectively
        and then bootstraps the next state value.
    td_steps: int, optional default=10
        The number of steps to incorporate explicitly in the return calculation, i.e., the 'n'
        in the n-step Temporal Difference update.
    n_layers: int, optional, default=3
        The number of hidden layers to use in the neural network architecture.
    n_neurons: int, optional, default=512
        The number of neurons to use per hidden layer in the neural network.
    activation: str or tensorflow callable
        The activation function to use for hidden layers in the neural network.
    **env_params: key-value pairs
        Any parameters that should be used to initialize the environment.
    """
    metadata = {
        "return_funcs": {
            "mc_mean": monte_carlo_discounted_mean_reward,
            "n_step_mean": n_step_discounted_mean_reward,
            "n_step_td": n_step_temporal_difference
        }
    }

    def __init__(self, env_cls, global_T, global_queue, global_weights, weight_shapes,
                 policy, tmax=16, total_steps=1000000, name="A3C_Worker",
                 beta=0.01, gamma=0.9, returns="mc_mean", td_steps=10, n_layers=2, n_neurons=512,
                 activation="relu", learning_rate=1e-3, env_params=None, logdir="log", log=True):

        # init Process
        super().__init__()

        # override name
        self.name = name

        # create environment and save some info about it
        self.env_cls = env_cls
        self.env_params = env_params

        # save the policy
        self.policy = policy

        # the shared memory objects
        self.global_T = global_T
        self.global_queue = global_queue
        self.global_weights_rawarray = global_weights
        self.weight_shapes = weight_shapes  # not shared but used to reconstruct weights

        # hyperparameters
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.learning_rate = learning_rate
        self.beta = beta  # the contribution of entropy to the objective function
        self.tmax = tmax
        self.gamma = gamma
        self.calc_returns = self.metadata["return_funcs"][returns]
        self.td_steps = td_steps
        self.total_steps = total_steps

        # small value to prevent log(0)
        self.tiny_value = 1e-8

        # tensorboard
        self.logdir = os.path.join(logdir, self.name)
        self.log = log

    def _init_graph(self):
        """Build and initialize the Tensorflow computation graph."""
        self.session = tf.Session()

        with tf.variable_scope(self.name):
            
            with tf.variable_scope("shared"):
                self.state_ph = tf.placeholder(tf.float64, shape=(None,) + self.obs_shape, name="state_ph")
                
            # Predict action probabilities and state value with model. Note that the function
            # deals with putting parts of the model in the right variable scope.
            self.action_probs, self.value_pred = build_actor_critic_mlp(
                self.state_ph,
                self.n_actions,
                scope_shared="shared/",
                scope_actor="actor",
                scope_critic="critic",
                n_layers=self.n_layers,
                n_neurons=self.n_neurons,
                activation=self.activation
            )

            with tf.variable_scope("actor/"):
                # placeholders for taken action and computed advantage
                self.action_ph = tf.placeholder(tf.int32, shape=(None, 1), name="action_ph")
                self.actor_target_ph = tf.placeholder(tf.float64, shape=(None, 1), name="actor_target_ph")  # normally the Advantage

                # compute the entropy of the current policy pi(a|s) for regularization
                self.entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs + self.tiny_value), keepdims=True, axis=1)
                self.mean_entropy = tf.reduce_mean(self.entropy)

                # compute policy gradients
                self.action_one_hot = tf.reshape(tf.one_hot(self.action_ph, self.n_actions, dtype=tf.float64), (-1, self.n_actions)) 
                self.taken_action_prob = tf.reduce_sum(self.action_probs * self.action_one_hot, keepdims=True, axis=1)
                self.log_action_prob = tf.log(self.taken_action_prob + self.tiny_value)
                self.objective = self.log_action_prob * self.actor_target_ph + self.entropy * self.beta
                self.actor_max_obj = tf.reduce_mean(tf.squeeze(self.objective))
                self.actor_loss = - self.actor_max_obj
                # self.actor_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            with tf.variable_scope("critic/"):
                self.critic_target_ph = tf.placeholder(tf.float64, shape=(None, 1), name="critic_target_ph")  # normally the n-step Return G_{t:t+n}

                # compute value loss: (root) mean squared error
                self.value_loss = tf.reduce_mean(tf.squeeze(tf.square(self.critic_target_ph - self.value_pred)))

            # total loss
            self.total_loss = self.actor_loss + 0.5 * self.value_loss
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            
            # obtain rewards for publishing in Tensorboard
            self.rewards_ph = tf.placeholder(tf.float64, shape=(None, 1), name="rewards_ph")
            self.mean_reward = tf.reduce_mean(self.rewards_ph)
            self.total_reward = tf.reduce_sum(self.rewards_ph)

            # save the trainable variables for easy access
            self.var_list = tf.trainable_variables(scope=self.name)
            self.shared_var_list = tf.trainable_variables(scope=self.name + "/shared")
            self.actor_var_list = self.shared_var_list + tf.trainable_variables(scope=self.name + "/actor")
            self.critic_var_list = self.shared_var_list + tf.trainable_variables(scope=self.name + "/critic")

            # compute the gradients
            self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss, self.var_list)

            # placeholders and operation for setting weights of global model to local one
            self.set_weights_phs = [tf.placeholder(tf.float64) for _ in self.var_list]
            self.assign_ops = [tf.assign(var, self.set_weights_phs[i])
                               for i, var in enumerate(self.var_list)]

        if self.log:
            total_reward_summary = tf.summary.scalar("total_episode_reward", self.total_reward)
            mean_reward_summary = tf.summary.scalar("mean_episode_reward", self.mean_reward)
            actor_loss_summary = tf.summary.scalar("actor_loss", self.actor_loss)
            critic_loss_summary = tf.summary.scalar("value_loss", self.value_loss)
            total_loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            action_summary = tf.summary.scalar("prop_no_relocation",
                tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.action_ph), 0), tf.float64))
            )
            entropy_summary = tf.summary.scalar("entropy", self.mean_entropy)
            actor_probs_summary = tf.summary.histogram("policy_pi", self.action_probs)
            self.summary_op = tf.summary.merge(
                [total_reward_summary, mean_reward_summary, actor_loss_summary,
                 critic_loss_summary, actor_probs_summary, entropy_summary,
                 total_loss_summary, action_summary]
            )
            self.summary_writer = tf.summary.FileWriter(self.logdir, self.session.graph)

        self.session.run(tf.global_variables_initializer())

    def send_gradients(self, gradients):
        bytes_ = pickle.dumps(gradients, protocol=-1)
        self.global_queue.put(bytes_)

    def _pull_global_weights(self):
        if (self.global_weights_rawarray is not None) and (self.weight_shapes is not None):
            # print("pulling global weights for {}".format(self.name))
            self.global_weights = [np.frombuffer(w, dtype=np.float64).reshape(self.weight_shapes[i])
                                   for i, w in enumerate(self.global_weights_rawarray)]

    def set_weights(self, weights):
        self.session.run(self.assign_ops, feed_dict={ph: w for ph, w in zip(self.set_weights_phs, weights)})

    def obtain_global_weights(self):
        self._pull_global_weights()
        self.set_weights(self.global_weights)

    def run(self):
        """Start training on the environment and sending updates to the global agent."""
        self.action_shape, self.n_actions, self.obs_shape, _  = \
                obtain_env_information(self.env_cls, self.env_params)
        self._init_graph()
        self.env = make_env(self.env_cls, self.env_params)
        self.done = True  # force reset of the environment

        self.local_episode_counter = 0
        while self.global_T.value < self.total_steps:
            self.run_session()
            self.local_episode_counter += 1

    def run_session(self):
        """Train one session, i.e., one episode or t_max steps."""
        if self.done:
            self.state = np.array(self.env.reset(), dtype=np.float64)

        self.obtain_global_weights()

        states = np.zeros((self.tmax,) + self.state.shape, dtype=np.float64)
        actions = np.zeros(self.tmax, dtype=np.int32)
        rewards = np.zeros(self.tmax, dtype=np.float64)
        dones = np.zeros(self.tmax, dtype=np.int32)

        for i in range(self.tmax):
            # predict policy pi(a|s) and value V(s)
            action_probabilities = self.session.run(
                self.action_probs,
                feed_dict={self.state_ph: np.reshape(self.state, (1, -1))}
            )

            # select and perform action
            self.action = self.policy.select_action(action_probabilities)
            new_state, self.reward, self.done, _ = self.env.step(self.action)

            # save experience
            states[i] = copy.copy(self.state)
            actions[i] = self.action
            rewards[i] = self.reward
            dones[i] = self.done

            self.state = np.asarray(new_state, dtype=np.float64)

            with self.global_T.get_lock():
                self.global_T.value += 1

            # stop if episode ends
            if self.done:
                states = states[0:(i + 1)]
                actions = actions[0:(i + 1)]
                rewards = rewards[0:(i + 1)]
                dones = dones[0:(i + 1)]
                break

        # calculate the gradients and send them to the global network
        gradients = self.collect_gradients(states, actions, rewards, dones)
        self.send_gradients(gradients)

    def collect_gradients(self, states, actions, rewards, dones):
        """Collect gradients according to a series of experiences."""
        # feed the states in the model (again) to obtain a batch of predictions
        states = np.reshape(states, (-1,) + self.obs_shape)
        action_probas, state_values = self.session.run(
            [self.action_probs, self.value_pred],
            feed_dict={self.state_ph: states}
        )

        # calculate the n-step returns and advantages
        returns = self.calc_returns(rewards, state_values,
                                    gamma=self.gamma, n=self.td_steps)

        # reshape for feeding into tensorflow (time dimension is axis 0)
        returns = returns.reshape((-1, 1))
        advantages = np.reshape(returns - state_values, (-1, 1))
        actions = np.reshape(actions, (-1, 1))
        rewards = np.reshape(rewards, (-1, 1))

        # calculate the gradients for both actor and critic
        summary, grads_vars = self.session.run(
            [self.summary_op, self.grads_and_vars],
            feed_dict={
                self.state_ph: states,
                self.action_ph: actions,
                self.critic_target_ph: returns,
                self.actor_target_ph: advantages,
                self.rewards_ph: rewards
            }
        )

        # feed summary data to tensorboard
        self.summary_writer.add_summary(summary, self.local_episode_counter)

        # keep only the gradient and sum those of actor and critic to obtain a single
        gradients = [grads for grads, var in grads_vars]
        return gradients


class A3CAgent(BaseAgent):
    """Agent that implements the A3C Algorithm.

    Parameters
    ----------
    env_cls: an openai.gym environment class
        The environment to train on.
    policy: an implementation of spyro.policies.BasePolicy()
        The action-selection or exploration-exploitation policy. This must be an initialized
        object.
    name: str, optional, default="A3C_Global_Agent"
        The name of the agent, which is used to define its variable scope in the Tensorflow
        graph to avoid mix-ups with other tensorflow activities.
    beta: float, optional, default=0.05
        The contribution of entropy to the total loss of the actor. Specifically, the actor's
        gradient is calculated over :math:`log \\pi_{\\theta}(a_t | s_t)A(a_t, s_t) + \\beta H`
        where :math:`H` is the entropy.
    gamma: float, optional, default=0.9
        The discount factor for future rewards. At time :math:`t`, the reward :math:`r_{t + i}`
        is discounted by :math:`\\gamma^i` in the return calculation of :math:`t`, for all
        :math:`i = 1, ..., td_steps`.
    td_steps: int, optional default=10
        The number of steps to incorporate explicitly in the return calculation, i.e., the 'n'
        in the n-step Temporal Difference update.
    returns: str, one of ['mc_mean', 'n_step_mean', 'n_step_td']
        The return calculation to use. For 'mc_mean', uses Monte Carlo Discounted Mean Rewards,
        for 'n_step_mean' and 'n_step_td', calculates mean or sum of first n rewards respectively
        and then bootstraps the next state value.
    learning_rate: float, optional, default=1e-3
        The learning rate of the RMSProp optimizer.
    n_layers: int, optional, default=3
        The number of hidden layers to use in the neural network architecture.
    n_neurons: int, optional, default=512
        The number of neurons to use per hidden layer in the neural network.
    activation: str or tensorflow callable
        The activation function to use for hidden layers in the neural network.
    **env_params: key-value pairs
        Any parameters that should be used to initialize the environment.

    References
    ---------- 
    [Mnih et al. (2016)](https://arxiv.org/abs/1602.01783)
    """
    metadata = {
        "return_funcs": {
            "mc_mean": monte_carlo_discounted_mean_reward,
            "n_step_mean": n_step_discounted_mean_reward,
            "n_step_td": n_step_temporal_difference
        }
    }

    def __init__(self, policy, name="A3C_Global_Agent", beta=0.05, returns="mc_mean", td_steps=10,
                 n_layers=3, n_neurons=512, activation="relu", max_queue_size=10,
                 *args, **kwargs):

        self.name = name
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            print("multiprocessing method not (re)set to 'spawn', because context was already given.")

        # model parameters
        self.model_params = {
            "n_layers": n_layers,
            "n_neurons": n_neurons,
            "activation": activation
        }

        # algorithm-specific parameters (the rest is passed to BaseAgent)
        self.beta = beta
        assert returns in self.metadata["return_funcs"].keys(), ("'returns' must be one of {}. Got {}."
            .format(self.metadata["return_funcs"].keys(), returns))
        self.return_func = returns
        self.td_steps = td_steps

        # assure global does not lack too far behind by letting workers wait for queue slot
        self.max_queue_size = max_queue_size

        # process rest of input
        super().__init__(policy, log_prefix="A3C_run", *args, **kwargs)

    def _init_graph(self):
        """Initialize Tensorflow graph."""
        self.session = tf.Session()
        with tf.variable_scope(self.name):
            # build the model
            self.state_ph = tf.placeholder(tf.float64, shape=(None,) + self.obs_shape)
            self.action_probs, self.value_pred = build_actor_critic_mlp(
                self.state_ph,
                self.n_actions,
                scope_shared="shared",
                scope_actor="actor",
                scope_critic="critic",
                **self.model_params
            )

            # save lists of trainable variables for easy access
            self.var_list = tf.trainable_variables(scope=self.name)
            self.actor_var_list = tf.trainable_variables(scope=self.name + "/shared") + tf.trainable_variables(scope=self.name + "/actor")
            self.critic_var_list = tf.trainable_variables(scope=self.name + "/shared") + tf.trainable_variables(scope=self.name + "/critic")

            # define optimizer
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            # apply gradients placeholders and operations
            self.grads_ph = [tf.placeholder(tf.float64) for _ in range(len(self.var_list))]
            self.train_op = self.optimizer.apply_gradients(zip(self.grads_ph, self.var_list))
            self.session.run(tf.global_variables_initializer())

    def apply_gradients(self, gradients):
        """Apply the gradients to the global network."""
        if isinstance(gradients, list):
            if self.gradient_clip is not None:
                a_min, a_max = self.gradient_clip
                gradients = [np.clip(grad, a_min=a_min, a_max=a_max) for grad in gradients]
            self.session.run(
                self.train_op,
                feed_dict={ph: grad for ph, grad in zip(self.grads_ph, gradients)}
            )

        elif isinstance(gradients, dict):            
            if self.gradient_clip is not None:
                a_min, a_max = self.gradient_clip
                gradients["actor"]  = [np.clip(grad, a_min=a_min, a_max=a_max) for grad in gradients["actor"]]
                gradients["critic"]  = [np.clip(grad, a_min=a_min, a_max=a_max) for grad in gradients["critic"]]

            # apply gradients
            self.session.run(
                self.actor_train_op,
                feed_dict={ph: grad for ph, grad in zip(self.actor_grads_ph, gradients["actor"])}
            )
            self.session.run(
                self.critic_train_op,
                feed_dict={ph: grad for ph, grad in zip(self.critic_grads_ph, gradients["critic"])}
            )

        else:
            raise ValueError("Received neither a list nor a dictionary of lists as gradients.")

    def get_weights(self):
        return self.session.run(self.var_list)

    def fit(self, env_cls, env_params=None, tmax=32, total_steps=50000, n_workers=-1, restart=True, verbose=True):
        """Train the A3C Agent on the environment.

        Parameters
        ----------
        env_cls: uninitialized Python class or str
            The environment to train on. If a class is provided, it must be uninitialized,
            so that it can be initialized in each worker agent. This is because not all
            environment are picklable and can therefore not be send to different Python
            processes. If a string is provided, this string is fed to `gym.make()` to create
            the environment.
        env_params: dict, optional, default=None
            Dictionary of parameter values to pass to `env_cls` upon initialization.
        total_steps: int, optional, default=50,000
            The total number of training steps of all workers together.
        tmax: int, optional, default=32
            The maximum number of steps for a worker to run before calculating gradients
            and updating the global model if the episode is not over earlier.
        n_workers: int, optional, default=-1
            The number of worker processes to use. If set to -1, uses all but one of
            the available cores.
        restart: boolean, default=True
            Whether to (re-)initialize the network (True) or to keep the current neural
            network parameters (False).
        """
        if isinstance(env_params, dict):
            env_params = env_params
        elif env_params is None:
            env_params = {}
        else:
            raise ValueError("env_params must be either a dict or None, got {}"
                             .format(type(env_params)))

        self.tmax = tmax

        # obtain action and observation space information
        self.action_shape, self.n_actions, self.obs_shape, _  = \
                obtain_env_information(env_cls, env_params)

        # init graph, continue training, or use loaded graph
        if restart:
            self.total_steps = total_steps
            self._init_graph()
        else:
            try:
                self.total_steps += total_steps
            except AttributeError:
                # model was loaded, steps start from zero
                self.total_steps = total_steps

        # determine number of workers
        if n_workers == -1:
            n_workers = mp.cpu_count() - 1
            print("CPU count: {}".format(mp.cpu_count()))

        # create multiprocessing communication objects
        global_queue = mp.Queue(self.max_queue_size)
        global_T = mp.Value("i", 0)
        initial_weights = self.get_weights()
        weight_shapes = [w.shape for w in initial_weights]
        self.weight_shapes = weight_shapes
        print("Shapes of trainable weights: {}".format(weight_shapes))
        shared_weights = [mp.RawArray("d", w.reshape(-1)) for w in initial_weights]
        numpy_weights = [np.frombuffer(w, dtype=np.float64).reshape(weight_shapes[i])
                         for i, w in enumerate(shared_weights)]

        # create the worker agents
        agents = [
            A3CWorker(env_cls, global_T, global_queue, shared_weights, weight_shapes,
                      self.policy, tmax=self.tmax, total_steps=total_steps,
                      name="A3C_Worker_{}".format(i), beta=self.beta,
                      gamma=self.gamma, returns=self.return_func, td_steps=self.td_steps,
                      learning_rate=self.learning_rate, logdir=self.logdir,
                      env_params=env_params, **self.model_params)
            for i in range(n_workers)
        ]

        # start training
        for agent in agents:
            agent.start()

        print("Workers started.")
        try:
            # receive and apply gradients while running
            while True:
                try:
                    message = global_queue.get(block=True, timeout=3)
                except queue.Empty:
                    if global_T.value >= total_steps:
                        break
                else:
                    if isinstance(message, bytes):
                        # received gradients, apply them
                        gradients = pickle.loads(message)
                        self.apply_gradients(gradients)
                        new_weights = self.get_weights()
                        for i in range(len(new_weights)):
                            np.copyto(numpy_weights[i], new_weights[i])
                        print("\rGlobal steps: {}".format(global_T.value), end="")
                    elif isinstance(message, str):
                        print(message)
                    else:
                        print("Queue received unidentified message of type {}"
                              .format(type(message)))
        except KeyboardInterrupt:
            global_T = total_steps  # make workers think we are done

        for agent in agents:
            agent.join()

    def evaluate(self, env_cls, n_episodes=10000, tmax=None, policy=None, env_params=None, init=False):
        """Evaluate the agent on an environemt without training."""
        if policy is not None:
            self.eval_policy = policy
        else:
            self.eval_policy = self.policy

        if tmax is None:
            self.tmax = 1000000
        else:
            self.tmax = tmax

        self.env = make_env(env_cls, env_params)
        self.action_shape, self.n_actions = get_space_shape(self.env.action_space)
        self.obs_shape, _ = get_space_shape(self.env.observation_space)
        print("Environment initialized.")

        if init:
            tf.reset_default_graph()
            self._init_graph()
            print("Graph created.")

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
                action_probabilities = self.session.run(
                    self.action_probs,
                    feed_dict={self.state_ph: np.reshape(self.state, (1, -1))}
                )

                # select and perform action
                self.action = self.eval_policy.select_action(action_probabilities.reshape(-1))
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

            print("\rCompleted episode {}/{}".format(ep, n_episodes), end="")

        return self.eval_results

    def get_config(self):
        """Return configuration of the agent as a dictionary."""
        config = {
            "name": self.name,
            "policy": self.policy.get_config(),
            "beta": self.beta,
            "gamma": self.gamma,
            "td_lambda": self.td_steps,
            "return_func": self.return_func,
            "learning_rate": self.learning_rate,
            "n_layers": self.model_params["n_layers"],
            "n_neurons": self.model_params["n_neurons"],
            "activation": self.model_params["activation"],
            "logdir": self.logdir,
            "gradient_clip": self.gradient_clip,
            "max_queue_size": self.max_queue_size
        }
