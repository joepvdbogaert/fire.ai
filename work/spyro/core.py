import os
from abc import ABCMeta
from abc import abstractmethod
import json

import tensorflow as tf

from spyro.policies import GreedyPolicy, EpsilonGreedyPolicy, SoftmaxPolicy
from spyro.utils import find_free_numbered_subdir, obtain_env_information


class BaseAgent(object):

    __metaclass__ = ABCMeta

    tf_optimizers = {'rmsprop': tf.train.RMSPropOptimizer,
                     'adam': tf.train.AdamOptimizer,
                     'sgd': tf.train.GradientDescentOptimizer,
                     'adagrad': tf.train.AdagradOptimizer}

    def __init__(self, policy, learning_rate=1e-3, gradient_clip=None,
                 gamma=0.9, clear_logs=False, logdir="log", log=True, log_prefix="run"):
        # logging
        if clear_logs:
            raise NotImplementedError("Automatically clearing log files is not implemented yet.")

        self.logdir = find_free_numbered_subdir(logdir, prefix=log_prefix)
        self.log = log
        if log:
            os.mkdir(self.logdir)
            print("Logging in directory: {}".format(self.logdir))

        # policies
        self.policy = policy

        # hyperparameters
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.gamma = gamma

    @abstractmethod
    def fit(self):
        """Train the agent on an environment."""

    @abstractmethod
    def test(self, env_cls, env_params=None, policy=None):
        """Test the agent, possibly using a different policy than in training."""

    def save_weights(self):
        """Save the trainable variables to continue training later."""
        self.saver = tf.train.Saver()
        path = self.saver.save(self.session, os.path.join(self.logdir, "model.ckpt"))

        # also save agent configuration
        with open(os.path.join(self.logdir, "agent_config.json"), "w") as f:
            json.dump(self.get_config(), f, sort_keys=True, indent=4)

        print("Model and agent settings saved in {}.".format(path))

    def load_weights(self, path, env_cls, env_params=None):
        """Load previously fitted weights."""
        self.action_shape, self.n_actions, self.obs_shape, _ = \
                    obtain_env_information(env_cls, env_params)
        tf.reset_default_graph()
        self._init_graph()
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, path)
        print("Model restored from {}.".format(path))

    @abstractmethod
    def get_config(self):
        """Get agent hyperparameter settings as a Python dictionary."""
