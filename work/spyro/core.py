from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from spyro.policies import GreedyPolicy, EpsilonGreedyPolicy, SoftmaxPolicy
from spyro.utils import find_free_numbered_subdir


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
    def test(self, policy=None):
        """Test the agent, possibly using a different policy than in training."""

    def save_weights(self, filepath, scope=None):
        """Save the trainable variables to continue training later."""
        pass

    def load_weights(self, filepath):
        """Load previously fitted weights."""
