from abc import ABCMeta
from abc import abstractmethod

from spyro.policies import GreedyPolicy, EpsilonGreedyPolicy, SoftmaxPolicy
from spyro.utils import find_free_numbered_subdir


class BaseAgent(object):

    __metaclass__ = ABCMeta

    def __init__(self, policy, plan_policy=None, learning_rate=1e-3, gradient_clip=None,
                 gamma=0.9, clear_logs=False, logdir="log", log=True):
        # logging
        if clear_logs:
            raise NotImplementedError("Automatically clearing log files is not implemented yet.")

        self.logdir = find_free_numbered_subdir(logdir, prefix="a3c_run")
        self.log = log
        if log:
            print("Logging in directory: {}".format(self.logdir))

        # policies
        self.policy = policy
        self.plan_policy = plan_policy

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
