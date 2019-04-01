from abc import ABCMeta
from abc import abstractmethod

from spyro.policies import GreedyPolicy, EpsilonGreedyPolicy, SoftmaxPolicy


class BaseAgent(object):

    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass

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
