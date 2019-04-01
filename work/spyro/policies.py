from __future__ import division
from abc import abstractmethod
from abc import ABCMeta
import numpy as np


class BasePolicy(object):
    """Abstract Base Class for action-selection policies. Not useful to instantiate
    on its own.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, scores):
        """Select an action based on probabilities or value estimates."""

    @abstractmethod
    def get_config(self):
        """Return the policy configuration as a dictionary."""


class SoftmaxPolicy(BasePolicy):
    """Policy that selects actions based on their probabilities.

    This policy is mostly used in Policy Gradient methods, such as A3C,
    where the network outputs a vector of probabilities for each action.
    The policy simply samples an action according to the given probabilities.
    """
    name = "SoftmaxPolicy"

    def __init__(self):
        super().__init__()

    def select_action(self, probabilities):
        """Select an action based on given probabilities.

        Parameters
        ----------
        probs: array of floats
            The probabilities of each action.

        Returns
        -------
        action: int
            The sampled action.

        Notes
        -----
        Does not use `np.random.choice`, since it does not work for larger arrays.
        Our implementation achieves exactly the same, but works on arrays of arbitrary sizes. 
        """
        return int(np.digitize(np.random.sample(), np.cumsum(probabilities)))

    def get_config(self):
        """The softmax policy does not have any parameters."""
        return {}


class GreedyPolicy(BasePolicy):
    """Policy that selects the action with the highest probability or value estimate.

    A policy that does not do any exploration, but selects the `best` action according
    to the input. This policy is mostly used in value gradient techniques, but also
    works with for policy gradient methods by providing the probabilities as input.
    """

    def select_action(self, scores):
        """Select the action with the highest score.

        Parameters
        ----------
        scores: array of floats
            The value-estimates or probabilities for each action.

        Returns
        -------
        action: int:
            The action with the highest score.
        """
        return np.argmax(scores)

    def get_config(self):
        """The Greedy policy does not have any parameters."""
        return {}


class EpsilonGreedyPolicy(BasePolicy):
    """Policy that selects the action with the highest probability or value estimate.

    A policy that can be used to balance exploration and exploitation. With probability
    :math:`\\epsilon` takes a random action, uniformly sampled over the possible actions.
    With probability :math:`1 - \\epsilon`, takes the action with the highest value estimate.

    This policy is classically used in value gradient techniques, but also
    works for policy gradient methods by providing the probabilities as input instead of
    value estimates.

    Parameters
    ----------
    epsilon: float
        The (starting) value of epsilon, i.e., the probability with which random actions
        are taken.
    decay_factor: float
        Factor with which to update epsilon. The update is given by
        :math:`\\epsilon = decay_factor * \\epsilon` and is made every `steps_between_decay`
        time steps.
    decay_step: float
        Absolute value to subtract from epsilon every `steps_between_decay` time steps. The
        update is given by :math:`\\epsilon = \\epsilon - decay_step`.
    steps_between_decay: int, optional, default=5000
        After how may time steps to perform the epsilon update steps defined by `decay_factor`
        or `decay_step`.
    """

    def __init__(self, epsilon, decay_factor=None, decay_step=None, steps_between_decay=5000):
        self.start_eps = start_eps
        self.eps = epsilon
        self.decay_factor = decay_factor
        self.decay_step = decay_step
        assert (decay_step is not None) + (decay_factor is not None) < 2, \
            ("Provide at most one of `decay_factor` and `abs_decay`.")

        self.steps_between_decay = steps_between_decay
        self.counter = 0

    def select_action(self, values):
        """Select the action with the highest score.

        Parameters
        ----------
        values: array of floats
            The value-estimates (or probabilities) for each action.

        Returns
        -------
        action: int:
            The action with the highest score.
        """
        self.counter += 1
        if self.counter % self.steps_between_decay == 0:
            if self.decay_factor is not None:
                self.eps *= self.decay_factor
            if self.decay_step is not None:
                self.eps -= self.decay_step

        if np.random.sample() < self.eps:
            return np.random.randint(0, len(values))
        else:
            return np.argmax(scores)

    def get_config(self):
        """The Greedy policy does not have any parameters."""
        return {"start_epsilon": self.start_eps,
                "epsilon": self.eps,
                "decay_factor": self.decay_factor,
                "decay_step": self.decay_step,
                "steps_between_decay": self.steps_between_decay,
                "counter": self.counter}

    def reset(self):
        """Reset epsilon to its start value."""
        self.eps = self.start_eps
        self.counter = 0
