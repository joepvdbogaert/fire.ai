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


class FixedActionPolicy(BasePolicy):
    """Policy that always takes the same action. Useful for planning or obtaining baseline
    value estimates for states.

    Parameters
    ----------
    action: any
        The action to return on every call to `self.select_action()`.
    """
    name = "FixedActionPolicy"

    def __init__(self, action):
        self.action = action

    def select_action(self, *args, **kwargs):
        """Return the fixed action.

        Returns
        -------
        action: any
            The action that was specified upon initialization.
        """
        return self.action

    def get_config(self):
        return {"action": self.action}


class RandomPolicy(BasePolicy):
    """Policy that takes random uniform actions no matter its input."""
    name = "RandomPolicy"

    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def select_action(self, *args, **kwargs):
        """Randomly select one of n_actions."""
        return np.random.randint(0, len(values))

    def get_config(self):
        return {"n_actions": self.n_actions}


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

    def __init__(self, start_epsilon, final_epsilon=0.0, steps_to_final_eps=1e6, decay=None):
        self.start_eps = start_epsilon
        self.eps = start_epsilon
        self.final_eps = final_epsilon
        self.decay = decay
        self.steps_to_final_eps = steps_to_final_eps

        if decay is None:
            self.decay_step = 0.0

        elif decay == "linear":
            assert start_epsilon > final_epsilon, "Start value of epsilon must be higher than final"
            self.decay_step = (start_epsilon - final_epsilon) / steps_to_final_eps

        else:
            raise ValueError("decay must be one of [None, 'linear']")

    def select_action(self, values):
        """Select the action with the highest score with probability :math:`(1 - \\epsilon)`
        and a random action with probability :math:`\\epsilon`.

        Parameters
        ----------
        values: array of floats
            The value-estimates (or probabilities) for each action.

        Returns
        -------
        action: int:
            The action with the highest score.
        """
        if self.eps > self.final_eps:
            self.eps -= self.decay_step

        if np.random.sample() < self.eps:
            return np.random.randint(0, len(values))
        else:
            return np.argmax(values)

    def get_config(self):
        """The Greedy policy does not have any parameters."""
        return {"start_epsilon": self.start_eps,
                "epsilon": self.eps,
                "final_epsilon": self.final_eps,
                "decay": self.decay,
                "steps_to_final_eps": self.steps_to_final_eps,
                "counter": self.counter}

    def reset(self):
        """Reset epsilon to its start value."""
        self.eps = self.start_eps
        self.counter = 0
