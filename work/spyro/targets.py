from __future__ import division
import numpy as np
# from numba import jit


def n_step_advantage(rewards, state_values, gamma=0.90, n=10):
    """Compute the n-step forward view advantages of a series of experiences.

    This function calculates the n-step forward view temporal difference and subtracts the
    estimate of the state value as a baseline to obtain the advantage. See
    :code:`n_step_temporal_difference` for details on the n-step return.

    Parameters
    ----------
    rewards: array of floats
        The consecutively obtained scalar rewards.
    state_values: array of floats
        The estimates of the state values :math:`V(s_t)`. Note that these values should
        represent the states just before obtaining the corresponding reward, so
        `rewards[i]` was obtained by taking an action in the state with state value
        `state_values[i]`. This function cannot check or enforce this, so take care
        when providing input for this function.
    gamma: float, optional, default=0.9
        The discount factor for future rewards.
    n: int, optional, default=10
        The number of future rewards to incorporate explicitly in the return in a
        discounted fashion.

    Returns
    -------
    advantages: array of floats
        The advantages of every experience tuple corresponding to the rewards and
        state values.
    """
    returns = n_step_forward_view_return(rewards, state_values, gamma=gamma, n=n)
    return returns - np.asarray(state_values)


def n_step_temporal_difference(rewards, state_values, gamma=0.9, n=10):
    """Compute the n-step forward view temporal difference of a series of experiences.

    The n-step return at time step :math:`t` is the sum of discounted rewards of steps
    :math:`t:t + n`, plus the estimated value of the resulting state :math:`V(s_{t + n})`.

    The following equation describes this formally:
    :math:``` 
        R_t = r_t + \\gamma r_{r+1} + \\gamma^2 r_{r+2} + ... + \\gamma^{n - 1}r_{t + n - 1}
              + \\gamma^n V_t(S_{t + n})
    ```

    Parameters
    ----------
    rewards: array of floats
        The consecutively obtained scalar rewards.
    state_values: array of floats
        The estimates of the state values :math:`V(s_t)`. Note that these values should
        represent the states just before obtaining the corresponding reward, so
        `rewards[i]` was obtained by taking an action in the state with state value
        `state_values[i]`. This function cannot check or enforce this, so take care
        when providing input for this function.
    gamma: float, optional, default=0.9
        The discount factor for future rewards.
    n: int, optional, default=10
        The number of future rewards to incorporate explicitly in the return in a
        discounted fashion.

    Returns
    -------
    advantages: array of floats
        The n-step forward view return of every experience tuple corresponding to the
        provided rewards and state values.
    """
    assert len(rewards) == len(state_values), ("rewards and state_values must"
        " have the same length")

    t = len(rewards)
    returns = np.zeros(t)
    gamma_array = gamma ** np.arange(n + 1)
    rewards = np.append(rewards, np.zeros(n))
    values = np.append(state_values, np.zeros(n))
    returns = np.array([np.dot(gamma_array, np.append(rewards[i:(i + n)], values[i + n + 1])) for i in range(t)])
    return returns


# @jit(nopython=True)
def monte_carlo_discounted_mean_reward(rewards, *args, gamma=0.9, **kwargs):
    """Calculates the mean discounted reward, instead of summed.

    .. math::

        G_{t:t+n} = \\frac{\\sum_{i=0}^{n-1} \\gamma^i r_{t+i} + \\gamma^n V(s')}{\\sum_{i=0}^n \\gamma^i}

    Parameters
    ----------
    rewards: array of floats
        The consecutively obtained scalar rewards.
    gamma: float, optional, default=0.9
        The discount factor for future rewards.
    *args, **kwargs: any
        To make function input compatible with other target functions. Parameters
        are not used.

    Returns
    -------
    returns: array of floats
        The discounted mean reward of every experience tuple corresponding
        to the provided rewards..
    """
    T = len(rewards)
    gamma_array = gamma ** np.arange(T)
    return np.array([np.dot(rewards[i:], gamma_array[:(T - i)]) / np.sum(gamma_array[:(T - i)])
                     for i in range(T)])


# @jit(nopython=True)
def n_step_discounted_mean_reward(rewards, state_values, gamma=0.9, n=3):
    """Calculates the mean discounted reward, instead of summed.

    .. math::

        G_{t:t+n} = \\frac{\\sum_{i=0}^{n-1} \\gamma^i r_{t+i} + \\gamma^n V(s')}{\\sum_{i=0}^n \\gamma^i}

    Parameters
    ----------
    rewards: array of floats
        The consecutively obtained scalar rewards.
    state_values: array of floats
        The estimates of the state values :math:`V(s_t)`. Note that these values should
        represent the states just before obtaining the corresponding reward, so
        `rewards[i]` was obtained by taking an action in the state with state value
        `state_values[i]`. This function cannot check or enforce this, so take care
        when providing input for this function.
    gamma: float, optional, default=0.9
        The discount factor for future rewards.
    n: int, optional, default=10
        The number of future rewards to incorporate explicitly in the return in a
        discounted fashion.

    Returns
    -------
    returns: array of floats
        The n-step forward view discounted mean reward of every experience tuple corresponding
        to the provided rewards and state values.
    """
    T = len(rewards)
    gamma_array = gamma ** np.arange(n + 1)
    state_values = list(state_values)
    state_values.append(0)
    returns = np.zeros(T)
    for i in range(T):
        steps = min(n, T - i)
        returns[i] = (np.dot(rewards[i:(i + steps)], gamma_array[:steps]) + 
                      gamma_array[steps] * state_values[i + steps]) / np.sum(gamma_array[:(steps + 1)])

    return returns
