import os
import gym
import numpy as np


def find_free_numbered_subdir(dir, prefix="run"):
    """Find a numbered subdirectory that does not exist yet. Useful for logging new runs."""
    i = 0
    exists = True
    while exists:
        i += 1
        exists = os.path.exists(os.path.join(dir, prefix + str(i)))
    return os.path.join(dir, prefix + str(i))


def make_env(env_cls, env_params):
    """Makes a training environment from either a string representing a gym-registered env or
    a (custom) callable with possible parameters.
    """
    if isinstance(env_cls, str):
        try:
            env = gym.make(env_cls)
        except:
            raise ValueError("{} is not a valid gym environment.".format(env_cls))
    else:  # it should be a callable
        env = env_cls(**env_params)
    return env


def get_space_shape(space):
    """Retrieves the shape and dimension of a gym.spaces.Space.

    Parameters
    ----------
    space: gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.Tuple
        The space to analyze.

    Returns
    -------
    shape: tuple or list of tuples
        The shape of the space if it is a single (simple) space or a list of
        shapes if the space has subspaces.
    n: int or array-like of ints
        The dimension of a Discrete or MultiDiscrete space. This is especially useful
        when it concerns an action space, in which case it defines how many actions
        are possible.

    Notes
    -----
    Currently only supports Tuple, Discrete, and MultiDiscrete spaces.
    """
    if isinstance(space, gym.spaces.Tuple):
        # get shapes of subspaces
        shapes, ns = [], []
        for subspace in space.spaces:
            sh, n = get_space_shape(subspace)
            shapes.append(sh)
            ns.append(n)

        return shapes, ns

    elif isinstance(space, gym.spaces.Discrete):
        return space.shape, space.n

    elif isinstance (space, gym.spaces.MultiDiscrete):
        return space.shape, space.nvec

    elif isinstance(space, gym.spaces.Box):
        return space.shape, np.inf

    else:
        raise ValueError("Spyro currently only supports Discrete, MultiDiscrete, "
                         "Box, and Tuple spaces. Got {}".format(type(space)))


def obtain_env_information(env_cls, env_params):
    """Obtain necessary information about the environment's state and action spaces.

    Parameters
    ----------
    env_cls: unitialized class or environment name from gym register
        The environment to extract information from.
    env_params: dict
        Parameters and values to pass to env_cls if it is a class.

    Returns
    -------
    action_shape: tuple
        The shape of the action space.
    action_n: int
        The dimension of the action space.
    obs_shape: tuple
        The shape of the observations space.
    obs_n: int
        The dimension of the observation space.
    """
    env = make_env(env_cls, env_params)
    observation_space = env.observation_space
    action_space = env.action_space
    del env

    action_shape, action_n = get_space_shape(action_space)
    obs_shape, obs_n = get_space_shape(observation_space)

    return action_shape, action_n, obs_shape, obs_n
