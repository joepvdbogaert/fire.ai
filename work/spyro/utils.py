import os
import gym


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
