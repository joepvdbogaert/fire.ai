"""The rewards module specifies different reward signals to be used in training
Reinforcement Learning agents on the FireCommander environments.

Every function takes the response_time and the target response time as inputs, to
ensure compatibility with every ennvironment, even if it doesn't use both of them.
"""
import numpy as np


def binary_reward(response_time, target, valid=True):
    return 1 if response_time <= target else 0

def response_time_penalty(response_time, target, valid=True):
    return -response_time

def linear_lateness_penalty(response_time, target, valid=True):
    return np.minimum(target - response_time, 0)

def squared_lateness_penalty(response_time, target, valid=True):
    return -(np.minimum(target - response_time, 0)**2)

def tanh_reward(response_time, target, valid=True):
    return np.tanh(target - response_time)

def on_time_plus_minus_one(response_time, target, valid=True, invalid_reward=-5):
    if not valid:
        return invalid_reward
    return 1 if response_time <= target else -1