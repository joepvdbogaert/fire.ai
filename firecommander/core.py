from abc import ABCMeta, abstractmethod
from copy import copy
import numpy as np

from firecommander.actions import ActionSpace
from firecommander.rewards import (
    binary_reward,
    response_time_penalty,
    linear_lateness_penalty,
    squared_lateness_penalty,
    tanh_reward
)


class BaseEnv(object):
    """ A base class for all training environments at Fire Deparment
    Amsterdam-Amstelland.

    Parameters
    ----------
    path_to_simulator: str
        The path to a saved (pickled) fdsim.Simulator object that will be used
        as the environment.
    reward: str
        The reward function to use. One of ['binary_reward', 'response_time_penalty',
        'linear_lateness_penalty', 'squared_lateness_penalty', 'tanh_reward']. See
        the `actions` module for explanations of the corresponding functions.
    vehicle_types: array-like of strings
        The types of vehicles to propose relocations for. A subset of
        ['TS', 'RV', 'HV', 'WO'].
    """
    __metaclass__ = ABCMeta
    metadata = {'reward_functions': ['binary_reward', 'response_time_penalty',
                'linear_lateness_penalty', 'squared_lateness_penalty', 'tanh_reward']}

    def __init__(self, path_to_simulator, reward, vehicle_types):
        # load the simulation engine
        self.sim = quick_load_simulator(path_to_simulator)
        self.sim.initialize_without_simulating()
        # store attributes
        self.station_names = np.sort(self.sim.stations["kazerne"].values)
        self.vehicle_types = vehicle_types
        # create reward function and action space
        self._assign_reward_function(reward)
        self.action_space = ActionSpace(self.station_names, self.vehicle_types)

    def _assign_reward_function(self, reward):
        """ Set the reward function as a method of self. """
        if reward == "binary_reward":
            self._get_reward = binary_reward
        elif reward == "response_time_penalty":
            self._get_reward = response_time_penalty
        elif reward == "linear_lateness_penalty":
            self._get_reward = linear_lateness_penalty
        elif reward == "squared_lateness_penalty":
            self._get_reward = squared_lateness_penalty
        elif reward == "tanh_reward":
            self._get_reward = tanh_reward
        else:
            raise ValueError("'reward' must be one of {}. Received {}".format(
                             self.metadata['reward_functions'], reward))

    def _take_action(self, action):
        vehicle_type, origin, destination = self.action_space.get_action_meaning(action)
        if vehicle_type != "NONE":
            self.sim.relocate_vehicle(vehicle_type, origin, destination)

    def step(self, action):
        """ Take the action and return the next state and reward. """
        self._take_action(action)
        response_time, target = self._simulate()
        reward = self._get_reward(response_time, target)
        new_state = self.extract_state()

        assert new_state.shape == self.state_template.shape
        return new_state, reward

    def _simulate(self):
        """ Simulate a single incident and all corresponding deployments"""
        # sample incident and update status of vehicles at new time t
        self.sim.t, time, type_, loc, prio, req_vehicles, func, dest = \
            self.sim._sample_incident(self.sim.t)

        self.sim._update_vehicles(self.sim.t)

        # sample dispatch time
        dispatch = self.sim.rsampler.sample_dispatch_time(type_)

        # keep track of minimum TS response time
        min_ts_response = np.inf

        # sample rest of the response time (and log everything)
        for v in req_vehicles:

            vehicle, estimated_time = self.sim._pick_vehicle(loc, v)
            if vehicle is None:
                turnout, travel, onscene, response = [np.nan]*4
            else:
                turnout, travel, onscene = self.sim.rsampler.sample_response_time(
                    type_, loc, vehicle.current_station, vehicle.type,
                    estimated_time=estimated_time)

                vehicle.dispatch(dest, self.sim.t + (onscene/60) + (estimated_time/60))

                response = dispatch + turnout + travel

                if response < min_ts_response and v == "TS":
                    min_ts_response = response

        target = 10*60
        return response, target

    @abstractmethod
    def extract_state(self):
        """ Get the current state. This will differ for various state representations
        and is therefore implemented in subclasses. """
        pass

    def get_legal_actions(self):
        return self.action_space.get_legal_actions(self.sim.vehicles)


class FireCommanderLevel1(BaseEnv):
    """ A simple version of FireCommander with only the number of pumper vehicles per
    station as state.

    Parameters
    ----------
    path_to_simulator, reward, vehicles_types: parameters of BaseEnv,
        See BaseEnv for description.
    """

    def __init__(self, path_to_simulator, reward, vehicle_types):
        super(FireCommanderLevel1, self).__init__(path_to_simulator, reward, vehicle_types)

        self.state_template = pd.Series(np.zeros(len(self.station_names)),
                                        index=self.station_names)

        self.state_shape = len(self.station_names)

    def extract_state(self):
        """ Get the current state (a vector where every entry represents the number of pumper
        vehicles at a specific station). """
        stations, counts = np.unique([v.current_station for v in self.sim.vehicles.values()
                                      if v.type == "TS"], return_counts=True)

        state = np.zeros(len(self.station_names), dtype=np.int8)
        state[np.in1d(self.station_names, stations)] = counts
        return state
