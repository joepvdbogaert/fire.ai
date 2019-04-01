import os
import gym
import pickle
import numpy as np

from itertools import product
from pkg_resources import resource_filename

from fdsim.simulation import Simulator

from gym_firecommander.rewards import (
    binary_reward, response_time_penalty, linear_lateness_penalty,
    squared_lateness_penalty, tanh_reward, on_time_plus_minus_one
)


class FireCommanderEnv(gym.Env):
    metadata = {'render.modes': ['human'],
                'reward_functions': ['binary_reward', 'response_time_penalty', 'tanh_reward',
                                     'linear_lateness_penalty', 'squared_lateness_penalty',
                                     'plus_minus_one']}

    def __init__(self, reward_func='response_time_penalty', worst_response=25*60, action_type="num",
                 allow_nonempty=True):
        # load simulator
        self.sim = self._load_simulator()
        self.sim.initialize_without_simulating()
        # save some info
        self.station_names = np.sort([s.name for s in self.sim.stations.values()])
        self.num_stations = len(self.station_names)
        self.num_vehicles = len(self.sim.vehicles)
        # define action and observation spaces and reward function
        self.observation_space = gym.spaces.MultiDiscrete(np.ones(self.num_stations) * (self.num_vehicles + 1))
        self.action_type = action_type
        if action_type == "num":
            self.station_numbers = np.arange(len(self.station_names))
            self.action_num_to_tuple = [(None, None)] + [tup for tup in product(self.station_numbers, self.station_numbers) if tup[0] != tup[1]]
            self.action_space = gym.spaces.Discrete(len(self.action_num_to_tuple))
        elif action_type == "tuple":
            self.action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(self.num_stations + 1), gym.spaces.Discrete(self.num_stations)))
        self._assign_reward_function(reward_func)
        # define worst possible response time
        self.worst_response = worst_response
        # whether to consider relocation to a non-empty station as invalid
        self.allow_nonempty = allow_nonempty
        # for rendering actions and states on top of each other
        np.set_printoptions(sign=" ")

    def step(self, action):
        """Take one step in the environment and give back the results.

        Parameters
        ----------
        action: tuple(int, int)
            The action to take as combination of (origin, destination).

        Returns
        -------
        new_observation, reward, is_done, info: tuple
            new_observation: a gym.space
                The next state after taking a step.
            reward: float
                The immediate reward for the current step.
            is_done: boolean
                True if the episode is finished, False otherwise.
            info: dict
                Possible additional info about what happened.
        """
        # check if suggested action is valid
        valid = self._take_action(action)
        if not valid:
            response = self.worst_response
            target = 6*60
        else:
            # simulate until a TS response is needed
            response = np.inf
            while response == np.inf:
                response, target = self._simulate()

        self.last_action = action if self.action_type == "tuple" else self.action_num_to_tuple[action]
        # calculate reward and new state
        self.reward = self._get_reward(response, target, valid=valid)
        self.state, self.is_done = self._extract_state()
        return self.state, self.reward, self.is_done, {"note": "nothing to report"}

    def reset(self):
        """Reset by starting a new episode. This is done by simulating until the
        episode start-condition is met."""
        while not self._check_episode_start_condition():
            self._simulate()
        self.state, _ = self._extract_state()
        return self.state

    def render(self, mode='human'):
        """Render the current state and the last action."""
        a = np.zeros(self.num_stations, dtype=np.float32)
        # a = [0] * self.num_stations
        if self.last_action[0] is not None:
            a[self.last_action[0]] = int(-1)
            a[self.last_action[1]] = int(1)
        print("------------------------------------------------------------")
        print("Action: {}".format(a))
        print("Reward: {}".format(self.reward))
        print("State:  {}".format(np.asarray(self.state, dtype=np.float32)))

    def close(self):
        """Close the environment."""
        pass

    def _take_action(self, action):
        assert self.action_space.contains(action), "Action not in action space."
        if self.action_type == "num":
            # action=0 means do not relocate
            if action == 0:
                return True
                self.last_action = (None, None)
            else:
                action = self.action_num_to_tuple[action]
                self.last_action = action
        if self.action_type == "tuple":
            # origin > num_stations means do not relocate
            if action[0] == self.num_stations:
                return True

        if self.state[action[0]] == 0:
            # invalid relocation: no vehicles available
            return False
        elif (not self.allow_nonempty) and (self.state[action[1]] > 0):
            return False
        else:
            # valid relocation, relocate
            self.sim.relocate_vehicle("TS", self.station_names[action[0]], self.station_names[action[1]])
            return True

    def _load_simulator(self):
        """Load Simulator resource in a zipped-egg safe way."""
        path = "/".join(["..", "backends", "firecommander_aa_17_ts_v0.fdsim"])
        with open(resource_filename(__name__, path), 'rb') as fd:
            sim = pickle.load(fd)
        sim.rsampler._create_response_time_generators()
        sim.isampler.reset_time()
        sim.set_max_target(sim.max_target)
        # filter to only TS vehicles
        rs = sim.resource_allocation.copy()
        rs[["RV", "HV", "WO"]] = 0
        sim.set_resource_allocation(rs)
        return sim

    def _extract_state(self):
        return self._get_available_vehicles(), self._check_episode_end_condition()

    def _get_available_vehicles(self):
        stations, counts = np.unique([v.current_station_name for v in self.sim.vehicles.values()
                                      if v.available and v.type=="TS"], return_counts=True)
        vehicles = np.zeros(len(self.station_names), dtype=np.int16)
        vehicles[np.in1d(self.station_names, stations)] = counts
        return vehicles

    def _check_episode_start_condition(self):
        """Check whether the simulator is at a situation where a train episode can start."""
        vehicles = self._get_available_vehicles()
        if np.sum(vehicles == 0) >= 2:
            return True
        else:
            return False

    def _check_episode_end_condition(self):
        """Check whether the simulator is at a situation where a train episode ends."""
        vehicles = self._get_available_vehicles()
        if np.sum(vehicles == 0) < 2:
            return True
        else:
            return False

    def _assign_reward_function(self, reward_func):
        """Set the reward function as a method of self.

        Parameters
        ----------
        reward_func: str, default='response_time_penalty'
            The reward function to use. One of ['binary_reward', 'response_time_penalty',
            'tanh_reward','linear_lateness_penalty', 'squared_lateness_penalty'].
        """
        if reward_func == "binary_reward":
            self._get_reward = binary_reward
        elif reward_func == "response_time_penalty":
            self._get_reward = response_time_penalty
        elif reward_func == "linear_lateness_penalty":
            self._get_reward = linear_lateness_penalty
        elif reward_func == "squared_lateness_penalty":
            self._get_reward = squared_lateness_penalty
        elif reward_func == "tanh_reward":
            self._get_reward = tanh_reward
        elif reward_func == "plus_minus_one":
            self._get_reward = on_time_plus_minus_one
        else:
            raise ValueError("'reward' must be one of {}. Received {}".format(
                             self.metadata['reward_functions'], reward))

    def _simulate(self):
        """ Simulate a single incident and all corresponding deployments"""
        # sample incident and update status of vehicles at new time t
        self.sim.t, time, type_, loc, prio, req_vehicles, func, dest = self.sim._sample_incident()

        self.sim._update_vehicles(self.sim.t, time)

        # sample dispatch time
        dispatch = self.sim.rsampler.sample_dispatch_time(type_)

        # keep track of minimum TS response time
        min_ts_response = np.inf

        # get target response time
        target = self.sim._get_target(type_, func, prio)

        # sample rest of the response time for TS vehicles
        for v in req_vehicles:
            if v == "TS":

                vehicle, estimated_time = self.sim._pick_vehicle(loc, v)
                if vehicle is None:
                    turnout, travel, onscene, response = [np.nan]*4
                else:
                    vehicle.assign_crew() # always full time in this case

                    turnout, travel, onscene = self.sim.rsampler.sample_response_time(
                        type_, loc, vehicle.current_station_name, vehicle.type, vehicle.current_crew,
                        prio, estimated_time=estimated_time)
                    
                    vehicle.dispatch(dest, self.sim.t + (onscene/60) + (estimated_time/60))

                response = dispatch + turnout + travel

                # we must return a numerical value
                if np.isnan(response):
                    response = self.worst_response

                if response < min_ts_response:
                    min_ts_response = response

        return min_ts_response, target