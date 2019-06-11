import copy
import numpy as np
import gym

from gym_firecommander.envs import FireCommanderBigEnv


class FireCommanderV2(FireCommanderBigEnv):

    def __init__(self, reward_func='scaled_spare_time', worst_response=2100):
        super().__init__(reward_func=reward_func, worst_response=worst_response)
        # adjust action space: we only choose the origin station
        # the state space gets an additional array to indicate the destination
        self.action_space = gym.spaces.Discrete(self.num_stations + 1)
        self.observation_space = gym.spaces.MultiDiscrete(
            [2]*self.num_stations + [1]*self.num_stations + [7, 24]
        )

        self.destination_candidates = []
        self.current_dest = None

    def step(self, action):
        """Take one step in the environment and give back the results.

        Parameters
        ----------
        action: int in [0, num_stations + 1]
            The station from which a vehicle should be relocated (origin).
            Action = 0, means no relocation to the given destination.

        Returns
        -------
        new_observation, reward, is_done, info: tuple
            new_observation: np.array
                The next state after taking a step.
            reward: float
                The immediate reward for the current step.
            is_done: boolean
                True if the episode is finished, False otherwise.
            info: dict
                Possible additional info about what happened.
        """
        valid = self._take_action(action)
        self.last_action = (None, None) if action == self.num_stations else \
                (action, self.current_dest)

        # go to next incident if all destinations have been decided on
        self.reward = 0.0
        while len(self.destination_candidates) == 0:

            # simulate until TS response needed
            response = np.inf
            while response == np.inf:
                response, target = self._simulate()

            # terminal state is reached
            if response is None:
                vehicles = self._get_available_vehicles()
                self.state, self.is_done = self._extract_state(vehicles)
                return self.state, self.reward, self.is_done, {"note": "terminal state reached"}

            # process NaN targets
            if np.isnan(target):
                target = response

            # calculate reward and empty stations
            self.reward += self._get_reward(response, target, valid=valid)
            self.destination_candidates, vehicles_per_station = self._get_empty_stations()
        else:
            vehicles_per_station = self._get_available_vehicles()

        # obtain the state with the next destination
        self.current_dest = self.destination_candidates.pop(0)
        self.state, self.is_done = self._extract_state(vehicles_per_station)


        # adjust reward if invalid action was taken
        if not valid:
            response = self.worst_response
            target = 6 * 60
            self.reward = self._get_reward(response, target, valid=valid)

        return self.state, self.reward, self.is_done, \
                {"last_action": self.last_action, "valid": valid}

    def reset(self, forced_vehicles=None):
        """Reset the environment to start a new episode."""
        self.sim.fast_simulate_big_incident(forced_num_ts=forced_vehicles)
        self.t_episode_end = self.sim.major_incident_info["duration"]
        self.time = self.sim.major_incident_info["time"]
        self.destination_candidates, vehicles_per_station = self._get_empty_stations()
        self.current_dest = self.destination_candidates.pop(0)
        self.state, _ = self._extract_state(vehicles_per_station)
        return self.state

    def _take_action(self, action):
        """Process a given action.

        Assumes that self.current_dest is (still) related to the provided action.

        Parameters
        ----------
        action: int
            The action to take. In this case, the action is the number of the station
            from which a vehicle is relocated, where N (the number of stations) means
            no relocation and 0 to N-1 refer to the available stations.

        Returns
        -------
        valid: bool
            Whether the provided action was a valid action given the current state.
        """
        if action == self.num_stations:  # no relocation
            return True
        elif self.state[action] == 0:  # invalid: no vehicle available at origin
            return False
        else:  # valid, relocate
            self.sim.relocate_vehicle("TS", self.station_names[action],
                                      self.station_names[self.current_dest])
            return True

    def _extract_state(self, vehicles_per_station):
        """Compose the current observation from its elements.

        Parameters
        ----------
        vehicles_per_station: array-like
            The number of available vehicles at each station, in the same order as
            self.station_names.

        Returns
        -------
        state: np.array
            The current state observation.
        is_done: bool
            Whether the new state is a terminal state or not.
        """
        dest_indicators = np.zeros(self.num_stations)
        dest_indicators[self.current_dest] = 1
        state = np.concatenate([
            vehicles_per_station,
            dest_indicators,
            np.array([self.time.weekday(), self.time.hour])
        ])
        return state, self._check_episode_end_condition()

    def _get_empty_stations(self):
        """Find the station numbers that are empty, i.e., have no vehicles available.

        Assumes self.state is up-to-date, so self._extract_state() must be called beforehand.

        Returns
        -------
        empty_stations: list of ints
            The numbers / indexes corresponding to empty stations.
        vehicles_per_station: np.array
            The number of vehicles at each station. In the same order as self.station_names.
        """
        vehicles = self._get_available_vehicles()
        return np.flatnonzero(vehicles == 0).tolist(), vehicles

    def _simulate(self):
        """Simulate a single incident and all corresponding deployments."""
        # sample incident and update status of vehicles at new time t
        self.sim.t, self.time, type_, loc, prio, req_vehicles, func, dest = self.sim._sample_incident()
        self.sim._fast_update_vehicles(self.sim.t, self.time)

        if self._check_episode_end_condition():
            return None, None

        # sample dispatch time
        dispatch = self.sim.rsampler.sample_dispatch_time(type_)

        # keep track of minimum TS response time
        min_ts_response = np.inf

        # get target response time
        target = self.sim._get_target(type_, func, prio)

        # sample rest of the response time for TS vehicles
        for v in req_vehicles:
            if v == "TS":

                vehicle, estimated_time = self.sim._fast_pick_vehicle(loc, v)
                if vehicle is None:
                    turnout, travel, onscene, response = [np.nan]*4

                else:
                    turnout = next(self.sim.rsampler.turnout_generators["fulltime"][prio][vehicle.type])
                    travel = self.sim.rsampler.sample_travel_time(estimated_time, vehicle.type)
                    onscene = next(self.sim.rsampler.onscene_generators[type_][vehicle.type])
                    response = dispatch + turnout + travel
                    vehicle.dispatch(dest, self.sim.t + (response + onscene + estimated_time) / 60)

                # we must return a numerical value
                if np.isnan(response):
                    response = self.worst_response

                if response < min_ts_response:
                    min_ts_response = response

        return min_ts_response, target

    def get_snapshot(self):
        """Retrieve data from the environment in order to return to its current state
        later. Does not include random states, so simulated output will change upon return.

        Returns
        -------
        data: dict
            Various parameters describing the current state of the system. Can be used as
            input for `FireCommanderEnv.set_snapshot()` to return to this state.
        """
        data = {
            "t": self.sim.t,
            "time": self.time,
            "t_episode_end": self.t_episode_end,
            "current_dest": self.current_dest,
            "destination_candidates": self.destination_candidates,
            "vehicles": self.sim.vehicles,
            "stations": self.sim.stations,
            "state": self.state,
            "done": self.is_done}
        return copy.deepcopy(data)

    def set_snapshot(self, data):
        """Set the environment back to an earlier retrieved snapshot.

        Parameters
        ----------
        data: dict
            Output of `FireCommanderEnv.get_snapshot()`.
        """
        self.sim.t = data["t"]
        self.time = data["time"]
        self.t_episode_end = data["t_episode_end"]
        self.current_dest = data["current_dest"]
        self.destination_candidates = data["destination_candidates"]
        self.sim.vehicles = data["vehicles"]
        self.sim.stations = data["stations"]
        self.state = data["state"]
        self.is_done = data["done"]
        self.sim._add_base_stations_to_vehicles()
        self.sim.set_max_target(self.sim.max_target)
