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
        try:
            self.current_dest = self.destination_candidates.pop(0)
            self.state, self.is_done = self._extract_state(vehicles_per_station)
        except IndexError:
            self.render()
            print("t: {}".format(self.sim.t))
            print("sim log:")
            print(self.sim.log[0:10])
            print("-----------")
            print("availabel vehicles:")
            print(self._get_available_vehicles())
            print(self.destination_candidates)
            raise IndexError("pop pop")

        # adjust reward if invalid action was taken
        if not valid:
            response = self.worst_response
            target = 6 * 60
            self.reward = self._get_reward(response, target, valid=valid)

        return self.state, self.reward, self.is_done, \
                {"last_action": self.last_action, "valid": valid}

    def reset(self):
        self.sim.simulate_big_incident()
        self.t_episode_end = self.sim.log[0, 11:13].sum() / 60
        self.time = self.sim.log[0, 1]
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
