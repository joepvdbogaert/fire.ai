import pickle
import copy
import numpy as np
import pandas as pd

from gym_firecommander.envs import FireCommanderEnv


class FireCommanderTestEnv(FireCommanderEnv):
    """A version of Fire Commander for evaluating agents on pre-simulated data."""

    def __init__(self, load=False, path=None, test_episodes=None, *args, **kwargs):
        if load:
            assert path is not None, "path must be given if load=True"
            self.load_test_episodes(path=path)

        if test_episodes is not None:
            self.test_episodes = test_episodes
            self.reset_test_episodes()

        super().__init__(*args, **kwargs)

    def create_test_episodes(self, n_episodes, episode_threshold=None):
        """Create a suite of episodes to evaluate agents on."""
        # load simulator to ensure fresh start
        self._load_simulator()
        self.sim.initialize_without_simulating()

        # take own episode threshold if none is given
        if episode_threshold is not None:
            self.episode_threshold = episode_threshold

        # store episodes in dictionary
        self.test_episodes = {}

        # keep track of number of episodes
        ep_counter = 0
        while ep_counter < n_episodes:
            
            while not self._check_episode_start_condition():
                self._simulate()

            # start condition is met: gather current state information
            self.is_done = False
            self.state, _ = self._extract_state()
            snapshot = self.get_snapshot()
            start_idx = self.sim.log_index

            while not self._check_episode_end_condition():
                # simulate a single incident and all its deployments and log to sim.log
                self.sim.simulate_single_incident()

            # end condition is met: gather simulated data
            sim_log = self.sim.log[start_idx:self.sim.log_index, :]
            sim_log = sim_log[sim_log[:, 6] == "TS"]

            # save only if log is not empty (can be empty if there were no TS responses)
            if len(sim_log) > 0:
                self.test_episodes[ep_counter] = {
                    "snapshot": copy.deepcopy(snapshot),
                    "log": sim_log
                }

                ep_counter += 1

        self.reset_test_episodes()

    def get_test_episodes(self):
        assert hasattr(self, "test_episodes"), "First run 'create_test_episodes' or 'load_test_episodes'"
        return copy.deepcopy(self.test_episodes)

    def save_test_episodes(self, path):
        assert self.test_episodes is not None, "Nothing to save"
        pickle.dump(self.test_episodes, open(path, 'wb'))

    def load_test_episodes(self, path=None):
        if path is not None:
            self.test_episodes = pickle.load(open(path, "rb"))
        else:
            raise ValueError("No path given.")
            # note, later we want to provide additional ways to select
            # predefined test suites.
        self.reset_test_episodes()

    def reset_test_episodes(self):
        """Start from the beginning of the test data."""
        self.ep = -1
        self.log_row = 0
        self.data = copy.deepcopy(self.test_episodes)

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
            _, _ = self._simulate_from_data()
            response = self.worst_response
            target = 6*60
        else:
            response, target = self._simulate_from_data()
            if np.isnan(target):  # prio 2 or 3 incident: no target exists
                target = response

        self.last_action = action if self.action_type == "tuple" else self.action_num_to_tuple[action]
        # calculate reward and new state
        self.reward = self._get_reward(response, target, valid=valid)
        self.state, _ = self._extract_state()
        # self.render()
        return self.state, self.reward, self.is_done, {"note": "nothing to report"}

    def reset(self):
        """Reset the environment to start a new episode. This function must be called
        every time when the environment is in a terminal state.

        Note that this overwrites the original reset method of the FireCommanderEnv
        because the episodes are now predefined.
        """
        self.ep += 1
        self.log_row = 0
        self.current_log = self.data[self.ep]["log"]
        self.set_snapshot(self.data[self.ep]["snapshot"])
        return self.state

    def _simulate_from_data(self):
        """Simulate a single incident and all corresponding deployments.

        In contrast to the normal FireCommanderEnv, incidents are given by pre-simulated
        test episodes and so are the dispatch and on-scene times. The travel and turnout
        times are simulated, since these depend on the available vehicles and crews.
        """
        # collect incident information
        self.sim.t, self.time, type_, loc, prio, func, _, _, dispatch, _, _, _, _, target, _, _, _ = \
                self.current_log[self.log_row, :]

        # update vehicles
        self.sim._update_vehicles(self.sim.t, self.time)

        # find required number of vehicles, all are TS vehicles so we need not check for type
        onscene_times = self.current_log[self.current_log[:, 0] == self.sim.t, 11]

        # keep track of minimum TS response time
        min_ts_response = np.inf

        # dispatch all the needed vehicles and collect the turnout and travel time
        for i, onscene in enumerate(onscene_times):

            vehicle, estimated_time = self.sim._pick_vehicle(loc, "TS")

            if vehicle is None:
                turnout, travel = np.nan, np.nan
                response = self.worst_response
                vtype = "EXTERNAL"
                vid = "EXTERNAL"
                current_station = "EXTERNAL"
                base_station = "EXTERNAL"
                crew = "EXTERNAL"
            else:
                vehicle.assign_crew()  # always full time in this case

                turnout = next(self.sim.rsampler.turnout_generators[vehicle.current_crew][prio]["TS"])
                travel = self.sim.rsampler.sample_travel_time(estimated_time, "TS")

                response = dispatch + turnout + travel
                vehicle.dispatch(loc, self.sim.t + (response + onscene + estimated_time) / 60)

                vtype, vid, current_station, base_station, crew = (vehicle.type, vehicle.id,
                    vehicle.current_station_name, vehicle.base_station_name, vehicle.current_crew)

            # we must return a numerical value
            if response < min_ts_response:
                min_ts_response = response

            # log results for this deployment to data for possible further analysis
            self.data[self.ep]["log"][self.log_row + i, :] = [
                self.sim.t, self.time, type_, loc, prio, func, vtype,
                vid, dispatch, turnout, travel, onscene, response, target,
                current_station, base_station, crew
            ]

        # increment the row in the log / multiple steps to the next incident
        # and indicate whether the end of the episode is reached.
        self.log_row += len(onscene_times)
        if self.log_row == len(self.current_log):
            self.is_done = True
        else:
            self.is_done = False

        return min_ts_response, target

    def get_test_log(self):
        """Return the concatenated log of test episodes."""
        concat_log = np.concatenate(
            [np.append(d["log"], np.ones((len(d["log"]), 1)) * key, axis=1)
             for key, d in self.data.items()]
        )
        return pd.DataFrame(concat_log, columns=self.sim.log_columns + ["episode"])
