import numpy as np

from fdsim.simulation import Simulator
from gym_firecommander.envs import FireCommanderEnv


class FireCommanderBigEnv(FireCommanderEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        """Take one step in the environment and give back the results.

        Parameters
        ----------
        action: tuple(int, int)
            The action to take as combination of (origin, destination).

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
        # check if suggested action is valid
        valid = self._take_action(action)
        self.last_action = action if self.action_type == "tuple" else self.action_num_to_tuple[action]

        response = np.inf
        while response == np.inf:
            response, target = self._simulate()

        if response is None:
            # terminal state reached
            self.reward = 0.0
            self.state, self.is_done = self._extract_state()
            return self.state, self.reward, self.is_done, {"note": "nothing to report"}

        elif not valid:
            response = self.worst_response
            target = 6*60
                
        elif np.isnan(target):  # prio 2 or 3 incident: no target exists
            target = response

        # calculate reward and new state
        self.reward = self._get_reward(response, target, valid=valid)
        self.state, self.is_done = self._extract_state()
        return self.state, self.reward, self.is_done, {"note": "nothing to report"}

    def reset(self):
        """Reset by starting a new episode. This is done by simulating a big incident that
        requires at least 3 TS deployments.
        """
        self.sim.simulate_big_incident()
        self.t_episode_end = self.sim.log[0, 11:13].sum() / 60
        self.time = self.sim.log[0, 1]
        self.state, _ = self._extract_state()
        return self.state

    def _check_episode_start_condition(self):
        """Check whether the simulator is at a situation where a train episode can start."""
        return self.sim.t < self.t_episode_end

    def _check_episode_end_condition(self):
        """Check whether the simulator is at a situation where a train episode ends."""
        return self.sim.t >= self.t_episode_end

    def _simulate(self):
        """ Simulate a single incident and all corresponding deployments"""
        # sample incident and update status of vehicles at new time t
        self.sim.t, self.time, type_, loc, prio, req_vehicles, func, dest = self.sim._sample_incident()
        self.sim._update_vehicles(self.sim.t, self.time)

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

                vehicle, estimated_time = self.sim._pick_vehicle(loc, v)
                if vehicle is None:
                    turnout, travel, onscene, response = [np.nan]*4

                else:
                    vehicle.assign_crew()

                    turnout, travel, onscene = self.sim.rsampler.sample_response_time(
                        type_, loc, vehicle.current_station_name, vehicle.type, vehicle.current_crew,
                        prio, estimated_time=estimated_time)

                    response = dispatch + turnout + travel
                    vehicle.dispatch(dest, self.sim.t + (response + onscene + estimated_time) / 60)

                # we must return a numerical value
                if np.isnan(response):
                    response = self.worst_response

                if response < min_ts_response:
                    min_ts_response = response

        return min_ts_response, target
