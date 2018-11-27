"""This module provides the ActionSpace object, which is an auxiliary class
to help organize and specify action spaces, without much boilerplate code in
the main environment classes.
"""
import numpy as np


class ActionSpace(object):
    """ Action space of FDAA Reinforcement Learning environments.

    This class can be initialized with any number of stations and sets of
    vehicle types. It automatically creates all possible relocations as separate
    actions.

    Parameters
    ----------
    station_names: array-like (str)
        Names of the stations in the environment.
    vehicle_types: array-like (str)
        The types of vehicles to take actions on.
    """

    def __init__(self, station_names, vehicle_types):

        # create dictionary of all possible actions
        self._action_set = {0: ("NONE", "NONE", "NONE")}
        station_combos = np.array([action for action in product(station_names, repeat=2) if action[0] != action[1]])
        counter = 1
        for v in vehicle_types:
            for origin, dest in station_combos:
                self._action_set[counter] = (v, origin, dest)
                counter += 1

        # store some attributes
        self.dim = len(self._action_set)
        self.vehicle_types = vehicle_types

    def _is_legal(self, relocation, vehicles):
        """ Indicate if a given relocation is possible given the current vehicle positions.

        Parameters
        ----------
        relocation: tuple(str, str, str),
            The proposed relocation as (vehicle type, origin station, destination station).
        vehicles: array-like of fdsim.Vehicle objects,
            The vehicles with their current positions and availability.
        """
        origins = [v.current_station for v in vehicles if v.type == relocation[0] and v.available]
        if relocation[1] in origins:
            return True
        else:
            return False

    def contains(self, x):
        """ Indicate whether something is a valid action. """
        if int(x) >= 0 and int(x) < self.dim:
            return True
        else:
            return False

    def get_legal_actions(self, vehicles):
        """ Get the set of actions that is legal given the available vehicles. """
        vehicle_locations = {vtype: np.unique([v.current_station for v in vehicles.values() if v.type == vtype and v.available]) for vtype in self.vehicle_types}
        action_vtypes, action_origins, _ = list(zip(*self._action_set.values()))
        return [0] + [idx for idx in np.arange(1, self.dim) if action_origins[idx] in vehicle_locations[action_vtypes[idx]]]

    @staticmethod
    def _random_uniform_choice(a):
        """ Like np.random.choice but for larger arrays and only uniform. """
        p = np.ones(len(a)) / len(a)
        return a[np.digitize(np.random.sample(), np.cumsum(p))]

    def sample_random_action(self, vehicles):
        """ Return random action among the legal actions. """
        return self._random_uniform_choice(self.get_legal_actions(vehicles))

    def get_action_meaning(self, idx):
        """ Get the relocation corresponding to a certain action.

        Parameters
        ----------
        idx: int,
            The action index to get the meaning for.

        Returns
        -------
        relocation: tuple(str, str, str)
            The relocation corresponding to idx as a tuple like
            (vehicle type, origin station, destination station).
        """
        return self._action_set[idx]
