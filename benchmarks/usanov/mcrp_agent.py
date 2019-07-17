import os
import numpy as np
import pandas as pd
import pickle
import msgpack
import copy
import pulp

from fdsim.helpers import create_service_area_dict

import sys; sys.path.append("../../work")
from spyro.utils import obtain_env_information, make_env, progress


STATION_NAME_TO_AREA = {
    'ANTON': '13781551',
    'AALSMEER': '13670052',
    'AMSTELVEEN': '13710131',
    'DIRK': '13780387',
    'DIEMEN': '13560037',
    'DRIEMOND': '13780419',
    'DUIVENDRECHT': '13750041',
    'HENDRIK': '13780449',
    'NICO': '13781234',
    'UITHOORN': '13760075',
    'OSDORP': '13780011',
    'PIETER': '13780057',
    'TEUNIS': '13780583',
    'VICTOR': '13780255',
    'WILLEM': '13780402',
    'IJSBRAND': '13780162',
    'ZEBRA': '13780194'
}
STATION_AREA_TO_NAME = {value: key for key, value in STATION_NAME_TO_AREA.items()}
STATION_NAMES = ['AALSMEER', 'AMSTELVEEN', 'ANTON', 'DIEMEN', 'DIRK', 'DRIEMOND',
                 'DUIVENDRECHT', 'HENDRIK', 'IJSBRAND', 'NICO', 'OSDORP', 'PIETER',
                 'TEUNIS', 'UITHOORN', 'VICTOR', 'WILLEM', 'ZEBRA']
NUM_STATIONS = len(STATION_NAMES)
STATION_NAME_TO_ACTION = {STATION_NAMES[i]: i for i in range(len(STATION_NAMES))}


def save_station_areas(save_path='./data/updated_kazerne_set.msg'):
    """Save the set of areas in which there is a fire station.

    Parameters
    ----------
    save_path: str
        The path to save the msgpack file.
    """
    new_kazerne_set = list(STATION_NAME_TO_AREA.values())
    with open(save_path, 'wb') as f:
        msgpack.pack(new_kazerne_set, f)
    return new_kazerne_set


def create_new_rates_dict(incidents, kvt_path, station_mapping=STATION_NAME_TO_AREA, vehicle_type='TS',
                          loc_col='hub_vak_bk', save=True, save_path='./data/updated_rates_dict.msg'):
    """Calculate the overall incident rates per service area and return / save them in the format
    required by the MCRP Agent.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data. Must be filtered to all relevant cases (e.g., incidents with a deployment
        of the vehicle type of interest or with high priority); all we do here is count
        the number of incidents in each service area.
    kvt_path: str
        Path to the KVT data.
    station_mapping: dict
        Like {'STATION_NAME' -> 'area_code'}. Must contain all relevant stations.
    vehicle_type: str, one of ['TS', 'HV', 'RV', 'WO']
        The vehicle type to consider.
    """
    service_areas = create_service_area_dict(kvt_path, station_filter=list(station_mapping.keys()))
    rates = {}
    for s, locs in service_areas[vehicle_type].items():
        rates[STATION_NAME_TO_AREA[s]] = int(np.sum(np.in1d(incidents[loc_col].values, locs)))

    if save:
        with open(save_path, 'wb') as f:
            msgpack.pack(rates, f)

    return rates


def create_new_travel_time_input(path, areas, in_minutes=True, save=True, save_path='./data/updated_traveltimes.pickle'):
    """Create a matrix of travel / relocation times between the areas that have
    a fire station.

    Parameters
    ----------
    path: str
        The path to the full time matrix.
    areas: list(str)
        The area codes in which there is a station.
    in_minutes: bool, default=True
        If True, reports travel time in minutes, otherwise in seconds.
    save: bool, default=True
        Whether to save the resulting matrix at save_path.
    save_path: str, default='./data/updated_traveltimes.pickle'
        The path to save the result as a pickle.

    Returns
    -------
    travel_matrix: pd.DataFrame
        Matrix of shape [len(areas), len(areas)], with the travel times.
    """
    time_matrix = pd.read_csv(path, index_col='index', dtype={'index': str})
    travel_times = time_matrix.loc[areas, areas]
    if in_minutes:
        travel_times = np.round(travel_times / 60, 1)
    if save:
        pickle.dump(travel_times, open(save_path, 'wb'))
    return travel_times


def extract_vehicles_from_state(state):
    return state[:NUM_STATIONS]


def extract_current_destination_area(state):
    return STATION_NAME_TO_AREA[
        STATION_NAMES[np.flatnonzero(state[NUM_STATIONS:(2 * NUM_STATIONS)])[0]]
    ]


def area_to_action(area):
    if area is None:
        return NUM_STATIONS
    else:
        return STATION_NAME_TO_ACTION[STATION_AREA_TO_NAME[area]]


class MCRPAgent():
    """Agent based on _Fire Truck Relocation During Major Incidents_ (Usanov et al., 2019)."""

    def __init__(self, W=1, trucks_per_rn=1, kazerne_path='./data/updated_kazerne_set.msg', regions_path='./data/Incidence.p',
                 travel_times_path='./data/updated_traveltimes.pickle', rates_path='./data/updated_rates_dict.msg'):
        """Load required data and save parameters."""
        # load data
        with open(kazerne_path, "rb") as f:
            self.kazerne_set = msgpack.unpackb(f.read(), raw = False)
        self.incidence = pickle.load(open(regions_path, "rb"))
        self.traveltimes_fs = pickle.load(open(travel_times_path, "rb"))
        with open(rates_path, "rb") as f:
            self.rates_dict = msgpack.unpackb(f.read(), raw=False)

        # parameters for willingness to relocate (W) and
        # minimum number of trucks per response neighborhood
        self.W = W
        self.trucks_per_rn = trucks_per_rn

    def evaluate(self, env_cls, n_episodes=10000, tmax=None, env_params=None):
        """Evaluate the agent on an environemt without training.

        Parameters
        ----------
        env_cls: uninitialized Python class or str
            The environment to train on. If a class is provided, it must be uninitialized.
            Parameters can be passed to the environment using env_params. If a string
            is provided, this string is fed to `gym.make()` to create the environment.
        n_episodes: int, optional, default=10,000
            The number of episodes to run.
        tmax: int, optional, default=None
            The maximum number of steps to run in each episode. If None, set to 10,000 to
            not enforce a limit in most environments.
        env_params: dict, optional, default=None
            Dictionary of parameter values to pass to `env_cls` upon initialization.
        """
        if tmax is None:
            self.tmax = 10000
        else:
            self.tmax = tmax

        self.env = make_env(env_cls, env_params)
        self.action_shape, self.n_actions, self.obs_shape, _ = \
                obtain_env_information(env_cls, env_params)

        self.episode_counter = 0
        self.step_counter = 0
        self.done = True

        self.eval_results = {
            "total_episode_reward": np.zeros(n_episodes),
            "mean_episode_reward": np.zeros(n_episodes),
            "episode_length": np.zeros(n_episodes),
        }

        seen_states = {}
        for ep in range(n_episodes):
            self.state = np.asarray(self.env.reset(), dtype=np.int16)
            self.episode_step_counter = 0
            self.episode_reward = 0

            for i in range(self.tmax):

                # get relocations from dictionary if problem was solved before
                # otherwise solve it and save the results for next time
                try:
                    relocations = seen_states[tuple(extract_vehicles_from_state(self.state))]
                except KeyError:
                    relocations = self.get_relocations(self.state)
                    seen_states[tuple(extract_vehicles_from_state(self.state))] = relocations

                # get origin if current destination is in the relocations
                to_from = {d['to']: d['from'] for d in relocations.values()}
                destination_area = extract_current_destination_area(self.state)
                origin_area = to_from[destination_area] if destination_area in list(to_from.keys()) else None

                # select and perform action
                self.action = area_to_action(origin_area)
                new_state, self.reward, self.done, _ = self.env.step(self.action)

                # bookkeeping
                self.step_counter += 1
                self.episode_reward += self.reward
                self.episode_step_counter += 1
                self.state = np.asarray(copy.copy(new_state), dtype=np.int16)

                # end of episode
                if self.done:
                    break

            self.eval_results["total_episode_reward"][ep] = self.episode_reward
            self.eval_results["mean_episode_reward"][ep] = self.episode_reward / self.episode_step_counter
            self.eval_results["episode_length"][ep] = self.episode_step_counter

            progress("Completed episode {}/{}".format(ep + 1, n_episodes),
                     same_line=(ep > 0), newline_end=(ep + 1 == n_episodes))

        return self.eval_results
    
    def _state_to_fleet_dict(self, state):
        """Translate an array of vehicles per station to the fleet dict required
        for the MCRP algorithm.
        """
        vehicles = extract_vehicles_from_state(state)
        return {STATION_NAME_TO_AREA[STATION_NAMES[i]]: [vehicles[i], vehicles[i]]
                for i in range(len(vehicles))}

    def get_relocations(self, state):
        """Get the relocations for a given vehicles availability.

        Assumes all vehicles in `vehicles` are available for relocation.

        Parameters
        ----------
        state: array-like, 1D
            The number of vehicles available at the stations, in the order
            of STATION_NAMES, plus further observations.

        Returns
        -------
        fleet: dict
            The fleet dictionary like {'area code' -> [x, x]}, where
            x is the number of available vehicles according to the
            provided array.
        """
        fleet = self._state_to_fleet_dict(state)
        return self._get_relocations_from_fleet(fleet)

    def _get_relocations_from_fleet(self, fleet_dict):
        """Get the relocations suggested by the MCRP-LBAP solution for a given fleet."""
        for s in range(1, 20):
            A = self.incidence[s - 1]
            opt_model, relocations, fleet_after = self._solve_mcrp(A, fleet_dict)
            if opt_model.status == 1:
                break

        movements = self._solve_lbap(relocations)
        return movements

    def _solve_mcrp(self, A, fleet_dict):
        """Solve the Maximum Coverage Relocation Problem."""
        # arrival rates per service area
        rates = [self.rates_dict[x] for x in self.kazerne_set]
        # available fleet per station
        #fleet = [fleet_dict[x][0] for x in kazerne_set];
        fleet = [fleet_dict[x][0] for x in self.kazerne_set];
        # available for relocation
        active = [fleet_dict[x][1] for x in self.kazerne_set];

        ## Sets used for iteration over fire satations (kazerne)
        ## and response neighborhoods (rn)
        kazerne_range = range(len(self.kazerne_set))
        rn_set = range(A.shape[1])

        ## Sets of station ID's
        # empty stations
        empty_id = [x for x in kazerne_range if fleet[x] == 0]
        # stations with only one truck
        single_id = [x for x in kazerne_range if fleet[x] == 1]
        # stations with more than one truck
        many_id = [x for x in kazerne_range if fleet[x] > 1]
        ## Model
        # create 'opt_model' variable that contains problem data
        opt_model = pulp.LpProblem("MCRP", pulp.LpMaximize)

        ## VARIABLES: if relocation is made from station i to station j
        relocate = {(i, j):
                    pulp.LpVariable(cat='Binary',
                               name="relocate_{0}_{1}".format(i,j))
                    for i in kazerne_range for j in kazerne_range}

        ## VARIABLES: fleet after relocations are made
        fleet_after = {i:
                       pulp.LpVariable(cat='Integer',
                                  name="fleet_after_{0}".format(i))
                       for i in kazerne_range}

        ## VARIABLES: indicates if the station is unoccupied
        unoccupied = {i:
                      pulp.LpVariable(cat='Binary',
                                 name="unoccupied_{0}".format(i))
                      for i in many_id}

        ## OBJECTIVE
        objective = pulp.lpSum((self.W*(rates[j] - rates[i]) + (self.W - 1))*relocate[(i, j)] for i in single_id for j in empty_id) \
                    + pulp.lpSum((self.W*rates[j] + (self.W - 1))*relocate[(i, j)] for i in many_id for j in empty_id) \
                    + pulp.lpSum(-self.W*rates[i]*unoccupied[i] for i in many_id)

        # Constraint 1. Every RN should be covered by trucks_per_rn trucks
        constraints_1 = {r:
        pulp.LpConstraint(
                    e=pulp.lpSum(A[j][r]*fleet_after[j] for j in kazerne_range),
                    sense=pulp.LpConstraintGE,
                    rhs=self.trucks_per_rn,
                    name="constr_rn_{0}".format(r))
                for r in rn_set}

        # Constraint 2. Trucks flow control
        constraints_2 = {i:
        pulp.LpConstraint(
                    e= fleet_after[i] - pulp.lpSum(relocate[(j, i)] for j in kazerne_range) + pulp.lpSum(relocate[(i, j)] for j in kazerne_range),
                    sense=pulp.LpConstraintEQ,
                    rhs=fleet[i],
                    name="constr_move_{0}".format(i))
                for i in kazerne_range}

        # Constraint 3. Do not relocate more than available
        constraints_3 = {i:
        pulp.LpConstraint(
                    e=pulp.lpSum(relocate[(i, j)] for j in kazerne_range),
                    sense=pulp.LpConstraintLE,
                    rhs=active[i],
                    name="constr_reloc_avail_{0}".format(i))
                for i in kazerne_range}

        # Constraints 4. Fleet should be positive
        constraints_4 = {i:
        pulp.LpConstraint(
                    e=fleet_after[i],
                    sense=pulp.LpConstraintGE,
                    rhs=0,
                    name="constr_pos_fleet_{0}".format(i))
                for i in kazerne_range}

        # Constraint 5. Do not relocate more than 1 truck to the same empty station
        constraints_5 = {i:
        pulp.LpConstraint(
                    e=pulp.lpSum(relocate[(j, i)] for j in kazerne_range),
                    sense=pulp.LpConstraintLE,
                    rhs=1,
                    name="constr_single_truck_{0}".format(i))
                for i in empty_id}

        # Constraint 6. Do not relocate to full stations
        constraints_6 = {(i, j):
        pulp.LpConstraint(
                    e=relocate[(i, j)],
                    sense=pulp.LpConstraintEQ,
                    rhs=0,
                    name="constr_not_to_full_{0}_{1}".format(i, j))
                for i in kazerne_range for j in kazerne_range if j not in empty_id}

        # Constraint 7. Unoccupied consistency
        constraints_7 = {i:
        pulp.LpConstraint(
                    e=fleet_after[i]+unoccupied[i],
                    sense=pulp.LpConstraintGE,
                    rhs=1,
                    name="constr_unoccpd_consistency_{0}".format(i))
                for i in many_id}

        # add constraints to the model
        for r in rn_set:
            opt_model += constraints_1[r]
        for i in kazerne_range:
            opt_model += constraints_2[i]
            opt_model += constraints_3[i]
            opt_model += constraints_4[i]
            for j in [x for x in kazerne_range if x not in empty_id]:
                opt_model += constraints_6[(i, j)]

        for i in empty_id:
            opt_model += constraints_5[i]

        for i in many_id:
            opt_model += constraints_7[i]

        # add objective to the model
        opt_model += objective

        # solve the model
        opt_model.solve()

        # output solution
        try:
            relocate_m = [[int(relocate[(i, j)].varValue) for j in kazerne_range] for i in kazerne_range]
            fleet_after_m = [int(fleet_after[i].varValue) for i in kazerne_range]
        except:
            relocate_m = relocate,
            fleet_after_m = fleet_after

        return opt_model, relocate_m, fleet_after_m

    def _solve_lbap(self, relocations):
        """Assign origin stations to destination stations by solving the
        Linear Bottleneck Assignment Problem.
        """
        # make the set of origins
        # and the set of destinations
        origins_list = [[x for y in range(len(relocations[x])) if relocations[x][y] > 0] for x in range(len(relocations))]
        origins = [item for sublist in origins_list for item in sublist]
        #origins = [x for x in range(len(kazerne_set)) if sum(relocations[x]) > 0]
        destinations_list = [[y for y in range(len(relocations[x])) if relocations[x][y] > 0] for x in range(len(relocations))]
        destinations = [item for sublist in destinations_list for item in sublist]
        #destinations = [x for x in range(len(kazerne_set)) if sum([row[x] for row in relocations]) > 0]

        # set of origins' indeces
        origin_set = range(len(origins))
        # set of destinations' indeces
        destination_set = range(len(destinations))

        ## Model
        # create 'opt_model' variable that contains problem data
        opt_model = pulp.LpProblem("LBAP", pulp.LpMinimize)

        # Relocation decision variables:
        relocate = {(i, j):
                    pulp.LpVariable(cat='Binary',
                                   name="relocate_{0}_{1}".format(i,j))
                    for i in origin_set for j in destination_set}

        # Dummy decision variable - maximum travelling time
        maxtime = {0:
                   pulp.LpVariable(cat='Continuous',
                                  name="dummy_max")}

        # Dummy decision variable - total travelling time of all except the maximum
        totaltime = {0:
                     pulp.LpVariable(cat='Continuous',
                                    name="dummy_total")}

        ## OBJECTIVE
        # The objective is to minimize the maximum travelling time and then the total travelling time
        objective = 1000*maxtime[0]+totaltime[0]


        # Constraint 1. Every travel time should be less than the maximum travel time
        constraints_1 = {(i, j):
        pulp.LpConstraint(
                        e=self.traveltimes_fs.iloc[[origins[i]], [destinations[j]]].values[0][0]*relocate[(i, j)]-maxtime[0],
                        sense=pulp.LpConstraintLE,
                        rhs=0,
                        name="max_time_{0}_{1}".format(i, j))
                for i in origin_set for j in destination_set}

        # Constraint 2. Use each origin exactly once
        constraints_2 = {i:
        pulp.LpConstraint(
                        e=pulp.lpSum(relocate[(i, j)] for j in destination_set),
                        sense=pulp.LpConstraintEQ,
                        rhs=1,
                        name="reloc_or_{0}".format(i))
                for i in origin_set}

        # Constraint 3. Use each destination exactly once
        constraints_3 = {i:
        pulp.LpConstraint(
                        e=pulp.lpSum(relocate[(j, i)] for j in origin_set),
                        sense=pulp.LpConstraintEQ,
                        rhs=1,
                        name="reloc_dest_{0}".format(i))
                for i in destination_set}

        # Constraint 4. Definition of the totaltime dummy variable
        constraints_4 = {0:
        pulp.LpConstraint(
                        e=pulp.lpSum(self.traveltimes_fs.iloc[[origins[i]], [destinations[j]]].values[0][0]*relocate[(i, j)] for i in origin_set for j in destination_set)-totaltime[0],
                        sense=pulp.LpConstraintEQ,
                        rhs=0,
                        name="total_time")
                        }

        # add objective to the model
        opt_model += objective

        # add constraints to the model
        for i in origin_set:
            opt_model += constraints_2[i]
            for j in destination_set:
                opt_model += constraints_1[(i, j)]
        for i in destination_set:
            opt_model += constraints_3[i]
        opt_model += constraints_4[0]

        # solve the model
        opt_model.solve()

        # output solution
        try:
            relocate_m = {}
            indx = 1
            for i in origin_set:
                for j in destination_set:
                    if relocate[(i, j)].varValue > 0:
                        relocate_m[indx] = {'from': self.kazerne_set[origins[i]], 'to': self.kazerne_set[destinations[j]]}
                        indx += 1
        except:
            relocate_m = relocate

        return relocate_m
