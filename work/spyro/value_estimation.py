"""This module is aimed specifically at gathering experiences from FireCommanderV2 by
using parallel worker-simulators to gather experiences from specific states. The goal
is to obtain state-value estimates of all (or at least the most relevant) states.
"""
import numpy as np
import multiprocessing as mp
import queue
import pickle
import cProfile

from abc import abstractmethod
from itertools import product

from spyro.utils import make_env, obtain_env_information


# global variables specifying some FireCommanderV2 characteristics
NUM_STATIONS = 17
FIRECOMMANDERV2_MAX_VEHICLES = [2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
FIRECOMMANDERV2_MAX_VEHICLES_DAY_HOUR = FIRECOMMANDERV2_MAX_VEHICLES + [6, 23]


def extract_vehicles_from_state(state):
    return state[:NUM_STATIONS]


def extract_vehicles_day_hour_from_state(state):
    return state[:NUM_STATIONS+2]


class BaseParallelValueEstimator(object):
    """Base class that deploys parallel workers to gather experiences from an environment.
    Not useful to instantiate on its own.

    Parameters
    ----------
    num_workers: int, default=-1
        The number of worker processes to use. If -1, uses one per available per CPU core.
    """

    def __init__(self, num_workers=-1):
        # set number of worker processes
        if num_workers == -1:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
        print("Using {} workers".format(self.num_workers))

        # initialize task and result queues
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        # set spawn method for consistency
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            print("multiprocessing method not (re)set to 'spawn', because context was already given.")

    def define_tasks(self, include_time=False, reps=100, debug_subset=None):
        """Define the states that will be explored by the worker processes.

        Parameters
        ----------
        include_time: bool, default=False
            Whether to include day of week and hour of day in the states to permute.
        reps: int, default=100
            The number of repetitions for each state.
        permute: bool, default=True
            Whether to force all permutations to be visited or just to simulate
            according to the probabilities in the environment.
        """
        if include_time:
            self.state_processor = extract_vehicles_day_hour_from_state
            self.max_values = FIRECOMMANDERV2_MAX_VEHICLES_DAY_HOUR
        else:
            self.state_processor = extract_vehicles_from_state
            self.max_values = FIRECOMMANDERV2_MAX_VEHICLES

        self.total_vehicles = np.sum(self.max_values[:NUM_STATIONS])

        # create exhaustive list of states
        ranges = [np.arange(0, y + 1) for y in self.max_values]
        all_states = np.array([x for x in product(*ranges)])
        state_sums = [all_states[i, :].sum() for i in range(len(all_states))]

        tasks = [
            {"state": all_states[i, :], "available": state_sums[i],
             "deployed": self.total_vehicles - state_sums[i], "reps": reps}
            for i in range(len(all_states))
            if (state_sums[i] != 0) and (state_sums[i] != self.total_vehicles)
        ]

        if debug_subset is not None:
            return tasks[:debug_subset]
        else:
            return tasks

    def run(self, env_cls, include_time=False, reps=100, env_params=None, timeout=10, debug_subset=None):
        """Gather experiences.

        Parameters
        ----------
        env_cls: Python class
            The environment to gather experiences from. This class was designed for
            FireCommanderV2, but similar environments might work as well.
        include_time: bool, default=False
            Whether to include day of the week and hour of the day in the state
            representation. Note: setting to True significantly increases the number
            of available states, and thus run time.
        reps: int, default=100
            The number of repetitions/experiences to gather for each state.
        env_params: dict, default=None
            Key-value pairs passed to env_cls.
        timeout: int, default=10
            The maximum time to wait for workers to produce results. After timeout
            seconds, the main process stops getting results from the queue and
            wraps up the other processes.
        """
        # define tasks and put them in the queue
        tasks = self.define_tasks(include_time=include_time, reps=reps, debug_subset=debug_subset)
        self.counter = 0
        self.num_tasks = len(tasks)
        _ = list(map(self.task_queue.put, tasks))
        print("Put {} tasks in Queue (queue length: {})".format(self.num_tasks, self.task_queue.qsize()))

        # initialize workers
        workers = [
            ExperienceGatheringProcess(
                env_cls, self.task_queue, self.result_queue,
                env_params=env_params, state_processor=self.state_processor
            )
            for _ in range(self.num_workers)
        ]

        for worker in workers:
            worker.start()

        try:
            while True:
                try:
                    performed_task = self.result_queue.get(block=True, timeout=timeout)
                    self.process_performed_task(performed_task)
                    self.counter += 1
                    print("\rperformed {} / {} tasks".format(self.counter, self.num_tasks), end="")
                except queue.Empty:
                    print("\nQueue is empty. Breaking loop.")
                    break

        except KeyboardInterrupt:
            pass

        for worker in workers:
            if worker.is_alive():
                worker.join()

    @abstractmethod
    def process_performed_task(self, task):
        """Process the result of a performed task. May vary for different implementations"""


class TabularValueEstimator(BaseParallelValueEstimator):
    """Class that gathers experiences from all states using parallel workers and stores its
    performanc characteristics in a table."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table = {}

    def process_performed_task(self, task):
        """Store results of a task in a table."""
        state = tuple(task.pop("state"))
        self.table[state] = task

    def save_table(self, path="../results/state_value_table.pkl"):
        pickle.dump(self.table, open(path, "wb"))
        print("Table saved at {}".format(path))

    def load_table(self, path):
        self.table = pickle.load(open(path, "rb"))


class ExperienceGatheringProcess(mp.Process):
    """Worker-class that gathers experiences from specific states to obtain
    estimates of state-values.
    """

    def __init__(self, env_cls, task_queue, result_queue, env_params=None, state_processor=None):
        super().__init__()
        self.env_cls = env_cls
        self.env_params = env_params
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.state_processor = state_processor
        print("Worker initialized.")

    def run(self, profile=False, path="../results/profile_run.stats"):
        """Call the main functionality of the class."""
        if profile:
            cProfile.runctx('self._run()', globals(), locals(), path)
        else:
            self._run()

    def _run(self):
        """Start interacting with the environment and sending experiences to
        the global queue.
        """
        print("Run is called.")
        try:
            self.action_shape, self.n_actions, self.obs_shape, _  = \
                    obtain_env_information(self.env_cls, self.env_params)
            self.env = make_env(self.env_cls, self.env_params)
        except:
            print("Exception in env creation")

        while True:
            try:
                task = self.task_queue.get(timeout=1)
                self.perform_task(task)
            except queue.Empty:
                print("Empty task queue found at worker. Shutting down worker.")
                break

    def perform_task(self, task):
        """Perform a given task."""
        responses = np.zeros(task["reps"])
        targets = np.zeros(task["reps"])

        for i in range(task["reps"]):

            success = False

            while not success:
                state = self.state_processor(self.env.reset(forced_vehicles=task["deployed"]))
                self.manipulate_state(state, task["state"])
                response, target = self.env._simulate()
                if (response is not None) and (response != np.inf):
                    success = True

            responses[i], targets[i] = response, target

        task["responses"] = responses
        task["targets"] = targets
        self.result_queue.put(task)

    def manipulate_state(self, current_state, desired_state):
        """Move vehicles so that the desired state is obtained.

        Total number of vehicles must be the same in current_state
        and desired_state, otherwise this method will hang in an
        infinite loop.
        """
        delta = desired_state - current_state
        origins, destinations = [], []

        while not np.all(delta == 0):
            extra_origins = np.flatnonzero(delta < 0)
            origins = np.append(origins, extra_origins)
            extra_destinations = np.flatnonzero(delta > 0)
            destinations = np.append(destinations, extra_destinations)
            delta[extra_origins] += 1
            delta[extra_destinations] -= 1

        for i in range(len(origins)):
            self.env.sim.fast_relocate_vehicle("TS",
                                               self.env.station_names[int(origins[i])],
                                               self.env.station_names[int(destinations[i])]
                                               )
