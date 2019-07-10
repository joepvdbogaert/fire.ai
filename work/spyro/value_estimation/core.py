"""This module is aimed specifically at gathering experiences from FireCommanderV2 by
using parallel worker-simulators to gather experiences from specific states. The goal
is to obtain state-value estimates of all (or at least the most relevant) states.
"""
import numpy as np
import multiprocessing as mp
import time
import queue

from abc import abstractmethod
from itertools import product

from spyro.utils import progress, make_env


# global variables specifying some FireCommanderV2 characteristics
NUM_STATIONS = 17
FIRECOMMANDERV2_MAX_VEHICLES = [2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
FIRECOMMANDERV2_MAX_VEHICLES_DAY_HOUR = FIRECOMMANDERV2_MAX_VEHICLES + [6, 23]
STATION_NAMES = ['AALSMEER', 'AMSTELVEEN', 'ANTON', 'DIEMEN', 'DIRK', 'DRIEMOND',
                 'DUIVENDRECHT', 'HENDRIK', 'IJSBRAND', 'NICO', 'OSDORP', 'PIETER',
                 'TEUNIS', 'UITHOORN', 'VICTOR', 'WILLEM', 'ZEBRA']


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

    def __init__(self, num_workers=-1, max_queue_size=100, include_time=False, name="ValueEstimator",
                 strategy='random', verbose=True):
        """Initialize general parameters."""
        self.verbose = verbose
        self.strategy = strategy

        # set number of worker processes
        if num_workers == -1:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
        # progress("Using {} workers".format(self.num_workers), verbose=self.verbose)

        # other parameters
        self.max_queue_size = max_queue_size
        self.name = name
        self.include_time = include_time

        # set state characteristics
        if include_time:
            self.state_processor = extract_vehicles_day_hour_from_state
            self.max_values = FIRECOMMANDERV2_MAX_VEHICLES_DAY_HOUR
        else:
            self.state_processor = extract_vehicles_from_state
            self.max_values = FIRECOMMANDERV2_MAX_VEHICLES

        self.total_vehicles = np.sum(self.max_values[:NUM_STATIONS])
        self.state_shape = (len(self.max_values),)

        # set spawn method for consistency
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            progress("multiprocessing method not (re)set to 'spawn', because context was "
                     "already given.", verbose=self.verbose)

    def define_tasks(self, reps=100, debug_subset=None):
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

    def perform_tasks(self, env_cls, reps=100, env_params=None, timeout=10,
                      debug_subset=None):
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
        # define tasks and put them in a global queue
        tasks = self.define_tasks(reps=reps, debug_subset=debug_subset)
        self.global_counter = 0
        self.num_tasks = len(tasks)

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        _ = list(map(self.task_queue.put, tasks))
        progress("Put {} tasks in Queue (queue length: {})".format(self.num_tasks, self.task_queue.qsize()), verbose=self.verbose)

        # initialize workers
        workers = [
            ExperienceGatheringProcess(
                env_cls, self.result_queue, task_queue=self.task_queue,
                env_params=env_params, state_processor=self.state_processor,
                strategy='tasks'
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
                    self.global_counter += 1
                    progress("performed {} / {} tasks".format(self.global_counter, self.num_tasks),
                             same_line=True, newline_end=False, verbose=self.verbose)
                except queue.Empty:
                    progress("\nQueue is empty. Breaking loop.", verbose=self.verbose)
                    break

        except KeyboardInterrupt:
            pass

        for worker in workers:
            if worker.is_alive():
                worker.join()

    def gather_random_experiences(self, env_cls, total_steps=50000000, start_step=0, env_params=None,
                                  strategy='random', timeout=3):
        """Collect random experiences from parallel workers.

        Parameters
        ----------
        env_cls: Python class
            The environment to train on.
        total_steps: int, default=50000000
            The total number of experiences to gather.
        env_params: dict, default=None
            Parameters passed to env_cls upon initialization.
        timeout: int, default=3
            The maximum time to wait for an item in the results queue if it is empty.
        """
        self.stop_indicator = mp.Value("i", 0)
        self.global_counter = start_step
        total_steps = total_steps + start_step
        self.result_queue = mp.Queue(self.max_queue_size)

        # initialize workers
        workers = [
            ExperienceGatheringProcess(
                env_cls, self.result_queue, stop_indicator=self.stop_indicator,
                env_params=env_params, state_processor=self.state_processor,
                max_values=self.max_values, strategy=self.strategy
            )
            for _ in range(self.num_workers)
        ]

        for worker in workers:
            worker.start()

        # wait for workers to start delivering
        time.sleep(5)

        try:
            while True:
                try:
                    experience = self.result_queue.get(block=True, timeout=timeout)
                    self.process_random_experience(experience)
                    self.global_counter += 1
                    progress("Processed {} / {} experiences".format(self.global_counter, total_steps),
                             same_line=True, newline_end=False, verbose=self.verbose)
                except queue.Empty:
                    progress("\nQueue is empty. Breaking loop.", verbose=self.verbose)
                    break

                if self.global_counter >= total_steps:
                    if self.stop_indicator.value == 0:
                        with self.stop_indicator.get_lock():
                            self.stop_indicator.value = 1
                        progress("\nSent stop signal to workers. Processing last results in queue.", verbose=self.verbose)

        except KeyboardInterrupt:
            progress("KeyboardInterrupt: sending stop signal and waiting for workers.", verbose=self.verbose)
            with self.stop_indicator.get_lock():
                self.stop_indicator.value = 1

        for worker in workers:
            if worker.is_alive():
                worker.join()

        progress("Workers stopped gracefully.", verbose=self.verbose)

    def fit(self, env_cls, env_params=None, *args, **kwargs):
        """Fit the estimator on the environment."""
        if self.strategy == 'tasks':
            self.perform_tasks(env_cls, env_params=None, *args, **kwargs)
        else:
            self.gather_random_experiences(env_cls, env_params=None, *args, **kwargs)

    @abstractmethod
    def process_performed_task(self, task):
        """Process the result of a performed task. May vary for different implementations"""

    @abstractmethod
    def process_random_experience(self, experience):
        """Process a random experience. May vary for different implementations"""

    @abstractmethod
    def get_config(self):
        """Return the estimator's configuration as a dictionary."""


class ExperienceGatheringProcess(mp.Process):
    """Worker-class that gathers experiences from specific states to obtain
    estimates of state-values.

    Parameters
    ----------
    strategy: str, one of ['random', 'tasks', 'uniform']
        If random, do not manipulate states. If 'tasks', process dictionaries with tasks and
        reps specified. If uniform, sample uniformly over all possible states and return results
        one-by-one.
    """

    def __init__(self, env_cls, result_queue, task_queue=None, stop_indicator=None,
                 state_processor=None, max_values=None, strategy='random', env_params=None, timeout=5,
                 verbose=False):
        super().__init__()
        self.env_cls = env_cls
        self.env_params = env_params
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.state_processor = state_processor
        self.stop_indicator = stop_indicator
        self.strategy = strategy
        self.max_values = max_values
        self.timeout = timeout
        self.verbose = verbose

        if self.strategy == 'tasks':
            assert task_queue is not None, "Must provide a task_queue if strategy='tasks'"
        if self.strategy != 'tasks':
            assert stop_indicator is not None, "Must provide a stop_indicator if strategy!='tasks"
        if self.strategy == 'uniform':
            assert max_values is not None, "max_values must be provided when strategy='uniform'"

        progress("Worker initialized.", verbose=self.verbose)

    def run(self):
        """Call the main functionality of the class."""
        if self.strategy == 'tasks':
            self._run_tasks()
        elif self.strategy == 'uniform':
            self._run_uniform()
        elif self.strategy == 'random':
            self._run_randomly()
        else:
            raise ValueError("strategy should be one of ['random', 'tasks', 'uniform']. Got {}"
                             .format(self.strategy))

    def _make_env(self):
        try:
            self.env = make_env(self.env_cls, self.env_params)
        except:
            print("Exception in env creation")

    def _run_tasks(self):
        """Start interacting with the environment to obtain specifically requested
        experiences (tasks) and send the results to the global queue.
        """
        progress("Start peforming tasks.", verbose=self.verbose)
        self._make_env()

        while True:
            try:
                task = self.task_queue.get(timeout=1)
                self.perform_task(task)
            except queue.Empty:
                progress("Empty task queue found at worker. Shutting down worker.", verbose=self.verbose)
                break

    def _run_randomly(self):
        """Start interacting with the environment without manipulating the state in-between
        steps and send the result of each step to the global results queue.
        """
        progress("Start obtaining experiences.", verbose=self.verbose)
        self._make_env()

        while self.stop_indicator.value != 1:

            # start episode by resetting env
            state = self.state_processor(self.env.reset())
            done = False

            # gather experiences until episode end
            while not done:
                response, target = self.env._simulate()

                if (response is not None) and (response != np.inf):
                    try:
                        self.result_queue.put(
                            {"state": state, "response": response, "target": target},
                            block=True, timeout=self.timeout
                        )
                    except queue.Full:
                        progress("Queue has been full for {} seconds. Breaking."
                                 .format(self.timeout), verbose=self.verbose)
                        break

                raw_state, done = self.env._extract_state(self.env._get_available_vehicles())
                state = self.state_processor(raw_state)

    def _run_uniform(self):
        """Manipulate the state to ensure uniform sampling over all possible states."""
        progress("Start sampling state values uniformly over states.", verbose=self.verbose)

        # find all states and create a generator to sample efficiently
        ranges = [np.arange(0, y + 1) for y in self.max_values]
        all_states = np.array([x for x in product(*ranges)])
        state_gen = self._state_generator(all_states, total_vehicles=np.sum(self.max_values))

        # init env
        self._make_env()

        while self.stop_indicator.value != 1:

            sampled_state, num_deployed = next(state_gen)

            while True:
                state = self.state_processor(self.env.reset(forced_vehicles=num_deployed))
                self.manipulate_state(state, sampled_state)
                response, target = self.env._simulate()
                if (response is not None) and (response != np.inf):
                    try:
                        self.result_queue.put(
                            {"state": sampled_state, "response": response, "target": target},
                            block=True, timeout=self.timeout
                        )
                    except queue.Full:
                        progress("Queue has been full for {} seconds. Breaking."
                                 .format(self.timeout), verbose=self.verbose)

                    break

    def _state_generator(self, all_states, total_vehicles=21):
        """Generate states uniformly."""
        indices = np.random.randint(0, len(all_states), size=50000)
        counter = 0
        while True:
            try:
                s = all_states[indices[counter], :]
                yield s, int(total_vehicles - np.sum(s))
            except IndexError:
                counter = 0
                np.random.randint(0, len(all_states), size=50000)

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
