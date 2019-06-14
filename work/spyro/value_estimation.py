"""This module is aimed specifically at gathering experiences from FireCommanderV2 by
using parallel worker-simulators to gather experiences from specific states. The goal
is to obtain state-value estimates of all (or at least the most relevant) states.
"""
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import queue
import pickle
import copy
import cProfile

from abc import abstractmethod
from itertools import product

from spyro.utils import make_env, obtain_env_information
from spyro.builders import build_mlp_regressor, build_distributional_dqn
from spyro.core import BaseAgent


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

    def __init__(self, num_workers=-1, max_queue_size=100, include_time=False, name="ValueEstimator"):
        """Initialize general parameters."""

        # set number of worker processes
        if num_workers == -1:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
        print("Using {} workers".format(self.num_workers))

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
        tasks = self.define_tasks(include_time=include_time, reps=reps, debug_subset=debug_subset)
        self.global_counter = 0
        self.num_tasks = len(tasks)

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        _ = list(map(self.task_queue.put, tasks))
        print("Put {} tasks in Queue (queue length: {})".format(self.num_tasks, self.task_queue.qsize()))

        # initialize workers
        workers = [
            ExperienceGatheringProcess(
                env_cls, self.result_queue, task_queue=self.task_queue,
                env_params=env_params, state_processor=self.state_processor,
                tasks=True
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
                    print("\rperformed {} / {} tasks".format(self.global_counter, self.num_tasks), end="")
                except queue.Empty:
                    print("\nQueue is empty. Breaking loop.")
                    break

        except KeyboardInterrupt:
            pass

        for worker in workers:
            if worker.is_alive():
                worker.join()

    def gather_random_experiences(self, env_cls, total_steps=50000000, env_params=None, timeout=10):
        """Collect random experiences from parallel workers.

        Parameters
        ----------
        total_steps: int, default=50000000
            The total number of experiences to gather.
        timeout: int, default=3
            The maximum time to wait for an item in the results queue if it is empty.
        """
        self.stop_indicator = mp.Value("i", 0)
        self.global_counter = 0
        self.result_queue = mp.Queue()
        # initialize workers
        workers = [
            ExperienceGatheringProcess(
                env_cls, self.result_queue, stop_indicator=self.stop_indicator,
                env_params=env_params, state_processor=self.state_processor,
                tasks=False
            )
            for _ in range(self.num_workers)
        ]

        for worker in workers:
            worker.start()

        try:
            while True:
                try:
                    experience = self.result_queue.get(block=True, timeout=timeout)
                    self.process_random_experience(experience)
                    self.global_counter += 1
                    print("\rObtained {} / {} experiences".format(self.global_counter, total_steps), end="")
                except queue.Empty:
                    print("\nQueue is empty. Breaking loop.")
                    break

                if self.global_counter >= total_steps:
                    with self.stop_indicator.get_lock():
                        self.stop_indicator.value = 1
                    print("Total steps reached. I sent the stop signal to workers, but will"
                          " keep processing incoming results.")
        except KeyboardInterrupt:
            print("KeyboardInterrupt: sending stop signal and waiting for workers...")
            with self.stop_indicator.get_lock():
                self.stop_indicator.value = 1

        for worker in workers:
            if worker.is_alive():
                worker.join()

        print("Stopped gracefully.")

    def fit(self, env_cls, permutations=False, env_params=None, *args, **kwargs):
        """Fit the estimator on the environment."""
        if permutations:
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
    """

    def __init__(self, env_cls, result_queue, task_queue=None, stop_indicator=None,
                 state_processor=None, tasks=False, env_params=None):
        super().__init__()
        self.env_cls = env_cls
        self.env_params = env_params
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.state_processor = state_processor
        self.stop_indicator = stop_indicator
        self.tasks = tasks

        if self.tasks:
            assert task_queue is not None, "Must provide a task_queue if tasks=True"
        else:
            assert stop_indicator is not None, "Must provide a stop_indicator if tasks=False"

        print("Worker initialized.")

    def run(self):
        """Call the main functionality of the class."""
        if self.tasks:
            self._run_tasks()
        else:
            self._run_randomly()

    def _make_env(self):
        try:
            self.env = make_env(self.env_cls, self.env_params)
        except:
            print("Exception in env creation")

    def _run_tasks(self):
        """Start interacting with the environment to obtain specifically requested
        experiences (tasks) and send the results to the global queue.
        """
        print("Start peforming tasks.")
        self._make_env()

        while True:
            try:
                task = self.task_queue.get(timeout=1)
                self.perform_task(task)
            except queue.Empty:
                print("Empty task queue found at worker. Shutting down worker.")
                break

    def _run_randomly(self):
        """Start interacting with the environment without manipulating the state in-between
        steps and send the result of each step to the global results queue.
        """
        print("Start obtaining experiences.")
        self._make_env()

        while self.stop_indicator != 1:

            # start episode by resetting env
            state = self.state_processor(self.env.reset())
            done = False

            # gather experiences until episode end
            while not done:
                response, target = self.env._simulate()

                if (response is not None) and (response != np.inf):
                    self.result_queue.put(
                        {"state": state, "response": response, "target": target}
                    )

                raw_state, done = self.env._extract_state(self.env._get_available_vehicles())
                state = self.state_processor(raw_state)

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


class TabularValueEstimator(BaseParallelValueEstimator):
    """Class that gathers experiences from all states using parallel workers and stores its
    performanc characteristics in a table."""

    def __init__(self, name="TabularEstimator", *args, **kwargs):
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


class NeuralValueEstimator(BaseParallelValueEstimator, BaseAgent):
    """Class that gathers experiences (values) from states using parallel workers and trains a
    neural network to predict them.

    Parameters
    ----------
    n_neurons: int, default=1024
        The number of neurons in each layer of the neural network.
    n_layers: int, default=4
        The number of hidden layers in the neural network.
    quantiles: bool, default=False
        Whether to use Quantile Regression instead of regular mean estimation.
    num_workers: int, default=-1
        The number of worker processes to use. If -1, uses one per available per CPU core.
    """

    def __init__(self, memory, n_neurons=1024, n_layers=4, quantiles=False, num_atoms=51,
                 activation="relu", optimization="adam", learning_rate=1e-4, name="NeuralEstimator",
                 gradient_clip=None, train_frequency=4, warmup_steps=50000, batch_size=64,
                 log=True,  logdir="./log/value_estimation", *args, **kwargs):

        self.memory = memory
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.quantiles = quantiles
        self.num_atoms = num_atoms
        self.activation = activation
        self.optimization = optimization
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.warmup_steps = warmup_steps
        BaseParallelValueEstimator.__init__(self, *args, **kwargs)
        BaseAgent.__init__(self, None, learning_rate=learning_rate, logdir=logdir, log=log,\
                           log_prefix=self.name + "_run")

        self._init_graph()

    def _init_graph(self):
        """Initialize the Tensorflow graph based on initialization arguments."""
        self.session = tf.Session()
        with tf.variable_scope(self.name):

            self.states_ph = tf.placeholder(tf.float64, shape=(None,) + self.state_shape, name="states_ph")
            self.values_ph = tf.placeholder(tf.float64, shape=(None, 1), name="rewards_ph")

            # Quantile Regression
            if self.quantiles:
                self.value_prediction = build_distributional_dqn(
                    self.states_ph, 1, self.num_atoms,
                    n_layers=self.n_layers, n_neurons=self.n_neurons,
                    activation=self.activation
                )
                raise NotImplementedError("Quantile Regression not implemented yet.")

            # Regular Regression
            else:
                self.value_prediction = build_mlp_regressor(
                    self.states_ph, self.n_layers, self.n_neurons,
                    activation=self.activation, output_dim=1
                )
                self.loss = 0.5 * tf.reduce_mean(
                    tf.squeeze(tf.square(self.value_prediction - tf.stop_gradient(self.values_ph)))
                )

            # Minimize the loss using gradient descent (possibly clip gradients before applying)
            self.optimizer = self.tf_optimizers[self.optimization](learning_rate=self.learning_rate)
            self.weights = tf.trainable_variables(scope=self.name)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.weights)
            if self.gradient_clip is not None:
                a_min, a_max = self.gradient_clip
                self.grads_and_vars = [(tf.clip_by_value(grad, a_min, a_max), var) for grad, var in self.grads_and_vars]
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

        if self.log:
            self.summary_op = tf.summary.scalar("loss", self.loss)
            self.summary_writer = tf.summary.FileWriter(self.logdir, self.session.graph)

        self.session.run(tf.global_variables_initializer())

    def process_random_experience(self, experience):
        """Process an experience by storing it in memory and (at training time) sampling a
        batch from that memory and train on it.

        Parameters
        ----------
        experience: dict
            Contains a description of the experience: keys must be ['state', 'response', 'target'],
            where state is array-like and response and target are scalars (or NaN).
        """
        # store experience in memory
        self.memory.store(copy.copy(experience["state"]), experience["response"], experience["target"])
        
        # train at training time
        if (self.global_counter % self.train_frequency == 0) and (self.global_counter >= self.warmup_steps):

            # sample batch
            states, responses, _ = self.memory.sample(self.batch_size)

            # train and log
            if self.log:
                _, summary = self.session.run(
                    [self.train_op, self.summary_op],
                    feed_dict={
                        self.states_ph: states,
                        self.values_ph: responses
                    }
                )
                self.summary_writer.add_summary(summary, self.global_counter)
            # or just train
            else:
                self.session.run(
                    self.train_op, feed_dict={
                        self.states_ph: states,
                        self.values_ph: responses
                    }
                )

    def get_config(self):
        return {
            "memory": self.memory.get_config(),
            "n_neurons": self.n_neurons,
            "n_layers": self.n_layers,
            "quantiles": self.quantiles,
            "num_atoms": self.num_atoms,
            "activation": self.activation,
            "optimization": self.optimization,
            "learning_rate": self.learning_rate,
            "name": self.name,
            "gradient_clip": self.gradient_clip,
            "train_frequency": self.train_frequency,
            "warmup_steps": self.warmup_steps,
            "batch_size": self.batch_size,
            "log": self.log,
            "logdir": self.logdir
        }
