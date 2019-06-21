"""This module is aimed specifically at gathering experiences from FireCommanderV2 by
using parallel worker-simulators to gather experiences from specific states. The goal
is to obtain state-value estimates of all (or at least the most relevant) states.
"""
import numpy as np
import pandas as pd
import multiprocessing as mp
import tensorflow as tf
import time
import queue
import pickle
import copy
import cProfile

from abc import abstractmethod
from itertools import product
from scipy.stats import wasserstein_distance

from spyro.utils import make_env, obtain_env_information
from spyro.builders import build_mlp_regressor, build_distributional_dqn
from spyro.core import BaseAgent
from spyro.losses import quantile_huber_loss
from spyro.utils import progress


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

    def __init__(self, num_workers=-1, max_queue_size=100, include_time=False, name="ValueEstimator",
                 verbose=True):
        """Initialize general parameters."""
        self.verbose = verbose

        # set number of worker processes
        if num_workers == -1:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
        progress("Using {} workers".format(self.num_workers), verbose=self.verbose)

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

    def gather_random_experiences(self, env_cls, total_steps=50000000, start_step=0, env_params=None, timeout=3):
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
                tasks=False
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
                 state_processor=None, tasks=False, env_params=None, timeout=5,
                 verbose=False):
        super().__init__()
        self.env_cls = env_cls
        self.env_params = env_params
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.state_processor = state_processor
        self.stop_indicator = stop_indicator
        self.tasks = tasks
        self.timeout = timeout
        self.verbose = verbose

        if self.tasks:
            assert task_queue is not None, "Must provide a task_queue if tasks=True"
        else:
            assert stop_indicator is not None, "Must provide a stop_indicator if tasks=False"

        progress("Worker initialized.", verbose=self.verbose)

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
        progress("Table saved at {}".format(path))

    def load_table(self, path):
        self.table = pickle.load(open(path, "rb"))


class DataSetCreator(BaseParallelValueEstimator):
    """Class that gathers experiences from parallel workers and creates a dataset with
    states and responses (and targets) from those states.

    This class is intended to create test (and validation) datasets to evaluate trained
    estimators on.

    Parameters
    ----------
    name: str, default="DataSetCreator"
        Name of the object.
    *args, **kwargs: any
        Parameters passed to the Base class.
    """

    def __init__(self, name="DataSetCreator", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        # use the same method when performing tasks as when obtaining random experiences
        self.process_random_experience = self._process_experience
        self.process_performed_task = self._process_experience

    def create_data(self, env_cls, size=1000000, permutations=False, save=True,
                    env_params=None, save_path="../results/dataset.csv", *args, **kwargs):
        """Generate a dataset.

        Parameters
        ----------
        env_cls: Python class
            The environment to obtain experiences from.
        size: int, default=1000000
            The number of samples to simulate.
        permutations: bool, default=False
            Whether to force every state to be visited (True) or simulate according to patterns
            in the simulation.
        save: bool, default=True
            Whether to save the resulting dataset.
        env_params: dict, default=None
            Dictionary with key-value pairs to pass to the env_cls upon initialization.
        save_path: str, default="../results/dataset.csv"
            Where to save the resulting dataset.
        *args, **kwargs: any
            Arguments passed to the fit method of the BaseParallelValueEstimator.
        """
        self.data_state = np.empty((size,) + self.state_shape)
        self.data_response = np.empty(size)
        self.data_target = np.empty(size)
        self.index = 0
        progress("Creating dataset of {} observations.".format(size))
        self.fit(env_cls, permutations=permutations, total_steps=size, env_params=env_params, *args, **kwargs)
        progress("Dataset created.")
        if save:
            self.save_data(path=save_path)

    def _process_experience(self, experience):
        """Store results of a task in a table."""
        try:
            self.data_state[self.index, :] = experience["state"]
            self.data_response[self.index] = experience["response"]
            self.data_target[self.index] = experience["target"]
            self.index += 1
        except IndexError:
            # if data template is full, skip remaining experiences.
            pass

    def save_data(self, path="../results/dataset.csv"):
        """Save the created data as a csv file."""
        df = pd.DataFrame(
            np.concatenate([self.data_state, self.data_response.reshape(-1, 1), self.data_target.reshape(-1, 1)], axis=1),
            columns=["state_{}".format(j) for j in range(self.data_state.shape[1])] + ["response", "target"]
        )
        df.to_csv(path, index=False)
        progress("Dataset saved at {}".format(path))


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
                 kappa=1, log=True, logdir="./log/value_estimation", *args, **kwargs):

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
        self.kappa = kappa
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

                self.targets = tf.reshape(tf.tile(self.values_ph, [1, self.num_atoms]), (-1, self.num_atoms))
                self.quantile_predictions = tf.reshape(self.value_prediction, (-1, self.num_atoms))
                self.errors = tf.subtract(tf.stop_gradient(self.targets), self.quantile_predictions)
                self.loss = quantile_huber_loss(self.errors, kappa=self.kappa, three_dims=False)

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

    def predict_quantiles(self, X, batch_size=10000):
        """Predict quantiles of the response time distribution of a set of states.

        Parameters
        ----------
        X: array-like, 2D
            The input data to predict.
        batch_size: int, default=10000
            The batch size to use when predicting. Influences memory and time costs.

        Returns
        -------
        Y_hat: np.array
            The predicted quantiles of the response time with shape [n_samples, n_quantiles].
        """
        assert self.quantiles, ("predict_quantiles can only be done for Quantile Regression"
                                "networks (initialize with quantiles=True).")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if batch_size is None:
            return self.session.run(self.quantile_predictions, feed_dict={self.states_ph: X})
        else:
            outputs = [
                self.session.run(
                    self.quantile_predictions,
                    feed_dict={self.states_ph: X[(i * batch_size):((i + 1)*batch_size), :]}
                )
                for i in range(int(np.ceil(len(X) / batch_size)))
            ]
            return np.concatenate(outputs, axis=0)

    def predict(self, X, batch_size=10000):
        """Predict the expected value / response time of set of states.

        Parameters
        ----------
        X: array-like, 2D
            The input data to predict.
        batch_size: int, default=10000
            The batch size to use when predicting. Influences memory and time costs.

        Returns
        -------
        Y_hat: np.array
            The predicted values / responses.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.quantiles:
            Y_hat = self.predict_quantiles(X, batch_size=batch_size)
            return Y_hat.mean(axis=1).reshape(-1, 1)
        elif batch_size is None:
            return self.session.run(self.value_prediction, feed_dict={self.states_ph: X})
        else:
            outputs = [
                self.session.run(
                    self.value_prediction,
                    feed_dict={self.states_ph: X[(i * batch_size):((i + 1)*batch_size), :]}
                )
                for i in range(int(np.ceil(len(X) / batch_size)))
            ]
            return np.concatenate(outputs, axis=0)

    def fit(self, env_cls, epochs=100, steps_per_epoch=100000, warmup_steps=50000,
            validation_freq=1, val_batch_size=10000, validation_data=None, permutations=False,
            env_params=None, metric="mae", eval_quants=False, verbose=True, save_freq=0, *args, **kwargs):
        """Fit the estimator on the environment.

        Parameters
        ----------
        env_cls: Python class
            The environment to train on.
        epochs: int, default=100
            The number of epochs to train.
        steps_per_epoch: int, default=100,000
            The number of steps to count as one epoch.
        validation_freq: int, default=1
            After how many epochs to evaluate on validation data. Set to 0 if you don't want
            to validate.
        val_batch_size: int, default=10,000
            The batch size to use in validation.
        validation_data: tuple(array-like, array-like)
            The data to use for validation.
        permutations: bool, default=False
            Whether to sample all state-permutations for training (True) or just sample
            according to distributions in the simulation (False).
        env_params: dict
            Parameters passed to env_cls upon initialization.
        metric: str, default='mae'
            Metric to use for evaluation. One of ['mae', 'mse', 'rmse' 'wasserstein'].
        eval_quants: bool, default=False
            Whether to evaluate on quantile values directly (True) or on expectation (False).
            Only relevant when self.quantiles=True.
        verbose: bool, default=True
            Whether to print progress updates.
        save_freq: int, default=0
            After how many epochs to save the model weights to the log directory.
            If save_freq=0 or self.log=False, does not save.
        *args, **kwargs: any
            Parameters passed to perform_tasks or gather_random_experiences.
        """
        def is_time(bool_, freq):
            if bool_ and (freq > 0):
                if epoch % freq == 0:
                    return True
            return False

        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        self.verbose = verbose

        if (validation_data is not None) and (validation_freq > 0):
            val_x, val_y = validation_data
            validate = True

        for epoch in range(epochs):
            # train
            if permutations:
                raise NotImplementedError("Training on all permutations is not implemented yet.")
            else:
                self.gather_random_experiences(env_cls, env_params=None, total_steps=steps_per_epoch,
                                               start_step=epoch*steps_per_epoch, *args, **kwargs)

            # evaluate
            if is_time(validate, validation_freq):
                loss = self.evaluate(val_x, val_y, metric=metric, raw_quantiles=eval_quants,
                                     batch_size=val_batch_size, verbose=False)
                progress("Epoch {}/{}. Val score: {}".format(epoch + 1, epochs, loss), verbose=verbose)

            # save weights
            if is_time(self.log, save_freq):
                self.save_weights()

        progress("Completed {} epochs of training.".format(epochs), verbose=verbose)

        if validate and (epochs % validation_freq != 0):  # if not validated after the last epoch
            loss = self.evaluate(val_x, val_y, metric=metric, raw_quantiles=eval_quants,
                                 batch_size=val_batch_size, verbose=False)
            progress("Final validation score: {}", verbose=verbose)

    def evaluate(self, X, Y, metric="mae", raw_quantiles=False, batch_size=10000, verbose=True):
        """Evaluate on provided data after training.

        Parameters
        ----------
        X, Y: array-like, 2D
            The input data and corresponding labels to evaluate on.
        batch_size: int, default=10000
            The batch size to use when predicting. Influences memory and time costs.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        if raw_quantiles:
            Y_hat = self.predict_quantiles(X, batch_size=batch_size)
        else:
            Y_hat = self.predict(X, batch_size=batch_size)

        if metric == "mae":
            loss = np.abs(Y - Y_hat).mean()
        elif metric == "mse":
            loss = np.square(Y - Y_hat).mean()
        elif metric == "rmse":
            loss = np.sqrt(np.square(Y - Y_hat).mean())
        elif metric == "wasserstein":
            assert raw_quantiles, "Wasserstein distance is only relevant when raw_quantiles=True"
            loss = np.mean([wasserstein_distance(Y[i, :], Y_hat[i, :]) for i in range(len(Y))])

        progress("Evaluation score ({}): {}".format(metric, loss), verbose=self.verbose)
        return loss

    def get_config(self):
        """Get the configuration of the estimator as a dictionary. Useful for reconstructing a
        trained estimator."""
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
