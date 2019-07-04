import os
import json
import pickle
import numpy as np
import pandas as pd

from spyro.utils import progress
from spyro.memory import ReplayBuffer
from spyro.policies import (
    EpsilonGreedyPolicy, GreedyPolicy,
    RandomPolicy, SoftmaxPolicy,
    FixedActionPolicy
)

from spyro.agents import (
    DQNAgent,
    A3CAgent,
    QuantileRegressionDQNAgent,
)
from spyro.value_estimation import NeuralValueEstimator


POLICY_MAP = {
    "EpsilonGreedyPolicy": EpsilonGreedyPolicy,
    "GreedyPolicy": GreedyPolicy,
    "RandomPolicy": RandomPolicy,
    "SoftmaxPolicy": SoftmaxPolicy,
    "FixedActionPolicy": FixedActionPolicy
}


MEMORY_MAP = {
    "ReplayBuffer": ReplayBuffer
}


AGENT_MAP = {
    "DQN_Agent": DQNAgent,
    "A3C_Agent": A3CAgent,
    "QR_DQN_Agent": QuantileRegressionDQNAgent,
    "NeuralEstimator": NeuralValueEstimator
}


LOG_COLUMNS = ["t", "time", "incident_type", "location", "priority",
               "object_function", "vehicle_type", "vehicle_id", "dispatch_time",
               "turnout_time", "travel_time", "on_scene_time", "response_time",
               "target", "station", "base_station_of_vehicle", "crew_type"]


def init_agent_from_config(config_path, force_no_log=False):
    """Initialize an agent based on a config file from a previous run.

    Parameters
    ----------
    config_path: str
        The path to the config JSON file.
    force_no_log: bool, default=False
        If true, sets log=False in agent's init. Useful to prevent the new agent
        from logging in a subdirectory of the original logdir.
    """
    # load config
    config = json.load(open(config_path, 'r'))

    # determine agent class
    agent_cls = AGENT_MAP[config["name"]]

    # set logging to False if specified
    if force_no_log:
        config["log"] = False
        del config["logdir"]

    # retrieve policy
    try:
        policy_config = config.pop("policy")
        has_policy = True
        policy_name = policy_config.pop("name")
        if policy_name == "EpsilonGreedyPolicy":
            del policy_config["epsilon"]
        policy = POLICY_MAP[policy_name](**policy_config)
    except KeyError:
        has_policy = False

    # retrieve memory
    try:
        memory_config = config.pop("memory")
        has_memory = True
        _ = memory_config.pop("name")
        memory = ReplayBuffer(**memory_config)
    except KeyError:
        has_memory = False

    # init agent
    if has_policy and has_memory:
        agent = agent_cls(policy, memory, **config)
    elif has_policy:
        agent = agent_cls(policy, **config)
    elif has_memory:
        agent = agent_cls(memory, **config)
    else:
        agent = agent_cls(**config)

    progress("Agent reconstructed from config.")
    return agent


def load_trained_agent(dirpath, env_cls, env_params=None, **kwargs):
    """Load a pre-trained agent with its weights.

    Parameters
    ----------
    dirpath: str
        The path to the directory in which the model parameters and agent config
        file are stored.
    env_cls: class or str
        The environment to train on, if a string is provided, it must be the name of a gym env.
    env_params: dict, default=None
        Key-value pairings to pass to env_cls if a class is provided.

    Returns
    -------
    agent: spyro.agents.*
        An agent object with loaded / pre-trained weights.
    """
    config_path = os.path.join(dirpath, "agent_config.json")
    agent = init_agent_from_config(config_path, **kwargs)
    agent.load_weights(os.path.join(dirpath, "model.ckpt"), env_cls=env_cls, env_params=env_params)
    progress("Agent's weights loaded.")
    return agent


def evaluate_saved_agent(dirpath, env_cls, n_episodes=100000, tmax=1000, policy=None,
                         env_params=None, save=True, evaluator=None):
    """Load a trained and saved agent from disk and evaluate it on a test environment.

    Parameters
    ----------
    dirpath: str
        The path to the directory in which the model parameters and agent config
        file are stored.
    env_cls: class or str
        The environment to train on, if a string is provided, it must be the name of a gym env.
    n_episodes: int, default=100000
        The number of episodes to use for evaluation.
    tmax: int, default=1000
        The maximum number of steps per episode.
    env_params: dict, default=None
        Key-value pairings to pass to env_cls if a class is provided.

    Returns
    -------
    results: any
        Output of agent.evaluate. Usually, this is a  dictionary with 'mean_episode_reward',
        'total_episode_reward', and 'episode_length' as keys and numpy arrays as values.
    test_log: pd.DataFrame
        The simulation log of all tested episodes.
    """
    agent = load_trained_agent(dirpath, env_cls, env_params=None, force_no_log=True)
    progress("Start test run on {} episodes.".format(n_episodes))
    results = agent.evaluate(env_cls, n_episodes=n_episodes, tmax=tmax, policy=policy, env_params=env_params)

    test_log = agent.env.get_test_log()

    if save:
        progress("Saving results to {}.".format(dirpath))
        pickle.dump(results, open(os.path.join(dirpath, "test_results_dict.pkl"), "wb"))
        test_log.to_csv(os.path.join(dirpath, "test_log.csv"), index=False)

    if evaluator is not None:
        progress("Extracting metrics using the evaluator")
        summary = evaluator.evaluate(test_log)

    else:
        summary = None

    progress("Evaluation completed.")
    return results, test_log, summary


def print_test_summary(summary):
    for name, data in summary.items():
        print("\n{}\n-----------------".format(name))
        print(data.T, end="\n\n")


def load_test_log(dirpath):
    """Load the test log from an agent's log directory.

    Parameters
    ----------
    dirpath: str
        The path to the agent's log directory.

    Returns
    -------
    log: pd.DataFrame
        The simulation log of the agent ran on the test episodes.
    """
    return pd.read_csv(os.path.join(dirpath, "test_log.csv"))


def construct_test_log_from_episodes(test_episodes, log_columns=LOG_COLUMNS):
    """Reconstruct a single simulation log from separate test episodes."""
    concat_log = np.concatenate(
        [np.append(d["log"], np.ones((len(d["log"]), 1)) * key, axis=1)
         for key, d in test_episodes.items()]
    )

    df = pd.DataFrame(concat_log, columns=log_columns + ["episode"])

    for col in ["t", "dispatch_time", "turnout_time", "travel_time",
                "on_scene_time", "response_time", "target", "episode"]:
        df[col] = df[col].astype(np.float)

    return df


def evaluate_quantile_estimator(model, table_path):
    """Evaluate a Quantile Regression Value Estimator on a table of 'true' quantiles.

    Parameters
    ----------
    model: NeuralValueEstimator
        The trained model to evaluate.
    table_path: str
        The path to the table to evaluate on.

    Returns
    -------
    wasserstein: float
        The mean Wasserstein distances between the learned and provided distributions.
    """
    quantile_table = pickle.load(open(table_path, "rb"))
    loss = model.evaluate_on_quantiles(quantile_table)
    return loss


def evaluate_saved_quantile_estimator(model_dir, table_path=None, quantile_table=None):
    """Load a trained and saved agent from disk and evaluate it on a test environment.

    Parameters
    ----------
    model_dir: str
        The path to the directory in which the model parameters and agent config
        file are stored.
    table_path: str, default=None
        The path to the table to evaluate on. Ignored if quantile_table is provided
        directly.
    quantile_table: dict, default=None
        The table to evaluate on. If None, a path must be provided to load the table.

    Returns
    -------
    wasserstein: float
        The mean Wasserstein distances between the learned and provided distributions.
    """
    assert (table_path is not None) or (quantile_table is not None), \
        "One of 'table_path' and 'quantile_table' must be provided."

    # load table if not provided directly
    if quantile_table is None:
        quantile_table = pickle.load(open(table_path, "rb"))

    # load model
    model = load_trained_agent(model_dir, None)

    # evaluate
    loss = model.evaluate_on_quantiles(quantile_table)
    return loss
