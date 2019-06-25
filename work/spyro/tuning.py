import numpy as np
import pandas as pd
import multiprocessing
import pickle
import tensorflow as tf

try:
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("Failed to import GPyOpt, either install it or refrain from using Bayesian Optimization")

from spyro.evaluation import evaluate_quantile_estimator


TUNING_OUTPUT_DEFAULT = '../results/tuning_output.pkl'


def print_eval_result(params, score, verbose=True):
    if verbose:
        print("--\nScore: {}\nParams: {}".format(score, params))


def tune_agent(agent_cls, env_cls, params, *args, env_params=None, fit_params=None,
               max_iter=250, max_time=None, init_size=8,
               model_type='GP', acquisition_type='LCB', acquisition_weight=0.2,
               eps=1e-6, batch_method='local_penalization', batch_size=1, maximize=False,
               eval_func=evaluate_quantile_estimator, eval_params=None, verbose=True,
               save=True, write_to=TUNING_OUTPUT_DEFAULT, **kwargs):
    """Automatically configures hyperparameters of ML algorithms using Bayesian Optimization.
    Suitable for reasonably small sets of params.

    Parameters
    ----------
    agent_cls: Python class
        The predictors class.
    params: dict
        Dictionary with three keys: {"name": <str>, "type": <'discrete'/'continuous'>,
        "domain": <list/tuple>}. The continuous variables must first be specified, followed
        by the discrete ones.
    max_iter: int
        The maximum number of iterations / evaluations. Note that this excludes initial
        exploration. Also, it might converge before reaching max_iter.
    max_time: int
        The Maximum time to be used in optimization in seconds.
    init_size: int, default=8
        The number of iterations in the initial random exploration phase.
    model_type: str
        The model used for optimization. Defaults to Gaussian Process ('GP').
            -'GP', standard Gaussian process.
            -'GP_MCMC',  Gaussian process with prior in the hyper-parameters.
            -'sparseGP', sparse Gaussian process.
            -'warperdGP', warped Gaussian process.
            -'InputWarpedGP', input warped Gaussian process.
            -'RF', random forest (scikit-learn).
    acquisition_type: str
        Function used to determine the next parameter settings to evaluate.
            -'EI', expected improvement.
            -'EI_MCMC', integrated expected improvement (requires GP_MCMC model type).
            -'MPI', maximum probability of improvement.
            -'MPI_MCMC', maximum probability of improvement (requires GP_MCMC model type).
            -'LCB', GP-Lower confidence bound.
            -'LCB_MCMC', integrated GP-Lower confidence bound (requires GP_MCMC model type).
    acquisition_weight: int
        Exploration vs Exploitation parameter.
    eps: float
        The minimum distance between consecutive candidates x.
    batch_method: str
        Determines the way the objective is evaluated if batch_size > 1 (all equivalent if batch_size=1).
            -'sequential', sequential evaluations.
            -'random': synchronous batch that selects the first element as in a sequential
            policy and the rest randomly.
            -'local_penalization': batch method proposed in (Gonzalez et al. 2016).
            -'thompson_sampling': batch method using Thompson sampling.
    batch_size: int
        The number of parallel optimizations to run. If None, uses batch_size = number of cores.
    verbose: bool
        Whether or not progress messages will be printed (prints if True).
    save: bool
        If set to true, will write tuning results to a pickle file at the `write_to` path.
    write_to: str
        If save=True, this defines the filepath where results are stored.
    *args, **kwargs: any
        Parameters passed to the agent upon initialization.

    Returns
    -------
    best_params, best_score: dict, float
        the best parameters as a dictionary like {<name> -> <best value>} and the corresponding
        evaluation score.
    """
    for arg in [eval_params, fit_params]:
        if arg is None:
            arg = {}

    print("Using Bayesian Optimization to tune {} in {} iterations and {} seconds."
          .format(agent_cls, max_iter, max_time))

    use_log2_scale, use_log10_scale = [], []
    for p in params:
        try:
            if p["log"] == 2:
                use_log2_scale.append(p["name"])
            elif p["log"] == 10:
                use_log10_scale.append(p["name"])
        except KeyError:
            pass

    print("Using log2-scale for parameters {} and log10-scale for {}".format(use_log2_scale, use_log10_scale))

    def create_mapping(p_arr):
        """Changes the 2d np.array from GPyOpt to a dictionary.

        Takes care of translating log-scaled parameters to their actual value
        assuming a log2 scale.

        Parameters
        ----------
        p_arr: 2d np.array
            array with parameter values in the same order as `params`.

        Returns
        -------
        mapping: dict
            Parameter mapping like {"name" -> value}.
        """
        mapping = dict()
        for i in range(len(params)):
            value = int(p_arr[0, i]) if params[i]["type"] == "discrete" else p_arr[0, i]
            if params[i]["name"] in use_log2_scale:
                value = 2**value
            if params[i]["name"] in use_log10_scale:
                value = 10**value
            mapping[params[i]["name"]] = value

        return mapping

    def f(parameter_array):
        """The objective function to maximize."""
        param_dict = create_mapping(parameter_array)
        print("Evaluating params: {}".format(param_dict))
        tf.reset_default_graph()
        agent = agent_cls(*args, **param_dict, **kwargs)
        agent.fit(env_cls, env_params=env_params, **fit_params)
        score = eval_func(agent, **eval_params)
        results["scores"].append(score)
        results["params"].append(param_dict)
        print_eval_result(param_dict, score, verbose=verbose)
        # only return score to optimizer
        if maximize:
            return -score
        else:
            return score

    # scores are added to these lists in the optimization function f
    results = {"params": [], "scores": []}

    # run optimization in parallel
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    # set batch_size equal to num_cores if no batch_size is provided
    if not batch_size:
        batch_size = num_cores

    if verbose:
        print("Running in batches of {} on {} cores using {}."
              .format(batch_size, num_cores, batch_method))
        print("Begin training.")

    # define optimization problem
    opt = BayesianOptimization(
        f, domain=params, model_type=model_type, acquisition_type=acquisition_type,
        normalize_Y=False, acquisition_weight=acquisition_weight, num_cores=1,
        batch_size=batch_size, initial_design_type='random', initial_design_numdata=init_size
    )

    # run optimization
    try:
        opt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping optimization and returning results.")

    # report results
    if save:
        pickle.dump(results, open(write_to, "wb"))

    best = np.argmax(results["scores"])
    best_params, best_score = results["params"][best], results["scores"][best]

    return best_params, best_score
