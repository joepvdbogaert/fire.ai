import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def quantile_range(num_quantiles=50):
    """Generate evenly spaced values in (0, 1) that can be used as quantile-positions.

    Parameters
    ----------
    num_quantiles: int, default=50
        The number of quantile-positions to generate.
    """
    return np.arange(0.5 * (1. / num_quantiles), 1, 1. / num_quantiles)


def get_table_quantiles(table, num_quantiles=51, inner_key="responses"):
    """Calculate quantiles over the simulated responses.

    Parameters
    ----------
    table: dict
        The table containing all possible states as keys and a dictionary of results
        as values.
    num_quantiles: int, default=51
        The number of quantiles to compute. Quantiles will be evenly spread over the
        interval (0, 1), e.g., when num_quantiles=5, will use q=[0.1, 0.3, 0.5, 0.7, 0.9].
    inner_key: str, default="responses"
        The key of the inner results dictionary that points to the array to compute
        quantiles over.
    """
    taus = quantile_range(num_quantiles)
    quantile_table = {}
    for key in table.keys():
        quantile_table[key] = np.quantile(table[key][inner_key], taus)
    return quantile_table


def get_reachable_states(table, state):
    """Filter a table to keep only the states are reachable from the current state.

    Parameters
    ----------
    table: dict
        The table containing all possible states. Must have states as keys.
    state: tuple, array
        The current state.
    """
    n = sum(state)
    return {key: value for key, value in table.items() if sum(key) == n}


def get_num_relocations(state1, state2):
    """Get the total number of relocations necessary to get from one state to the other.

    Parameters
    ----------
    state1, state2: tuple, array
        The two states to transition between.

    Returns
    -------
    num_relocations: int
        The number of relocations.
    """
    return np.maximum(np.array(state1) - np.array(state2), 0).sum()


def group_states_by_num_relocations(table, state, max_relocs=None):
    """Organze a table of reachable states by the number of relocations that is
    required to reach that state.

    Parameters
    ----------
    table: dict
        A table with states as keys. Results are only valid if the table
        contains only states that are reachable from the current state; i.e.,
        sums of the states must be the same.
    state: tuple, array
        The current state.
    max_relocs: int, default=None
        The maximum number of relocations to consider. Leaves any states requiring
        a higher number out of the results.

    Returns
    -------
    table_dict: dict
        A dictionary with integers representing the required number of relocations
        as keys and the part of the original table corresponding to this number
        as values.
    """
    nums = [get_num_relocations(key, state) for key in table.keys()]
    if max_relocs is None:
        max_relocs = max(nums)

    tables_dict = {n: {} for n in range(max_relocs + 1)}
    for i, (key, value) in enumerate(table.items()):
        try:
            tables_dict[nums[i]][key] = value
        except KeyError:
            pass

    return tables_dict


def get_state_expectations(table, inner_key=None, std=False):
    """Find the state with the highest or lowest expectation.

    Parameters
    ----------
    table: dict
        A table with states as keys.
    inner_key: str, default="responses"
        The key of the inner results dictionary that points to the array with values.
    """
    def f(arr):
        return np.mean(arr), np.std(arr)

    func = f if std else np.mean

    if inner_key is None:
        expected_table = {k: func(v) for k, v in table.items()}
    else:
        expected_table = {k: func(v[inner_key]) for k, v in table.items()}
    return expected_table


def get_state_with_best_expectation(table, inner_key=None, minimum=True):
    """Find the state with the highest or lowest expectation.

    Parameters
    ----------
    table: dict
        A table with states as keys.
    inner_key: str, default="responses"
        The key of the inner results dictionary that points to the array with values.
    minimum: bool, default=True
        If true, finds the state with the minimal expectation, rather than the
        maximum.

    Returns
    -------
    state: tuple
        The best state in expectation.
    values: any
        The results corresponding to the best state. Exact contents depend on
        inputs. Usually this will be a dictionary with multiple arrays, or
        otherwise an array.
    """
    arg_best = np.argmin if minimum else np.argmax
    expected_table = get_state_expectations(table, inner_key=inner_key)
    best_key = list(table.keys())[arg_best(list(expected_table.values()))]
    return best_key, table[best_key]


def get_best_states_by_num_relocations(table, state, to_quantiles=True, inner_key="responses",
                                       minimum=True, num_quantiles=50, max_relocs=None):
    """Find the best state that we can reach from the current state by different
    number of relocations.

    Parameters
    ----------
    table: dict
        A table with states as keys.
    state: tuple, array
        The current state.
    minimum: bool, default=True
        If true, finds the state with the minimal expectation, rather than the
        maximum.
    inner_key: str, default="responses"
        The key of the inner results dictionary that points to the array with values.
    """
    reachable_table = get_reachable_states(table, state)
    if to_quantiles:
        reachable_table = get_table_quantiles(reachable_table, num_quantiles=num_quantiles, inner_key=inner_key)
        inner_key = None

    tables_by_relocs = group_states_by_num_relocations(reachable_table, state, max_relocs=max_relocs)

    results = {}
    for k, ktable in tables_by_relocs.items():
        state, values = get_state_with_best_expectation(ktable, inner_key=inner_key, minimum=minimum)
        results[k] = {"state": state, "values": values, "num_relocs": k}
    return results


def augment_quantile_data(quantiles, values, num_points=1000):
    """Generate data points based on quantile positions and values, so that a dense distribution
    can be plotted based on them.

    Parameters
    ----------
    quantiles: array-like
        The quantile-positions.
    values: array-like
        The values of the quantile-positions in `quantiles`. Must be of the same length.
    num_points: int
        The number of data points to generate. More points results in a more dense distribution
        but may be more computationally expensive in subsequent tasks (e.g., plotting).

    Returns
    -------
    data: array-like
        A generated array with data points, simulated by interpolating between the provided
        quantiles.
    """
    return np.interp(np.arange(0, np.max(quantiles), 1 / num_points), quantiles, values)


def quantile_kde_plot(y, *args, **kwargs):
    """plot a kernel density plot based on quantile values.

    Parameters
    ----------
    y: array-like
        The values of the quantiles. Entries are assumed to refer to the values of equally
        spaced quantiles, e.g., [0.1, 0.2, 0.3, ...], and not [0.1, 0.2, 0.35, 0.40, ...].
        The positions of the quantiles are then derived from the length of the array.
    """
    qs = quantile_range(len(y))
    y_aug = augment_quantile_data(qs, y)
    return sns.kdeplot(y_aug, *args, **kwargs)


def plot_best_reachable_states(table, state, quantile_input=False, to_quantiles=False,
                               inner_key="responses", minimum=True, num_quantiles=50,
                               max_relocs=None, value_name="response times", clip=None):
    """Find the best state that we can reach from the current state by different
    number of relocations.

    Parameters
    ----------
    table: dict
        A table with states as keys.
    state: tuple, array
        The current state.
    minimum: bool, default=True
        If true, finds the state with the minimal expectation, rather than the
        maximum.
    inner_key: str, default="responses"
        The key of the inner results dictionary that points to the array with values.
    max_relocs: int, default=None
        The maximum number relocations to consider.
    """
    if quantile_input:
        to_quantiles = False
        inner_key = None

    best_states_per_reloc = get_best_states_by_num_relocations(
        table, state, to_quantiles=to_quantiles, inner_key=inner_key,
        minimum=minimum, num_quantiles=num_quantiles, max_relocs=max_relocs
    )
    if to_quantiles or quantile_input:
        df = pd.concat([pd.DataFrame({"relocations": k, value_name: v["values"]})
                        for k, v in best_states_per_reloc.items()],
                       axis=0)
    else:
        df = pd.concat([pd.DataFrame({"relocations": k, value_name: v["values"][inner_key]})
                    for k, v in best_states_per_reloc.items()],
                   axis=0)

    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, light=.7)
    g = sns.FacetGrid(df, row="relocations", hue="relocations", aspect=6, height=1.5, palette=pal)

    # Draw the densities in a few steps
    if quantile_input or to_quantiles:
        g.map(quantile_kde_plot, value_name, clip_on=False, shade=True, alpha=1., lw=1.5, bw=.2, gridsize=50, clip=clip)
        g.map(quantile_kde_plot, value_name, clip_on=False, color="w", lw=1.5, bw=.2, gridsize=50, clip=clip)
    else:
        g.map(sns.kdeplot, value_name, clip_on=False, shade=True, alpha=1., lw=1.5, bw=.2, gridsize=50, clip=clip)
        g.map(sns.kdeplot, value_name, clip_on=False, color="w", lw=1.5, bw=.2, gridsize=50, clip=clip)
    
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", size=16, transform=ax.transAxes)

    g.map(label, value_name)

    # Set the subplots to overlap
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=1.02, hspace=-.45)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    g.fig.suptitle("Response time improvement by number of relocations", weight="bold", size=18)
    return g.fig


def group_states_by_vehicle_count(table):
    """Organize all states by their total vehicle count.

    Parameter
    ---------
    table: dict
        A table with states as keys.
    """
    nums = [sum(key) for key in table.keys()]

    tables_dict = {n: {} for n in range(max(nums) + 1)}
    for i, (key, value) in enumerate(table.items()):
        tables_dict[nums[i]][key] = value

    return tables_dict


def get_best_state_by_vehicle_count(table, inner_key="responses", minimum=True):
    """Find the best state per available number of vehicles.

    Parameters
    ----------
    inner_key: str, default="responses"
        The key of the inner results dictionary that points to the array with values.
    minimum: bool, default=True
        If true, finds the state with the minimal expectation, rather than the
        maximum.
    """
    tables_by_count = group_states_by_vehicle_count(table)
    best_states = {}
    for k, ktable in tables_by_count.items():
        if ktable:  # checks if dict is empty (usually the case for all-zero state)
            best_state, value = get_state_with_best_expectation(ktable, inner_key=inner_key)
            best_states[k] = {"state": best_state, "data": value}

    return best_states
