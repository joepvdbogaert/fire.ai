import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_sns_params(font_scale=1.2, **kwargs):
    sns.set(font_scale=font_scale, **kwargs)


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


def ridge_plot_response_distributions(data, row_col, value_col, quantile_input=False, clip=None,
                                      title="Response time improvement by number of relocations"):
    """Make a Ridge Plot of response time distributions for different situations.

    Parameters
    ----------
    data: pd.DataFrame
        The data to plot.
    row_col, value_col: str
        The columns to split by and to plot the distributions of respectively.
    """
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, light=.7)
    g = sns.FacetGrid(data, row=row_col, hue=row_col, aspect=6, height=1.5, palette=pal)

    # Draw the densities in a few steps
    if quantile_input:
        g.map(quantile_kde_plot, value_col, clip_on=False, shade=True, alpha=1., lw=1.5, bw=.2, gridsize=50, clip=clip)
        g.map(quantile_kde_plot, value_col, clip_on=False, color="w", lw=1.5, bw=.2, gridsize=50, clip=clip)
    else:
        g.map(sns.kdeplot, value_col, clip_on=False, shade=True, alpha=1., lw=1.5, bw=.2, gridsize=50, clip=clip)
        g.map(sns.kdeplot, value_col, clip_on=False, color="w", lw=1.5, bw=.2, gridsize=50, clip=clip)
    
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", size=16, transform=ax.transAxes)

    g.map(label, value_col)

    # Set the subplots to overlap
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=1.02, hspace=-.45)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    g.fig.suptitle(title, weight="bold", size=18)
    return g.fig


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

    return ridge_plot_response_distributions(df, row_col="relocations", value_col=value_name,
                                             quantile_input=to_quantiles or quantile_input,
                                             title="Response time improvement by number of relocations")


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
    table: dict
        A table with states as keys.
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


def get_top_n_states(table, n=10, inner_key=None, minimum=True):
    """Retrieve the data for the top n best states.

    Parameters
    ----------
    table: dict
        A table with states as keys.
    """
    means = get_state_expectations(table, inner_key=inner_key, std=False)
    df = pd.DataFrame({"state": list(means.keys()), "mean": list(means.values())})
    df.sort_values("mean", ascending=True if minimum else False, inplace=True)
    return {k: table[k] for k in df["state"].iloc[0:n]}


def get_top_n_states_by_vehicle_count(table, n=10, inner_key="responses", minimum=True):
    """Find the best n states per available number of vehicles.

    Parameters
    ----------
    table: dict
        A table with states as keys.
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
            top_n_dict = get_top_n_states(ktable, inner_key=inner_key, minimum=minimum)
            best_states[k] = top_n_dict

    return best_states


def table_to_df(table, inner_key=None, add_rank=False, state_as_index=True):
    """Convert a dictionary-structured table into a Pandas DataFrame.

    Parameters
    ----------
    table: dict
        The table to convert.
    inner_key: str, default=None
        The key of the inner results dictionary that points to the array with values.

    Returns
    -------
    df: pd.DataFrame
        The resulting data frame.
    """
    ix = pd.Index([tuple(key) for key in table.keys()], dtype=tuple)
    if inner_key is None:
        df = pd.DataFrame([v for v in table.values()], index=ix)
    else:
        df = pd.DataFrame([v[inner_key] for v in table.values()], index=ix)

    if add_rank:
        df["rank"] = np.arange(1, len(df) + 1)

    df.index.name = "state"
    if not state_as_index:
        df = df.reset_index(drop=False)

    return df


def nested_table_to_df(table_dict, inner_key=None, outer_name="number of vehicles", add_rank=True):
    """Convert a nested table-dictionary to a DataFrame.

    Parameters
    ----------
    table_dict: dict
        The nested dictionary.
    inner_key: str, default="responses"
        The key of the inner results dictionary that points to the array with values.
    outer_name: str, default='number of vehicles'
        The new name of the column corresponding to the outer-most keys in the dictionary.

    Returns
    -------
    df: pd.DataFrame
        The results as a DataFrame with the states (2nd level keys) as index, a column referring
        to the first level keys and columns corresponding to the values.
    """
    outer_keys = list(table_dict.keys())
    dfs = [table_to_df(tab, inner_key=inner_key, add_rank=add_rank) for tab in table_dict.values()]

    for i in range(len(outer_keys)):
        dfs[i][outer_name] = outer_keys[i]

    df = pd.concat(dfs, axis=0)
    return df


def get_station_occurences_from_grouped_table(table_dict, inner_key=None,
                                              outer_name="total vehicles",
                                              station_names=None, to_long=True):
    """Extract the number of times a station is occupied in the states in a grouped table,
    e.g., in a table of top 10 states grouped by the total number of vehicles.

    Parameters
    ----------
    table_dict: dict
        The nested dictionary of tables.
    inner_key: any, default=None
        Key to the values if the tables are nested as well.
    outer_name: str, default="total vehicles"
        How to call the keys of the outer-most dictionary.
    station_names: iterable, default=None
        Names of the stations in the order of the states. If None, dummy names are provided.
    to_long: bool, default=True
        Whether to create a column 'station' (True) or to keep each station as a column
        (False).

    Returns
    -------
    df: pd.DataFrame
        The resulting DataFrame.
    """

    df_table = nested_table_to_df(table_dict, outer_name=outer_name, add_rank=True)
    states = np.array([list(v) for v in df_table.index.values])

    if station_names is None:
        station_names = ["station_{}".format(i) for i in range(len(states[0]))]

    df_states = pd.DataFrame(states, columns=station_names)

    df = df_states
    df[outer_name] = df_table[outer_name].values
    df["rank"] = df_table["rank"].values

    if to_long:
        df = pd.melt(df, [outer_name, 'rank']).rename(columns={'variable': 'station', 'value': 'vehicles'})

    return df


def plot_heatmap_of_vehicle_positions(data, x="station", y="total vehicles", values="vehicles",
                                      ax=None, title="Station occupancy among best 10 states",
                                      *args, **kwargs):
    """Plot a heatmap of how often stations are occupied in the states in the data.

    Parameters
    ----------
    data: pd.DataFrame
        The data to plot. Designed for the output of
        `get_station_occurences_from_grouped_table`.
    x, y, values: str
        Columns to use at the x and y axes and as values in the heatmap.
    ax: matplotlib.Axis, default=None
        Axis to plot on. If None, creates new.
    *args, **kwargs: any
        Parameters of `sns.heatmap`.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        The heatmap.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    pivoted = pd.pivot_table(data, index=x, columns=y, values=values)
    ax = sns.heatmap(pivoted, ax=ax, cmap="YlGnBu", *args, **kwargs)
    ax.set_title(title, weight="bold", size=20, pad=20)
    return fig


def heatmap(x, y, values, data=None, *args, **kwargs):
    """Simple heatmap function that pivots data and plots a heatmap.

    This function is used in `plot_faceted_heatmap_of_vehicle_positions`.

    Parameters
    ----------
    x, y, values: str
        Columns to use at the x and y axes and as values in the heatmap.
    data: pd.DataFrame
        The data to plot.
    *args, **kwargs: any
        Parameters of `sns.heatmap`.
    """
    pivoted = pd.pivot_table(data, index=y, columns=x, values=values)
    ax = sns.heatmap(pivoted, *args, **kwargs)
    return ax


def plot_faceted_heatmap_of_vehicle_positions(data, x="rank", y="station", values="vehicles",
        multiples="total vehicles", title="Station occupancy in best states by total number of vehicles",
        *args, **kwargs):
    """Plot small multiple heatmaps.

    Parameters
    ----------
    data: pd.DataFrame
        The data to plot.
    x, y, values, multiples: str
        Columns to use at the x and y axes and as values in the heatmap and to group on for the
        small plots.
    *args, **kwargs: any
        Parameters passed to `sns.heatmap`.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        The resulting figure.
    """
    set_sns_params()
    g = sns.FacetGrid(data, col=multiples, height=4, aspect=1, col_wrap=4, sharex=False)
    g.map_dataframe(heatmap, x=x, y=y, values=values, data=data, yticklabels=True, xticklabels=True, *args, **kwargs)
    for ax in g.axes:
        ax.set_xlabel("state rank (1=best)")
        ax.set_ylabel(y)

    g.fig.tight_layout()
    g.fig.suptitle(title, weight="bold", size=20)
    g.fig.subplots_adjust(top=0.95)
    return g.fig


def plot_faceted_barplot_of_vehicle_positions(data, y="station", x="vehicles",
                                              multiples="total vehicles", *args, **kwargs):
    """Plot small multiple heatmaps.

    Parameters
    ----------
    data: pd.DataFrame
        The data to plot.
    x, y, values, multiples: str
        Columns to use at the x and y axes and as values in the heatmap and to group on for the
        small plots.
    *args, **kwargs: any
        Parameters passed to `sns.heatmap`.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        The resulting figure.
    """
    sns.set(style='white')
    # aggregate if necessary
    data = data.groupby([y, multiples])[x].sum().reset_index()
    # plot
    g = sns.FacetGrid(data, col=multiples, col_wrap=5, sharex=True)
    g.map_dataframe(sns.barplot, x=x, y=y, orient="h", *args, **kwargs)
    for ax in g.axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    g.fig.tight_layout()
    return g.fig
