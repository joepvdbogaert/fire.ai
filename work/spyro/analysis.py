import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from spyro.utils import progress
from spyro.value_estimation import STATION_NAMES

try:
    from fdsim.helpers import lonlat_to_xy
except:
    progress("fdsim not installed, some functions might not work.")
try:
    import geopandas as gpd
except:
    progress('geopandas not installed, some functions might not work.')


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

    tables_dict = {n: {} for n in range(1, max_relocs + 1)}
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
                                      max_y=16, *args, **kwargs):
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
    title: str, default='Station occupancy among best 10 states'
        The title of the plot.
    max_y: int, default=16
        The maximum number of vehicles to plot.
    *args, **kwargs: any
        Parameters of `sns.heatmap`.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        The heatmap.
    """
    sns.set(font_scale=1.6)
    fig, ax = plt.subplots(figsize=(15, 10))
    pivoted = pd.pivot_table(data, index=x, columns=y, values=values)

    if max_y is not None:
        pivoted = pivoted.loc[:, pivoted.columns <= max_y]

    ax = sns.heatmap(pivoted, ax=ax, cmap="YlGnBu", *args, **kwargs)
    ax.set_title(title, weight="bold", size=20, pad=20)

    fig.tight_layout()
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


def load_tables(*paths):
    """Read multiple tables from disk and return them in a list.

    Parameters
    ----------
    paths: str
        The paths to table files to load."""
    return [pickle.load(open(p, "rb")) for p in paths]


def append_arrays_in_dicts(*dikts, keys=["responses", "targets"]):
    """Create a dictionary with arrays appended from multiple dictionaries.

    Parameters
    ----------
    dikts: dict
        The dictionaries to merged.
    keys: list, str, default=['responses', 'targets']
        The keys in the dictionaries that point to arrays that should be appended.
    """
    return {k: np.concatenate([d[k] for d in dikts]) for k in keys}


def merge_tables(*tables, to_quantiles=True, key="responses", num_quantiles=51, save_path=None):
    """Merge multiple tables into one big one.

    Parameters
    ----------
    tables: dict
        The tables to merge. Should all have the same set of keys / states.
    to_quantiles: bool, default=True
        Whether to calculate quantiles over the obtained values rather than keep the raw ones.
    key: str, default="responses"
        If to_quantiles=True, the key is the key in the inner dictionary that points to the array
        over which to compute quantiles. If to_quantiles=False, key is the (list of) keys for which
        arrays of different tables should be appended.
    num_quantiles: int, default=51
        The number of quantiles to compute when to_quantiles=True.
    save_path: str, default=None
        The path to save the resulting table. If None, does not save.

    Returns
    -------
    merged_table: dict
        The merged table.
    """
    assert len(tables) > 1, "Must provide more than one table"
    assert set(tables[0].keys()) == set(tables[0].keys()), "Keys are not the same for all tables"

    merged = tables[0]
    for i, state in enumerate(tables[0].keys()):
        progress("Merging results for state {} / {}.".format(i + 1, len(merged)), same_line=True, newline_end=(i + 1 == len(merged)))
        merged[state] = append_arrays_in_dicts(*[t[state] for t in tables], keys=key if isinstance(key, list) else [key])

    if to_quantiles:
        progress("Obtaining quantiles for '{}''".format(key))
        merged = get_table_quantiles(merged, num_quantiles=num_quantiles, inner_key=key)

    if save_path is not None:
        pickle.dump(merged, open(save_path, "wb"))
        progress("Merged table save at {}".format(save_path))

    return merged


def calc_mean_quantiles(*tables):
    """Calculate the means of multiple quantile estimates.

    Assumes the quantiles have been estimated from samples
    of the same size.

    Parameters
    ----------
    *tables: dicts
        The tables to merge. Must have states as keys and
        numpy arrays of quantiles as values.

    Returns
    -------
    quantile_table: dict
        States as keys and numpy array of merged quantile estimates as values.
    """
    n = len(tables)
    matrices = [np.array(list(table.values())) for table in tables]
    mean = np.zeros_like(matrices[0])

    for i in range(n):
        mean += matrices[i] / n
    keys = list(tables[0].keys())
    result = {keys[i]: mean[i, :] for i in range(len(keys))}
    return result


def merge_tables_in_chunks(*paths):
    """Load tables in chunks, calculate quantiles, and get the mean
    of the quantiles estimates of different chunks.

    For correct estimation of the quantiles, it is assumed that each
    file has the same sample size.

    Parameters
    ----------
    *paths: str
        The paths to the tables that should be combined.

    Returns
    -------
    mean_table: dict
        The quantile estimates for each state.
    """
    assert len(paths) % 2 == 0, "Must be an even number of paths"
    num_chunks = len(paths) / 2
    path_chunks = [[paths[i], paths[i + 1]] for i in range(len(paths)) if i % 2 == 0]
    qtables = []
    for chunk_paths in path_chunks:
        chunk_tables = load_tables(*chunk_paths)
        qtables.append(merge_tables(*chunk_tables, to_quantiles=True, save_path=None))
    
    mean_table = calc_mean_quantiles(*qtables)
    return mean_table


def get_station_coords(path, station_col="kazerne"):
    """Obtain x, y coordinates for each station.

    Parameters
    ----------
    path: str
        The path to the station location Excel file.
    station_col: str, default='kazerne'
        The column in the station data that gives the station names or IDs.

    Returns
    -------
    coord_dict: dict
        The coordinates like {'STATION_NAME' -> (x, y)}.
    """
    station_locations = pd.read_excel(path, sep=";", decimal=".")
    station_locations[["x", "y"]] = station_locations[["lon", "lat"]].apply(
                lambda x: lonlat_to_xy(x[0], x[1]), axis=1).apply(pd.Series)

    coord_dict = {}
    for i, station in enumerate(station_locations[station_col]):
        coord_dict[station.upper()] = tuple(station_locations[['x', 'y']].iloc[i])

    return coord_dict


def plot_state_on_map(state, geo_df, coords, prev_state=None, station_names=STATION_NAMES,
                      annotate=True, figsize=None, ax=None, yshift=450,
                      shift_stations=["HENDRIK", "ANTON"]):
    """Plot a state on the map, showing which stations are occupied or empty.

    Parameters
    ----------
    state: tuple
        The station occupancy.
    geo_df: geopandas.DataFrame
        The polygons of the underlying map.
    coords: dict
        Coordinates of the stations in a dictionary like {STATION -> (x, y)}.
    station_names: list-like, default=spyro.value_estimation.STATION_NAMES
        The names of the stations corresponding to `state`.
    annotate: bool, default=True
        Whether to print station names on the map.
    figsize: tuple, default=None
        The figure size.
    ax: matplotlib.pyplot.Axes, default=None
        The Axis to plot on. If None, creates new one.

    Returns
    -------
    ax: Axis
        The plotted map.
    """
    def map_color(old, new, palette):
        if old == new:
            return palette[0]
        elif new == 0:
            return palette[1]
        else:
            return palette[2]

    def annotate_station(j, txt, condition=True, size=9, grey=False):
        txt_color = 'grey' if grey else 'black'
        if condition:
            if station_names[j] in shift_stations:
                ax.annotate(txt, (x[j], y[j]), xytext=(x[j]+100, y[j]+100 + yshift), size=9, color=txt_color)
            else:
                ax.annotate(txt, (x[j], y[j]), xytext=(x[j]+100, y[j]+100), size=9, color=txt_color)

    if prev_state is None:
        prev_state = state

    ax = geo_df.plot(figsize=figsize, alpha=0.3, ax=ax)

    x = [coords[s][0] for s in station_names]
    y = [coords[s][1] for s in station_names]
    sizes = [100 if v > 0 else 15 for v in state]

    colors = sns.color_palette('tab10', 3)
    c = [map_color(prev_state[v], state[v], colors) for v in range(len(state))]
    ax.scatter(x, y, c=c, s=sizes)

    # annotate stations with names
    for i, txt in enumerate(station_names):
        if annotate == 'changed':
            annotate_station(i, txt, condition=(state[i] != prev_state[i]), grey=False)
        elif annotate:
            annotate_station(i, txt, grey=state[i] == 0)

    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def map_plot_facet_wrapper(states, geo_df=None, coords=None, ax=None, annotate=False, *args, **kwargs):
    """Wrap `plot_state_on_map` with a signature suitable for sns.FacetGrid."""
    ax = plt.gca()
    _ = plot_state_on_map(states.values[0], geo_df, coords, ax=ax, annotate=annotate)


def plot_best_states_on_map(table, geopath="../../Data/geoData/vakken_dag_ts.geojson",
                            stationpath="../../Data/kazernepositie en voertuigen.xlsx",
                            station_names=STATION_NAMES, min_count=0, max_count=16,
                            inner_key=None, minimum=True, top=0.92):
    """Plot the best configuration of vehicles on a map for each possible vehicle count.

    Parameters
    ----------
    table: dict
        Table of state-values like {'state' -> [value1, value2, ..., ...]}. Can be quantiles.
    geopath: str, default='../../Data/geoData/vakken_dag_ts.geojson'
        The path to the underlying map polygons.
    stationpath: str, default="../../Data/kazernepositie en voertuigen.xlsx"
        Path to the station coordinate data.
    station_names: list-like, default=spyro.value_estimation.STATION_NAMES
        The names of the stations corresponding to `state`.
    min_count, max_count: int, default=0, 16
        The min and max number of total vehicles to plot the best state for.
    inner_key, minimum: any
        Passed to `get_best_state_by_vehicle_count`.

    Returns
    -------
    fig: matplotlib.Figure
        The Faceted plot of configurations.
    """
    # load data
    stationcoords = get_station_coords(stationpath)
    geodf = gpd.read_file(geopath)

    # get best states and filter
    best_by_count = get_best_state_by_vehicle_count(table, inner_key=inner_key, minimum=minimum)
    if min_count is not None:
        best_by_count = {k: v for k, v in best_by_count.items() if k >= min_count}
    if max_count is not None:
        best_by_count = {k: v for k, v in best_by_count.items() if k <= max_count}

    df = pd.DataFrame.from_dict(best_by_count, orient='index')
    df.index.name = "vehicles"
    df = df.reset_index(drop=False)

    # plot
    sns.set(style="white")
    g = sns.FacetGrid(data=df, col="vehicles", col_wrap=4)
    g.map(map_plot_facet_wrapper, "state", geo_df=geodf, coords=stationcoords)
    g.fig.suptitle("Best vehicle configuration per number of vehicles", weight="bold", size=18)
    for ax in g.axes:
        ax.set_xlabel('')
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=top, wspace=0.01, hspace=0.01)
    return g.fig
