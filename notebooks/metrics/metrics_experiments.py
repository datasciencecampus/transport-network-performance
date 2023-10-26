# %% [markdown] noqa: D212, D400, D415
"""
# Metrics Experiments

A notebook developed for exerimenting with different approaches to calculating
transport performance metrics.

## Preamble
Call in script wide imports and the configuration information.
"""

# %%
import folium
import geopandas as gpd
import os
import pandas as pd
import pathlib
import pickle

from folium.map import Icon
from haversine import haversine_vector
from pyprojroot import here
from tqdm import tqdm
from typing import Union

from transport_performance.utils.defence import (
    _check_parent_dir_exists,
)

# %%
# name of area and source of metrics inputs
area = "london"

if area == "london":
    metrics_input_dir = here(
        "data/processed/analyse_network/london_e2e/od_matrix_london"
    )
    RP_PATH = here("data/processed/analyse_network/london_e2e/london_rp.pkl")
    TP_OUTPUT = here("outputs/e2e_london/london_transport_performance.html")
elif area == "newport":
    metrics_input_dir = here(
        "data/processed/analyse_network/newport_e2e/experiments/newport"
    )
    RP_PATH = here("data/processed/analyse_network/newport_e2e/rasterpop.pkl")
    TP_OUTPUT = here("outputs/e2e_newport/newport_transport_performance.html")

# set the maximum distance and time threshold for calculating performance
MAX_DISTANCE = 11.25
MAX_TIME = 45

# %% [markdown] noqa: D212, D400, D415
"""
## Preprocessing Inputs
This section looks to preprocess the inputs needed of a `metrics` module. It
takes an OD `r5py` result (in this case the Newport bus example), and converts
it to a collection of parquet files (as per the output `analyse_network`).
These files can then be used to experiment with different python modules when
calculating the transport performance.

> Note: this section only needs to be run as a 'one-off' FOR NEWPORT ONLY.
"""

# %%
# outputs from the analyse_network stage, to use during the experiment
ANALYSE_NETWORK_OUTPUTS = here(
    "data/processed/analyse_network/newport_e2e/travel_times.pkl"
)
BATCH_BY_COL = "from_id"

# create a single parquet file path
tt_parquet_path = str(ANALYSE_NETWORK_OUTPUTS).replace(".pkl", ".parquet")

# %%
# read in the travel times
travel_times = pd.read_pickle(ANALYSE_NETWORK_OUTPUTS)
travel_times.head()

# %%
# convert travel times to a single parquet
travel_times.to_parquet(tt_parquet_path)

# %%
# batch travel_times into individual parquet files
ids = travel_times[BATCH_BY_COL].unique()

# create the parent dir if it doesnt exist - dummy needed to create parent dir
_check_parent_dir_exists(
    os.path.join(metrics_input_dir, "dummy.txt"),
    "metrics_input_dir",
    create=True,
)

for id in tqdm(ids, total=len(ids)):

    # get a batch
    batch_df = travel_times[travel_times[BATCH_BY_COL] == id]

    # create the output filepath and check if parent exists in first pass
    batch_filepath = os.path.join(metrics_input_dir, f"newport_id{id}.parquet")

    # create batched parquet file
    batch_df.to_parquet(batch_filepath)

# %% [markdown] noqa: D212, D400, D415
"""
## Transport Performance - `pandas`
Use `pandas` methods to calculate transport performance
"""

# %%
# read in rasterpop class for area of interest and unpack attributes
with open(RP_PATH, "rb") as f:
    rp = pickle.load(f)
    pop_gdf = rp.pop_gdf
    centroid_gdf = rp.centroid_gdf

# %%


def _transport_performance_pandas(
    filepath_or_dirpath: Union[str, pathlib.Path],
    centroids: gpd.GeoDataFrame,
    populations: gpd.GeoDataFrame,
    travel_time_threshold: int = 45,
    distance_threshold: float = 11.25,
) -> pd.DataFrame:
    """Calculate transport performance using `pandas`.

    Parameters
    ----------
    filepath_or_dirpath : Union[str, pathlib.Path]
        Input filepath or directory to parquet file(s).
    centroids : gpd.GeoDataFrame
        A dataframe containing the centroid of each cell ID. "centroid" must
        be in "EPSG:4326"/WGS84.
    populations : gpd.GeoDataFrame
        A dataframe containing the "population" of each cell ID.
    travel_time_threshold : int, optional
        Maximum travel time, in minutes, by default 45.
    distance_threshold : float, optional
        Maximum distance from the destination, in kilometers, by default 11.25.

    Returns
    -------
    pd.DataFrame
        The transport performance for each "to_id" cell, and the corresponding
        accesible and proximity populations.

    """
    # convert centroid shapley object to tuple
    centroids["centroid_tuple"] = centroids["centroid"].apply(
        lambda coord: (coord.y, coord.x)
    )

    # read in travel time matrix
    tts = pd.read_parquet(filepath_or_dirpath).reset_index(drop=True)

    # merge on centroid coordinates tuples
    tts = (
        tts.merge(
            centroids[["id", "centroid_tuple"]],
            left_on="from_id",
            right_on="id",
            how="left",
        )
        .drop(["id"], axis=1)
        .rename(columns={"centroid_tuple": "from_centroid_tuple"})
        .merge(
            centroids[["id", "centroid_tuple"]],
            left_on="to_id",
            right_on="id",
            how="left",
        )
        .drop(["id"], axis=1)
        .rename(columns={"centroid_tuple": "to_centroid_tuple"})
        .merge(
            populations[["id", "population"]],
            left_on="from_id",
            right_on="id",
            how="left",
        )
        .drop(["id"], axis=1)
        .rename(columns={"population": "from_population"})
    )

    # calculate distance between centroids
    tts["inter_centroid_distance"] = haversine_vector(
        tts.from_centroid_tuple.to_list(),
        tts.to_centroid_tuple.to_list(),
    )

    # calculate the accessibility - total pop within time and distance
    accessibility = (
        tts[
            (tts.inter_centroid_distance <= distance_threshold)
            & (tts.travel_time <= travel_time_threshold)
        ]
        .groupby("to_id")["from_population"]
        .sum()
        .reset_index()
        .rename(columns={"from_population": "accessible_population"})
    )

    # calculate the proximity - total pop within distance
    proximity = (
        tts[(tts.inter_centroid_distance <= distance_threshold)]
        .groupby("to_id")["from_population"]
        .sum()
        .reset_index()
        .rename(columns={"from_population": "proximity_population"})
    )

    # merge together to start forming the results
    perf_df = accessibility.merge(proximity, on="to_id", validate="one_to_one")

    # calculate the transport performance
    perf_df["transport_performance"] = (
        perf_df.accessible_population.divide(perf_df.proximity_population)
        * 100
    )

    # merge on population geospatial data
    perf_gdf = populations.merge(
        perf_df,
        left_on="id",
        right_on="to_id",
        how="right",
    ).drop(["to_id"], axis=1)

    return perf_gdf


def plot(
    gdf: gpd.GeoDataFrame,
    column: str = None,
    column_control_name: str = None,
    uc_gdf: gpd.GeoDataFrame = None,
    show_uc_gdf: bool = True,
    point: gpd.GeoDataFrame = None,
    show_point: bool = False,
    point_control_name: str = "POI",
    point_color: str = "red",
    point_buffer: int = None,
    overlay: gpd.GeoDataFrame = None,
    overlay_control_name: str = "Overlay",
    cmap: str = "viridis_r",
    color: str = "#12436D",
    caption: str = None,
    max_labels: int = 9,
    save: str = None,
) -> folium.Map:
    """Plot travel times/transport performance.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The geospatial dataframe to visualise
    column : str, optional
        Column within the dataframe to visualise, by default None meaning no
        colourmap will be added
    column_control_name : str, optional
        Name to column to appear in folium control layer, by default None
        meaning the column name will be used in the folium control layer
    uc_gdf : gpd.GeoDataFrame, optional
        The urban centre geodataframe, by default None meaning no urban centre
        will be added to the visualisation.
    show_uc_gdf : bool, optional
        Boolean flag to control whether the urban centre is displayed on
        opening, by default True meaning it will be initially displayed until
        it is deselected on the contol layer
    point : gpd.GeoDataFrame, optional
        Point of interest marker to be added to the visual, by default None
        meaning no plot will be added.
    show_point : bool, optional
        Boolean flag to control whether the point of interest is displayed on
        opening, by default False meaning it will not be displayed initially
        until it is selected on the control layer.
    point_control_name : str, optional
        Name to give the point of interest in the layer control, by default
        "POI",
    point_color : str, optional
        Color of the point of interest marker, by default "red"
    point_buffer : int, optional
        Distance, in m, to added a dashed line from the point of interest,
        by default None meaning no buffer will be added
    overlay : gpd.GeoDataFrame, optional
        An extra geodataframe that can be added as an overlay layer to the
        visual, by default None meaning no overlay is added
    overlay_control_name : str, optional
        Name of the overlay layer in the overlay control menu, by default
        "Overlay".
    cmap : str, optional
        Color map to use for visualising data, by default "viridis_r". Only
        used when `column` is not None.
    color : str, optional
        Color to set the data (i.e. a fixed value), by default "#12436D". Only
        used when `cmap` is set to None.
    caption : str, optional
        Legend caption, by default None meaning `column` will be used.
    max_labels : int, optional
        Maximum number of legend labels, by default 9. Useful to control the
        distance between legend ticks.
    save : str, optional
        Location to save file, by default None meaning no file will be saved.

    Returns
    -------
    folium.Map
        Folium visualisation output

    """
    # create an empty map layer so individual tiles can be addeded
    m = folium.Map(tiles=None, control_scale=True, zoom_control=True)

    # infromation for carto positron tile
    tiles = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStre'
        'etMap</a> contributors &copy; <a href="https://carto.com/attribut'
        'ions">CARTO</a>'
    )

    # add Carto Positron tile layer
    folium.TileLayer(
        name="Carto Positron Basemap",
        tiles=tiles,
        attr=attr,
        show=False,
        control=True,
    ).add_to(m)

    # add OpenStreetMap tile layer
    folium.TileLayer(
        name="OpenStreetMap Basemap",
        show=False,
        control=True,
    ).add_to(m)

    # handle legend configuration
    legend_kwds = {}
    if caption is not None:
        legend_kwds["caption"] = caption
    legend_kwds["max_labels"] = max_labels

    # handle setting column layer name in control menu
    if column_control_name is None:
        column_control_name = column

    # add data to the map
    m = gdf.explore(
        column,
        m=m,
        color=color,
        cmap=cmap,
        legend_kwds=legend_kwds,
        name=column_control_name,
    )

    # add the urban centre layer, if one is provided
    if uc_gdf is not None:
        m = uc_gdf.explore(
            m=m,
            color="red",
            style_kwds={"fill": None},
            name="Urban Centre",
            show=show_uc_gdf,
        )

    # add a point marker to the map, if one is provided
    if point is not None:
        marker_kwds = {
            "icon": Icon(
                color="red",
                prefix="fa",
                icon="flag-checkered",
            )
        }
        m = point.explore(
            m=m,
            name=point_control_name,
            marker_type="marker",
            marker_kwds=marker_kwds,
            show=show_point,
        )

        # add in a dashed buffer around the point, if requested
        if point_buffer is not None:
            m = (
                point.to_crs("EPSG:27700")
                .buffer(point_buffer)
                .explore(
                    m=m,
                    color=point_color,
                    style_kwds={"fill": None, "dashArray": 5},
                    name="Max Distance from Destination",
                    show=show_point,
                )
            )

    # add in an extra overlay layer, if requested
    if overlay is not None:
        m = overlay.explore(
            m=m,
            color="#F46A25",
            name=overlay_control_name,
        )

    # get and fit the bounds to the added map layers
    m.fit_bounds(m.get_bounds())

    # add a layer control button
    folium.LayerControl().add_to(m)

    # write to file if requested
    if save is not None:
        dir_name = os.path.dirname(save)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        m.save(save)

    return m


# %%
perf_gdf = _transport_performance_pandas(
    metrics_input_dir,
    centroid_gdf,
    pop_gdf,
    travel_time_threshold=MAX_TIME,
    distance_threshold=MAX_DISTANCE,
)

# %%
# write plot to file
m = plot(
    perf_gdf,
    column="transport_performance",
    column_control_name="Transport Performance",
    caption=(
        "Transport Performance (%) (to a destination within the urban centre)"
    ),
    uc_gdf=None,
    max_labels=7,
    cmap="viridis",
    save=TP_OUTPUT,
)

# %%
# convert centroid shapley object to tuple
centroid_gdf["centroid_tuple"] = centroid_gdf["centroid"].apply(
    lambda coord: (coord.y, coord.x)
)

# %%
# read in travel time matrix, sort and reset index (not needed, but useful)
# for checking equivalence with `travel_times`
tt_df = (
    pd.read_parquet(metrics_input_dir)
    .sort_values(["from_id", "to_id"])
    .reset_index(drop=True)
)

# %%
# merge on centroid coordinates tuples
tt_df = (
    tt_df.merge(
        centroid_gdf[["id", "centroid_tuple"]],
        left_on="from_id",
        right_on="id",
        how="left",
    )
    .drop(["id"], axis=1)
    .rename(columns={"centroid_tuple": "from_centroid_tuple"})
    .merge(
        centroid_gdf[["id", "centroid_tuple"]],
        left_on="to_id",
        right_on="id",
        how="left",
    )
    .drop(["id"], axis=1)
    .rename(columns={"centroid_tuple": "to_centroid_tuple"})
    .merge(
        pop_gdf[["id", "population"]],
        left_on="from_id",
        right_on="id",
        how="left",
    )
    .drop(["id"], axis=1)
    .rename(columns={"population": "from_population"})
)

# %%
# calculate distance between centroids
tt_df["inter_centroid_distance"] = haversine_vector(
    tt_df.from_centroid_tuple.to_list(),
    tt_df.to_centroid_tuple.to_list(),
)

# %%
# calculate the accessibility - total pop within time and distance
accessibility = (
    tt_df[
        (tt_df.inter_centroid_distance <= MAX_DISTANCE)
        & (tt_df.travel_time <= MAX_TIME)
    ]
    .groupby("to_id")["from_population"]
    .sum()
    .reset_index()
    .rename(columns={"from_population": "accessible_population"})
)

# calculate the proximity - total pop within distance
proximity = (
    tt_df[(tt_df.inter_centroid_distance <= MAX_DISTANCE)]
    .groupby("to_id")["from_population"]
    .sum()
    .reset_index()
    .rename(columns={"from_population": "proximity_population"})
)

# merge together to start forming the results
perf_df = accessibility.merge(proximity, on="to_id", validate="one_to_one")

# %%
# calculate the transport performance
perf_df["transport_performance"] = (
    perf_df.accessible_population.divide(perf_df.proximity_population) * 100
)

# %%
# show descriptive statistics
perf_df.transport_performance.describe()

# %%
