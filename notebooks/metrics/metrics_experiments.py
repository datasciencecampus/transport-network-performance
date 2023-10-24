# %% [markdown] noqa: D212, D400, D415
"""
# Metrics Experiments

A notebook developed for exerimenting with different approaches to calculating
transport performance metrics.

## Preamble
Call in script wide imports and the configuration information.
"""

# %%
import os
import pandas as pd
import pickle

from haversine import haversine_vector
from pyprojroot import here
from tqdm import tqdm

from transport_performance.utils.defence import (
    _check_parent_dir_exists,
)

# %%
# name of area and source of metrics inputs
AREA_NAME = "newport"
metrics_input_dir = here(
    f"data/processed/analyse_network/newport_e2e/experiments/{AREA_NAME}"
)

# path to raster pop data, pickled
RP_PATH = here("data/processed/analyse_network/newport_e2e/rasterpop.pkl")

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

> Note: this section only needs to be run as a 'one-off'.
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
    batch_filepath = os.path.join(
        metrics_input_dir, f"{AREA_NAME}_id{id}.parquet"
    )

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
