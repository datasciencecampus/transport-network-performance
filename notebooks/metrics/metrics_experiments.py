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

# %%
# read in the travel times
travel_times = pd.read_pickle(ANALYSE_NETWORK_OUTPUTS)
travel_times.head()

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

# %%
