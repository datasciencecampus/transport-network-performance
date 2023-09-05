# %% [markdown] noqa: D212, D400, D415
"""
# An End-to-end Example

Am end to end example to run through urban centre detection, population
retrieval, gtfs manipulation and validation, OSM clipping , analysing the
transport network using `r5py` and calculating a performance metric.

## Preamble
Call in script wide imports and the configuration information.
"""

# %%
import toml

from pyprojroot import here

# %%
# config filepath, and loading
CONFIG_FILE = here("notebooks/e2e/config/e2e.toml")
config = toml.load(CONFIG_FILE)

# %%
