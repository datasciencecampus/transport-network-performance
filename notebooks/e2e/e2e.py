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
import os
import geopandas as gpd

from pyprojroot import here
from shapely.geometry import box
from transport_performance.urban_centres.raster_uc import UrbanCentre
from transport_performance.utils.raster import (
    merge_raster_files,
)

# %%
# config filepath, and loading
CONFIG_FILE = here("notebooks/e2e/config/e2e.toml")
config = toml.load(CONFIG_FILE)

# split out into separate configs to minimise line length
uc_config = config["urban_centre"]
pop_config = config["population"]

# %% [markdown] noqa: D212, D400, D415
"""
## Urban Centre Detection

Merge 1Km gridded data together. Then detect the urban centre.

### Data Sources

Using [GHS-POP 1Km gridded](https://ghsl.jrc.ec.europa.eu/download.php?ds=pop)
population estimaes, in a **Mollweide CRS**. The following tiles are expected
in `config["urban_centre"]["input_dir"]`(which include the British isles and
France):

- R3-C18
- R3-C19
- R4-C18
- R4-C19
"""

# %%
# merge the urban centre input raster files to form one larger area
if uc_config["override"]:
    merge_raster_files(
        here(uc_config["input_dir"]),
        os.path.dirname(here(uc_config["merged_path"])),
        os.path.basename(uc_config["merged_path"]),
    )

# %%
# put bbox into a geopandas dataframe for `get_urban_centre` input
bbox_gdf = gpd.GeoDataFrame(
    geometry=[box(*uc_config["bbox"])], crs="ESRI:54009"
)

# detect urban centre
uc = UrbanCentre(here(uc_config["merged_path"]))
uc_gdf = uc.get_urban_centre(
    bbox_gdf,
    centre=tuple(uc_config["centre"]),
    buffer_size=uc_config["buffer_size"],
)

# %%
# visualise outputs
m = uc_gdf[::-1].explore("label", cmap="viridis")

# write to file
if uc_config["override"]:
    if not os.path.exists(os.path.dirname(here(uc_config["output_map_path"]))):
        os.makedirs(os.path.dirname(here(uc_config["output_map_path"])))
    m.save(here(uc_config["output_map_path"]))

m
# %%
