"""Script to generate gdf with centroids to use as test fixture for r5py.

Data Sources
Using GHS-POP 100m gridded (https://ghsl.jrc.ec.europa.eu/download.php?ds=pop)
population estimates, in a Mollweide CRS. The following tiles are expected
in data/external/population (which include the British isles and
France):

- R3-C18
- R3-C19
- R4-C18
- R4-C19
"""
# %%

import os

import geopandas as gpd
from pyprojroot import here
from shapely.geometry import box

from transport_performance.population.rasterpop import RasterPop
from transport_performance.utils.raster import (
    merge_raster_files,
    sum_resample_file,
)

# %%
# split out into separate configs to minimise line length

os.path.join(here(), "data", "external", "population")
pop_config = {
    "override": True,
    "input_dir": os.path.join(here(), "data", "external", "population"),
    "merged_path": os.path.join(
        here(), "data", "interim", "population" "pop_merged.tif"
    ),
    "merged_resampled_path": os.path.join(
        here(), "data", "processed", "population" "pop_merged_resampled.tif"
    ),
    "threshold": 1,
}

# Bounding box for small are in Newport city centre, including bus station
# and crossing river, 134 points.
bbox = box(-3.01013, 51.57777, -2.97156, 51.59329)
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="epsg: 4326")
bbox_gdf_r = bbox_gdf.to_crs("esri: 54009")
aoi_bounds = bbox_gdf_r.loc[0][0]


# %%
# merge the population input raster files to form one larger area
# use regex to ensure 2020 data, CRS, and resolution are as expected in name
if pop_config["override"]:
    merge_raster_files(
        here(pop_config["input_dir"]),
        os.path.dirname(here(pop_config["merged_path"])),
        os.path.basename(pop_config["merged_path"]),
        subset_regex="GHS_POP_E2020_GLOBE_R2023A_54009_100_",
    )

# %%
# resample 100m grids to 200m grids (default resample factor used)
if pop_config["override"]:
    sum_resample_file(
        here(pop_config["merged_path"]),
        here(pop_config["merged_resampled_path"]),
    )

# %%
# get population data
rp = RasterPop(here(pop_config["merged_resampled_path"]))
pop_gdf, centroid_gdf = rp.get_pop(
    aoi_bounds, threshold=pop_config["threshold"]
)

# save centroids as pickle
centroid_gdf.to_pickle(
    os.path.join(here(), "tests", "data", "newport_centroids.pkl")
)
# %%
