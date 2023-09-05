# %% [markdown] noqa: D212, D400, D415
"""
# An End-to-end Example

Am end to end example to run through urban centre detection, population
retrieval, gtfs manipulation and validation, OSM clipping , analysing the
transport network using `r5py` and calculating a performance metric.

> Note: Some bugs have be raised as a result of developing this script. Be
> sure to check out the follow issues on GitHub to ensure they are closed or
> to see temporary workarounds if any of the cells herein do not run:
>
> - [#121](
https://github.com/datasciencecampus/transport-network-performance/issues/121)
> for an issue with `utils/raster/sum_resample_file()` writing to new
> directories

## Preamble
Call in script wide imports and the configuration information.
"""

# %%
import toml
import os
import geopandas as gpd
import gtfs_kit as gk

from pyprojroot import here
from shapely.geometry import box
from transport_performance.urban_centres.raster_uc import UrbanCentre
from transport_performance.population.rasterpop import RasterPop
from transport_performance.gtfs.gtfs_utils import bbox_filter_gtfs
from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.utils.raster import (
    merge_raster_files,
    sum_resample_file,
)

# %%
# config filepath, and loading
CONFIG_FILE = here("notebooks/e2e/config/e2e.toml")
config = toml.load(CONFIG_FILE)

# split out into separate configs to minimise line length
uc_config = config["urban_centre"]
pop_config = config["population"]
gtfs_config = config["gtfs"]

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
# use subset regex to ensure expected year, CRS and resolution are used
if uc_config["override"]:
    merge_raster_files(
        here(uc_config["input_dir"]),
        os.path.dirname(here(uc_config["merged_path"])),
        os.path.basename(uc_config["merged_path"]),
        subset_regex="GHS_POP_E2020_GLOBE_R2023A_54009_1000_",
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

# set the index to the label column to make filtering easier
uc_gdf.set_index("label", inplace=True)
# %%
# visualise outputs
m = uc_gdf[::-1].reset_index().explore("label", cmap="viridis")

# write to file
if uc_config["write_outputs"]:
    if not os.path.exists(os.path.dirname(here(uc_config["output_map_path"]))):
        os.makedirs(os.path.dirname(here(uc_config["output_map_path"])))
    m.save(here(uc_config["output_map_path"]))

m
# %% [markdown] noqa: D212, D400, D415
"""
## Population

Merge 100m gridded data sources together, then resample onto a 200m grid by
summing the consituent cells. Then retrive population data within the
buffered urban centre boundary detected in the step above.

### Data Sources

Using [GHS-POP 100m gridded](https://ghsl.jrc.ec.europa.eu/download.php?ds=pop)
population estimaes, in a **Mollweide CRS**. The following tiles are expected
in `config["population"]["input_dir"]`(which include the British isles and
France):

- R3-C18
- R3-C19
- R4-C18
- R4-C19
"""

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
# extract geometries from urban centre detection
aoi_bounds = uc_gdf.loc["buffer"].geometry
urban_centre_bounds = uc_gdf.loc["vectorized_uc"].geometry

# get population data
rp = RasterPop(here(pop_config["merged_resampled_path"]))
pop_gdf, centroid_gdf = rp.get_pop(
    aoi_bounds,
    threshold=pop_config["threshold"],
    urban_centre_bounds=urban_centre_bounds,
)

# %%
# write interactive visual to file
if pop_config["write_outputs"]:
    rp.plot(which="folium", save=here(pop_config["output_map_path"]))

# view static visual in interactive window
rp.plot(which="cartopy")

# %% [markdown] noqa: D212, D400, D415
"""
## GTFS

Clip the GTFS data to the buffered urban centre area, then clean and validate
then GTFS data.

### Data Sources

In this example a whole of Wales GTFS data source is used, provided by the
[Department for Transport's BODS ](https://data.bus-data.dft.gov.uk/). The
`itm_wales_gtfs.zip` file is expected to be within the directory set by
`config['gtfs']['input_path']`.
"""

# %%
# clip the GTFS to the extent of the urban centre buffered area
if gtfs_config["override"]:
    # get the extent of the urban centre bbox (includes buffer)
    gtfs_bbox = list(uc_gdf.loc["bbox"].geometry.bounds)

    # clip to region of interest, setting crs to match the bbox
    bbox_filter_gtfs(
        in_pth=here(gtfs_config["input_path"]),
        out_pth=here(gtfs_config["filtered_path"]),
        bbox_list=gtfs_bbox,
        units=gtfs_config["units"],
        crs=uc_gdf.crs.to_string(),
    )

# %%
# read in filtered gtfs feed
gtfs = GtfsInstance(
    gtfs_pth=here(gtfs_config["filtered_path"]),
    units=gtfs_config["units"],
)

# show valid dates
available_dates = gtfs.feed.get_dates()
s = available_dates[0]
f = available_dates[-1]
print(f"{len(available_dates)} dates available between {s} & {f}.")

# %%
# check validity, printing warnings and errors
gtfs.is_valid()
print("Errors:")
gtfs.print_alerts()
print("Warnings:")
gtfs.print_alerts(alert_type="warning")

# %%
# clean the gtfs, then re-check the validity and reprint errors/warnings
# note: this will remove 'Repeated pair (route_short_name, route_long_name)'
gtfs.clean_feed()
gtfs.is_valid()
print("Errors:")
gtfs.print_alerts()
print("Warnings:")
gtfs.print_alerts(alert_type="warning")

# %%
# get the route modes - frequency and proportion of modalities
gtfs.get_route_modes()

# %%
# summarise the trips by day of the week
gtfs.summarise_trips()

# %%
# summarise the routes by day of the week
gtfs.summarise_routes()

# %%
# write visuals (stops and hull) and cleaned feed to file
if gtfs_config["write_outputs"]:
    gtfs.viz_stops(here(gtfs_config["stops_map_path"]), create_out_parent=True)
    gtfs.viz_stops(
        here(gtfs_config["hull_map_path"]),
        geoms="hull",
        create_out_parent=True,
    )
    gtfs.feed.write(here(gtfs_config["cleaned_path"]))

# %%
# display a map of only the stops used in `stop_times.txt`, excluding parents
unique_stops = gtfs.feed.stop_times.stop_id.unique()
m = gk.stops.map_stops(gtfs.feed, stop_ids=unique_stops)
if gtfs_config["write_outputs"]:
    m.save(here(gtfs_config["used_stops_map_path"]))
m
# %%
