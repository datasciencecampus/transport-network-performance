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
import datetime
import geopandas as gpd
import pandas as pd
import gtfs_kit as gk
import folium

from pyprojroot import here
from shapely.geometry import box
from folium.map import Icon
from r5py import (
    TransportNetwork,
    TravelTimeMatrixComputer,
    TransportMode,
)
from transport_performance.urban_centres.raster_uc import UrbanCentre
from transport_performance.population.rasterpop import RasterPop
from transport_performance.gtfs.gtfs_utils import bbox_filter_gtfs
from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.osm.osm_utils import filter_osm
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
osm_config = config["osm"]
analyse_net_config = config["analyse_network"]
metrics_config = config["metrics"]

# %% [markdown] noqa: D212, D400, D415
"""
## Urban Centre Detection

Merge 1Km gridded data together. Then detect the urban centre.

### Data Sources

Using [GHS-POP 1Km gridded](https://ghsl.jrc.ec.europa.eu/download.php?ds=pop)
population estimaes, in a **Mollweide CRS**. The following tiles are expected
in `config["urban_centre"]["input_dir"]`(which include the British isles and
France). Must use 2020 Epoch or update the `subset_regex` pattern to match your
files in the cell below:

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
population estimates, in a **Mollweide CRS**. The following tiles are expected
in `config["population"]["input_dir"]`(which include the British isles and
France). Must use 2020 Epoch or update the `subset_regex` pattern to match your
files in the cell below:

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
# Can take a couple of minutes...
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
        bbox=gtfs_bbox,
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
# %% [markdown] noqa: D212, D400, D415
"""
## OpenStreetMap

Clip the OSM data to the buffered urban centre area.

### Data Sources

In this example a whole of Wales OSM data source is used, provided by the
[Geofabrik](https://download.geofabrik.de/europe/great-britain.html). The
`wales-latest.osm.pbf` file is expected to be within the directory set by
`config['osm']['input_path']`.
"""

# %%
# clip osm file to bbox of urban centre + buffer detected above
if osm_config["override"]:
    # get the extent of the urban centre bbox (includes buffer)
    # need to convert to EPSG:4326 here since this is required by osmosis
    osm_bbox = list(uc_gdf.to_crs("EPSG:4326").loc["bbox"].geometry.bounds)

    filter_osm(
        pbf_pth=here(osm_config["input_path"]),
        out_pth=here(osm_config["filtered_path"]),
        bbox=osm_bbox,
    )

# %% [markdown] noqa: D212, D400, D415
"""
## Analyse Network

This stage of the pipeline uses `r5py` to calculate the median travel time from
sources (all cells) to the desinations (urban centre cells), as per the set
configuration in `config/e2e.toml`. It also visualises the 'isochrones' of 3
urban centre destinations; in the centre, south west, and eastern regions.
"""

# %%
# build the transport network
trans_net = TransportNetwork(
    here(osm_config["input_path"]),
    [here(gtfs_config["cleaned_path"])],
)

# %%
# build the computer
travel_time_matrix_computer = TravelTimeMatrixComputer(
    trans_net,
    origins=centroid_gdf,
    destinations=centroid_gdf[centroid_gdf.within_urban_centre],
    departure=datetime.datetime(
        analyse_net_config["departure_year"],
        analyse_net_config["departure_month"],
        analyse_net_config["departure_day"],
        analyse_net_config["departure_hour"],
        analyse_net_config["departure_minute"],
    ),
    departure_time_window=datetime.timedelta(
        hours=analyse_net_config["departure_time_window"],
    ),
    max_time=datetime.timedelta(
        minutes=analyse_net_config["max_time"],
    ),
    transport_modes=[TransportMode.TRANSIT],
)

# %%
# run the computer
travel_times = travel_time_matrix_computer.compute_travel_times()

# %%
# a qa stage, checking the travel time results are the same as the accetped
if analyse_net_config["qa_travel_times"]:
    travel_time_qa_path = here(analyse_net_config["qa_path"])
    travel_times_test = pd.read_pickle(travel_time_qa_path)
    pd.testing.assert_frame_equal(travel_times, travel_times_test)

# %%
# alternatively, can save results to use for qa-ing at a later date
if analyse_net_config["save_travel_times_for_qa"]:
    travel_time_out_path = here(analyse_net_config["save_qa_path"])
    travel_times.to_pickle(travel_time_out_path)

# %%


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
# visualise for an ID within the UC
UC_ID = 4110
assert UC_ID in travel_times.to_id.unique()
snippet_id = travel_times[travel_times.to_id == UC_ID]
snippet_id = pop_gdf.merge(snippet_id, left_on="id", right_on="from_id")

if analyse_net_config["write_outputs"]:
    save_path = here(
        analyse_net_config["outputs_dir"] + f"{UC_ID}_cental_uc.html"
    )
else:
    save_path = None

plot(
    snippet_id,
    uc_gdf=uc_gdf[0:1],
    point=centroid_gdf[centroid_gdf.id == UC_ID],
    column="travel_time",
    column_control_name="Travel Time",
    point_control_name="Destination Centroid",
    caption="Median Travel Time (mins)",
    save=save_path,
)

# %%
# visualise for an ID within the UC
UC_ID = 5137
assert UC_ID in travel_times.to_id.unique()
snippet_id = travel_times[travel_times.to_id == UC_ID]
snippet_id = pop_gdf.merge(snippet_id, left_on="id", right_on="from_id")

if analyse_net_config["write_outputs"]:
    save_path = here(analyse_net_config["outputs_dir"] + f"{UC_ID}_sw_uc.html")
else:
    save_path = None

plot(
    snippet_id,
    uc_gdf=uc_gdf[0:1],
    point=centroid_gdf[centroid_gdf.id == UC_ID],
    column="travel_time",
    column_control_name="Travel Time",
    point_control_name="Destination Centroid",
    caption="Median Travel Time (mins)",
    save=save_path,
)

# %%
# visualise for an ID within the UC
UC_ID = 3974
assert UC_ID in travel_times.to_id.unique()
snippet_id = travel_times[travel_times.to_id == UC_ID]
snippet_id = pop_gdf.merge(snippet_id, left_on="id", right_on="from_id")

if analyse_net_config["write_outputs"]:
    save_path = here(analyse_net_config["outputs_dir"] + f"{UC_ID}_e_uc.html")
else:
    save_path = None

plot(
    snippet_id,
    uc_gdf=uc_gdf[0:1],
    point=centroid_gdf[centroid_gdf.id == UC_ID],
    column="travel_time",
    column_control_name="Travel Time",
    point_control_name="Destination Centroid",
    caption="Median Travel Time (mins)",
    save=save_path,
)

# %% [markdown] noqa: D212, D400, D415
"""
## Metrics

This stage of the pipeline takes the outpus from the above section and
calculates the transport performance for all cells within the urban centre.
"""

# %%
# merge on from centroid
distance_df = (
    travel_times.merge(
        centroid_gdf.to_crs("EPSG:27700")[["id", "centroid"]],
        left_on="from_id",
        right_on="id",
    )
    .drop(columns=["id"])
    .rename(columns={"centroid": "from_centroid"})
)

# merge on to centroid
distance_df = (
    distance_df.merge(
        centroid_gdf.to_crs("EPSG:27700")[["id", "centroid"]],
        left_on="to_id",
        right_on="id",
    )
    .drop(columns=["id"])
    .rename(columns={"centroid": "to_centroid"})
)

distance_df.head()

# %%
# convert to geoseries and caluclate distance between from and to
from_s = gpd.GeoSeries(distance_df.from_centroid, crs="EPSG:27700")
to_s = gpd.GeoSeries(distance_df.to_centroid, crs="EPSG:27700")
distance_df["centroid_distance"] = from_s.distance(to_s)

distance_df.head()

# %%
# get the population using the "from_id" (pop of origin to destination)
distance_df["from_population"] = distance_df.merge(
    pop_gdf[["id", "population"]],
    left_on="from_id",
    right_on="id",
    how="left",
)["population"]

distance_df.head()

# %%
# set the maximum distance and time threshold for calculating performance
MAX_DISTANCE = metrics_config["cut_off_distance"]
MAX_TIME = metrics_config["cut_off_time"]

# %%
# calculate total population reach a destination within the time and distance
# group by to_id so that it's total population that reaches the destination id
numerator = (
    distance_df[
        (distance_df.centroid_distance <= MAX_DISTANCE)
        & (distance_df.travel_time <= MAX_TIME)
    ]
    .groupby("to_id")["from_population"]
    .sum()
    .reset_index()
    .rename(columns={"from_population": "reachable_population"})
)
# %%
# calculate total population that is nearby within the distance threshold
# group by to_id so that it's total population nearby the destination
denominator = (
    distance_df[(distance_df.centroid_distance <= MAX_DISTANCE)]
    .groupby("to_id")["from_population"]
    .sum()
    .reset_index()
    .rename(columns={"from_population": "nearby_population"})
)
# %%
# create a transport performance gdf, but merging dataframes
# first the numberator - remove to_id column since it's not needed after merge
perf_gdf = pop_gdf.merge(numerator, left_on="id", right_on="to_id").drop(
    columns=["to_id"]
)

# then the denominator - remove to_id again, since it's not needed after merge
perf_gdf = perf_gdf.merge(denominator, left_on="id", right_on="to_id").drop(
    columns=["to_id"]
)

perf_gdf.head()
# %%
# calculate transport performance, as a percentage
perf_gdf["transport_performance"] = (
    perf_gdf["reachable_population"] / perf_gdf["nearby_population"]
) * 100

# %%
# visualise the transport performance
if metrics_config["write_outputs"]:
    save_path = here(
        metrics_config["outputs_dir"] + "transport_performance.html"
    )
else:
    save_path = None

plot(
    perf_gdf,
    column="transport_performance",
    column_control_name="Transport Performance",
    caption=(
        "Transport Performance (%) (to a destination within the urban centre)"
    ),
    uc_gdf=uc_gdf[0:1],
    max_labels=6,
    cmap="viridis",
    save=save_path,
)

# %%
# QA ID
ID = 4110

# filter distance df to only this id and select a sub-set of columns
snippet_df = distance_df[distance_df.to_id == ID]
snippet_df = snippet_df[
    [
        "from_id",
        "to_id",
        "centroid_distance",
        "travel_time",
        "from_population",
    ]
]

# merge on the geometries for all cells (to this ID), convert to gpd df
snippet_df = snippet_df.merge(
    pop_gdf[["id", "geometry"]], how="left", left_on="from_id", right_on="id"
).drop(columns=["id"])
snippet_gdf = gpd.GeoDataFrame(
    snippet_df, geometry=snippet_df.geometry, crs="ESRI:54009"
)

# %%
# plot the travel time of cells that can reach this ID
if metrics_config["write_outputs"]:
    save_path = here(metrics_config["outputs_dir"] + f"{ID}_travel_time.html")
else:
    save_path = None

plot(
    snippet_gdf[
        (snippet_gdf.travel_time <= MAX_TIME)
        & (snippet_gdf.centroid_distance <= MAX_DISTANCE)
    ],
    column="travel_time",
    caption="Median Travel Time (mins)",
    uc_gdf=uc_gdf[0:1],
    column_control_name="Travel Time",
    point=centroid_gdf[centroid_gdf.id == ID],
    point_control_name="Destination Centroid",
    save=save_path,
)

# %%
# plot population of only cells that can reach this ID
if metrics_config["write_outputs"]:
    save_path = here(
        metrics_config["outputs_dir"] + f"{ID}_reachable_population.html"
    )
else:
    save_path = None

plot(
    snippet_gdf[
        (snippet_gdf.travel_time <= MAX_TIME)
        & (snippet_gdf.centroid_distance <= MAX_DISTANCE)
    ],
    column="from_population",
    caption="Population",
    uc_gdf=uc_gdf[0:1],
    column_control_name="Population",
    point=centroid_gdf[centroid_gdf.id == ID],
    point_control_name="Destination Centroid",
    cmap="viridis",
    save=save_path,
)

# %%
# plot all population cells within MAX_DISTANCE of this ID
if metrics_config["write_outputs"]:
    save_path = here(
        metrics_config["outputs_dir"] + f"{ID}_nearby_population.html"
    )
else:
    save_path = None

plot(
    snippet_gdf[(snippet_gdf.centroid_distance <= MAX_DISTANCE)],
    column="from_population",
    caption="Population",
    uc_gdf=uc_gdf[0:1],
    column_control_name="Population",
    point=centroid_gdf[centroid_gdf.id == ID],
    point_control_name="Destination Centroid",
    cmap="viridis",
    save=save_path,
)

# %%
# plot the acceible cells ontop of the proximity cells
if metrics_config["write_outputs"]:
    save_path = here(
        metrics_config["outputs_dir"] + f"{ID}_population_overlays.html"
    )
else:
    save_path = None

plot(
    snippet_gdf[(snippet_gdf.centroid_distance <= MAX_DISTANCE)],
    column=None,
    caption=None,
    uc_gdf=uc_gdf[0:1],
    show_uc_gdf=False,
    column_control_name="Nearby Population",
    point=centroid_gdf[centroid_gdf.id == ID],
    show_point=True,
    point_control_name="Destination Centroid",
    point_buffer=MAX_DISTANCE,
    overlay=gpd.GeoDataFrame(
        geometry=[
            snippet_gdf[
                (snippet_gdf.travel_time <= MAX_TIME)
                & (snippet_gdf.centroid_distance <= MAX_DISTANCE)
            ].geometry.unary_union
        ],
        crs=snippet_gdf.crs,
    ),
    overlay_control_name="Reachable Population",
    cmap=None,
    color="#12436D",
    save=save_path,
)

# %%
# calculate the total population that can reach this ID
reachable_population = snippet_gdf[
    (snippet_gdf.travel_time <= MAX_TIME)
    & (snippet_gdf.centroid_distance <= MAX_DISTANCE)
].from_population.sum()

reachable_population
# %%
# calculate the total population within MAX_DISTANCE of this ID
nearby_population = snippet_gdf[
    (snippet_gdf.centroid_distance <= MAX_DISTANCE)
].from_population.sum()

nearby_population
# %%
# calculate the transport performance for this ID
tp = reachable_population / nearby_population * 100
tp
# %%
# confirm this ID matches with the overall result
assert perf_gdf[perf_gdf.id == ID].transport_performance.iloc[0] == tp
# %%
# build a results table

# describe columns to include
describe_cols = ["min", "25%", "50%", "75%", "max"]

# get results dataframe and transpose
tp_results = pd.DataFrame(
    perf_gdf[perf_gdf.within_urban_centre].transport_performance.describe()
).T[["min", "25%", "50%", "75%", "max"]]

# add in area and name columns, reset the index to make it a column
tp_results.index = ["Newport"]
tp_results.index.name = "urban centre name"
tp_results = tp_results.reset_index()

# calculate the urban centre area and total population
tp_results["urban centre area"] = uc_gdf[0:1].area[0] * 1e-6
tp_results["urban centre population"] = (
    perf_gdf[perf_gdf.within_urban_centre].population.sum().round().astype(int)
)

# reorder columns
tp_results = tp_results[
    [
        "urban centre name",
        "urban centre area",
        "urban centre population",
    ]
    + describe_cols
]

tp_results

# %%
