# %%
"""[EXPERIMENTAL] This notebook aims to explore gridded population data.

It uses GHSL - Global Human Settlement Layer - for the population input.
Specifically, GHS-POP-R2023A for the year 2020 and 100x100m population grids
[1]. Coverage, in this case, is most of the UK [2]. The data is cropped to a
region of interest (Newport) and then resampled to 200x200m grids.

It requires the dataset [2] to be unzipped, with the .tif file sorted within
the data/external/population/ directory, named GHSL_2020_UK.tif.

Additionally, an experiment is performed by merging together multiple GHSL-POP
rasters to cover UK and France [2-5]. These should be stored in the same
directory as above, but without any filename changes.

References
----------
- [1] Schiavina M., Freire S., Carioli A., MacManus K. (2023):
GHS-POP R2023A - GHS population grid multitemporal (1975-2030).European Commiss
ion, Joint Research Centre (JRC)
PID: http://data.europa.eu/89h/2ff68a52-5b5b-4a22-8f40-c41da8332cfe, doi:10.290
5/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE
- [2] https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023
A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_54
009_100_V1_0_R3_C18.zip
- [3] https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023
A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_54
009_100_V1_0_R3_C19.zip
- [4] https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023
A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_54
009_100_V1_0_R4_C18.zip
- [5] https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023
A/GHS_POP_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_54
009_100_V1_0_R4_C19.zip

"""

# %%
import rasterio as rio
import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import rioxarray
import requests
import textwrap

from datetime import datetime
from pyproj import Transformer
from pyprojroot import here
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
from rasterio.warp import Resampling
from geocube.vector import vectorize
from rioxarray.merge import merge_arrays
from rioxarray.exceptions import NoDataInBounds
from requests.exceptions import HTTPError
from shapely.geometry.polygon import Polygon
from xarray import DataArray


# %%
def setup_logger(
    logger_name: str,
    level: int = logging.INFO,
    file_name: str = None,
) -> logging.Logger:
    """Build a logger instance.

    Parameters
    ----------
    logger_name : str
        name of logger.
    level : int, optional
        logger level (e.g., logging.DEBUG, logging.WARNING etc.), by default
        logging.INFO.
    file_name : str, optional
        logger filename, if needed to write logs to file, by default None
        meaning log messages will not be written to file.

    Returns
    -------
    logging.Logger
        a logger instance with the requested properties.

    """
    # set up the logger and logging level
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # fix the logger format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up a stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)

    # set up logger file handler
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def reproj_bbox(bbox: tuple, crs_from: str, crs_to: str) -> tuple:
    """Reproject a bounding box from one CRS to another.

    Parameters
    ----------
    bbox : tuple
        Bounding box co-ordinates, in the same CRS are `crs_from`. This is the
        bounding box to reproject into `crs_to`. Bounding box coordinate order
        must be (left, bottom, right, top).
    crs_from : str
        CRS to convert from.
    crs_to : str
        CRS to convert to.

    Returns
    -------
    tuple
        `bbox` returned in CRS set by `crs_to`.

    """
    # create transformer and reproject bbox of interest into a window
    transformer = Transformer.from_crs(crs_from, crs_to)
    window_bl = transformer.transform(bbox[1], bbox[0])
    window_tr = transformer.transform(bbox[3], bbox[2])

    # return a single tuple in left, bottom, right, top order
    window = window_bl + window_tr

    return window


# %%
# population source file
SRC_FILENAME = "GHSL_2020_UK.tif"
MERGED_SRC_FILENAME = "GHSL_2020_merged.tif"
MERGED_SRC_FILENAME_2015 = "GHSL_2015_merged.tif"
SRC_DIR = os.path.join(here(), "data", "external", "population", SRC_FILENAME)
MERGED_SRC_DIR = os.path.join(
    here(), "data", "interim", "population", MERGED_SRC_FILENAME
)
MERGED_SRC_DIR_2015 = os.path.join(
    here(), "data", "interim", "population", MERGED_SRC_FILENAME_2015
)

# logger name, level and set-up
LOGGER_NAME = "pop-exploration"
LOGGER_LEVEL = logging.INFO
logger = setup_logger(LOGGER_NAME, level=LOGGER_LEVEL)

# bounding box around area of interest in EPGS:4326
# TODO: tidy up this section
# BBOX = (-3.205023, 51.503789, -2.725744, 51.680820)  # orig BBOX
# BBOX = (-3.206023, 51.503789, -2.726744, 51.680820)  # orig shifted by 100m
# BBOX = (-2.210408, 53.450903, -0.861837, 54.093681)  # leeds
# BBOX = (-1.054688, 51.134555, 0.873413, 51.835778)  # london
# BBOX = (3.319416, 42.673743, 7.873249, 44.671964)  # marseille

# area of interest
AREA_OF_INTEREST = "london"

# get bbox of interest
BBOX_DICT = {
    "newport": (-3.206023, 51.503789, -2.726744, 51.680820),
    "leeds": (-2.212408, 53.450803, -0.862837, 54.095581),
    "london": (-1.054688, 51.134555, 0.873413, 51.835778),
    "marseille": (3.319416, 42.673743, 7.873249, 44.671964),
}
BBOX = BBOX_DICT[AREA_OF_INTEREST]

# nomis urls for mye populations
NOMIS_URLS = {
    "newport": (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2010_1.data.json?geogra"
        "phy=1254276446...1254276882,1254277923...1254277960&date=latest&gende"
        "r=0&c_age=200&measures=20100"
    ),
    "leeds": (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2010_1.data.json?geogra"
        "phy=1254151943...1254154269,1254258198...1254258221,1254261711...1254"
        "261745,1254261853...1254261870,1254261894...1254261918,1254262125...1"
        "254262142,1254262341...1254262353,1254262394...1254262398,1254262498."
        "..1254262532,1254262620...1254262658,1254262922...1254262925&date=lat"
        "est&gender=0&c_age=200&measures=20100"
    ),
    "london": (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2010_1.data.json?geogra"
        "phy=1254096897...1254098287,1254098289...1254098739,1254098741...1254"
        "099990,1254099992...1254100115,1254100117...1254100340,1254100342...1"
        "254100465,1254100467,1254100476...1254100544,1254100546...1254100658,"
        "1254100660...1254102034,1254102068...1254102153,1254102185...12541022"
        "80,1254102282...1254103792,1254103795,1254103797...1254107351,1254107"
        "354...1254107963,1254107965...1254108049,1254108051...1254108089,1254"
        "108091...1254108146,1254108148...1254108290,1254108292...1254108501,1"
        "254108503...1254108514,1254108516...1254108735,1254108761...125411136"
        "2,1254111364...1254115627,1254115630...1254116999,1254117001...125412"
        "0302,1254204377,1254258303...1254258311,1254258313...1254258324,12542"
        "58365,1254258511...1254258526,1254258573...1254258581,1254259015...12"
        "54259033,1254259139...1254259180,1254259193...1254259246,1254259398.."
        ".1254259617,1254259767...1254259799,1254259957...1254259964,125426002"
        "7...1254260037,1254260245...1254260280,1254260330...1254260332,125426"
        "0420...1254260439,1254261069...1254261098,1254262366...1254262393,125"
        "4262935...1254262988,1254263053...1254263088,1254263090...1254263100,"
        "1254263102...1254263106,1254263114...1254263139,1254263141...12542631"
        "51,1254263436...1254263494,1254263661...1254263669,1254265357...12542"
        "65363,1254265589...1254265605,1254265661...1254265680,1254266095...12"
        "54266099,1254266191...1254266347,1254266360...1254266773,1254267710.."
        ".1254267747,1254267749...1254267811,1254267813...1254267830,125426794"
        "9...1254268089&date=latest&gender=0&c_age=200&measures=20100"
    ),
    "marseille": None,
}
NOMIS_URL = NOMIS_URLS[AREA_OF_INTEREST]

# resammpling scaling factor
RESAMPLING_SCALE_FACTOR = 2

# minimum population threshold for plotting
MIN_PLOT_THRESH = 10

# attributions - used during plotting
POPULATION_ATTR = "GHS-POP 2020 (R2023) "
BASE_MAP_ATTR = "(C) OpenSteetMap contributors"
ONS_MYE_ATTR = "ONS, 2020 (via nomisweb)"

# merge file list - in data/external/population/
MERGE_FILE_LIST = [
    "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R3_C18.tif",
    "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R3_C19.tif",
    "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R4_C18.tif",
    "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R4_C19.tif",
]

# merge file list for 2015 epoch - used to merge 2015 data sources
MERGE_FILE_LIST_2015 = [
    "GHS_POP_E2015_GLOBE_R2023A_54009_100_V1_0_R3_C18.tif",
    "GHS_POP_E2015_GLOBE_R2023A_54009_100_V1_0_R3_C19.tif",
    "GHS_POP_E2015_GLOBE_R2023A_54009_100_V1_0_R4_C18.tif",
    "GHS_POP_E2015_GLOBE_R2023A_54009_100_V1_0_R4_C19.tif",
]

# %%
# read in source crs to convert bounds of window
with rio.open(SRC_DIR) as src:
    window = reproj_bbox(BBOX, "EPSG:4326", src.crs.to_string())

# %%
# build a geometry representing the area of interest
geometries = [
    {
        "type": "Polygon",
        "coordinates": [
            [
                [window[0], window[1]],
                [window[2], window[1]],
                [window[2], window[3]],
                [window[0], window[3]],
            ]
        ],
    }
]

# %%
# open data and clip to the above geometry, using from disk (more performant)
xds = rioxarray.open_rasterio(SRC_DIR, masked=True).rio.clip(
    geometries, from_disk=True, all_touched=True
)

# set the variable name of the data to be population
xds.name = "population"

# plot data and show resolution
xds.plot()
print(xds.rio.resolution())

# %%
# show a histogram
xds.plot.hist(bins=10)

# %%
# resample based on scaling factor and using sum resampling
xds_resampled = xds.rio.reproject(
    xds.rio.crs,
    resolution=tuple(
        res * RESAMPLING_SCALE_FACTOR for res in xds.rio.resolution()
    ),
    resampling=Resampling.sum,
)

print(xds_resampled.rio.nodata)
xds_resampled.rio.transform()

# %%
# use geocube to conver raster to geopandas df
gdf = vectorize(xds_resampled.squeeze().astype(np.float32))

# %%
# visualise the results
gdf[gdf["population"] >= MIN_PLOT_THRESH].explore(
    "population",
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr=(
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMa'
        'p</a> contributors &copy; <a href="https://carto.com/attributions">CA'
        "RTO</a>"
    ),
    control_scale=True,
    zoom_control=True,
)

# %%
# get OpenStreetMap tile layer
map_tile = cimgt.OSM(desired_tile_form="L")

# build plot axis and add map tile
ax = plt.axes(projection=map_tile.crs)
ax.figure.set_size_inches(8, 10)
data_crs = ccrs.Mollweide()
ax.add_image(map_tile, 11, cmap="gray")

# build a colormap and add pcolormesh plot
cmap = colormaps.get_cmap("viridis")
cmap.set_under(alpha=0)

# build a colormap and add pcolormesh plot
plt_resampled_rst = np.copy(xds_resampled.squeeze())
plt_resampled_rst[plt_resampled_rst < MIN_PLOT_THRESH] = 0
ctf = ax.pcolormesh(
    xds_resampled.squeeze().x.to_numpy(),
    xds_resampled.squeeze().y.to_numpy(),
    plt_resampled_rst,
    cmap=cmap,
    vmin=MIN_PLOT_THRESH,
    transform=data_crs,
)

# add a colorbar - set base to notify not displaying 0 population
cbar = plt.colorbar(ctf, ax=ax, fraction=0.034, pad=0.04)
cbar.ax.set_ylabel("Population count per cell", rotation=270, labelpad=20)
cbar.set_ticks(
    np.concatenate([np.array([MIN_PLOT_THRESH]), cbar.get_ticks()[1:-1]])
)

# create an attribution string and add it to the axis
grid_res = xds_resampled.rio.resolution()
attribution = f"""
Generated on: {datetime.strftime(datetime.now(), "%Y-%m-%d")}
Population data: {POPULATION_ATTR}
Grid Size: {abs(grid_res[0])}m x {abs(grid_res[1])}m
Base map: {BASE_MAP_ATTR}"""
ax.text(
    0.01,
    0.01,
    attribution,
    transform=ax.transAxes,
    size=8,
    wrap=True,
    fontdict={"name": "Arial", "color": "#000000"},
    va="bottom",
    ha="left",
)

# show plot
plt.tight_layout()
plt.show()

# %%
# get the interim population directory
INTERIM_DIR = os.path.dirname(SRC_DIR).replace("external", "interim")

# make one if it does not exist
if not os.path.exists(INTERIM_DIR):
    os.mkdir(INTERIM_DIR)

# create full filepath for cropped and resampeld tif file
RESAMPLED_DIR = os.path.join(
    INTERIM_DIR,
    os.path.basename(SRC_DIR).replace(".tif", "_xarray_cropped_resampled.tif"),
)

# write to GeoTIFF raster file
# note: nodata arg will be reset to original value by rioxarry when writing
xds_resampled.rio.to_raster(RESAMPLED_DIR)

# %%
# re-open save file (as a check)
xds_res = rioxarray.open_rasterio(RESAMPLED_DIR, masked=True)

# set the variable name of the data to be population
xds_res.name = "population"

# plot data and show resolution
xds_res.plot()

# %%
# build merge file directories
merge_dirs = [
    os.path.join(here(), "data", "external", "population", name)
    for name in MERGE_FILE_LIST
]

# %%
# build mere dataset list
arrays = []
for merge_dir in merge_dirs:
    arrays.append(rioxarray.open_rasterio(merge_dir, masked=True))

# display bounds for checking
for array in arrays:
    logger.info(array.rio.bounds())

# %%
# merge the datasets together
xds_merged = merge_arrays(arrays)
logger.info(xds_merged.rio.bounds())

# %%
# plot merged data - note: this cell takes around 3.5mins to run
xds_merged.plot(vmin=MIN_PLOT_THRESH, norm=LogNorm())

# %%
# get the interim population directory
INTERIM_POP_DIR = os.path.join(here(), "data", "interim", "population")

# make dir if it does not exist
if not os.path.exists(INTERIM_POP_DIR):
    os.mkdir(INTERIM_POP_DIR)

# create full filepath for merged tif file
MERGED_DIR = os.path.join(INTERIM_POP_DIR, "GHSL_2020_merged.tif")

# write to GeoTIFF raster file
xds_merged.rio.to_raster(MERGED_DIR)

# %%
# resample merged data based on scaling factor and using sum resampling
# note: this cell takes around 1.75 minutes
xds_merged_resampled = xds_merged.rio.reproject(
    xds_merged.rio.crs,
    resolution=tuple(
        res * RESAMPLING_SCALE_FACTOR for res in xds_merged.rio.resolution()
    ),
    resampling=Resampling.sum,
)

# %%
# create full filepath for merged and resampeld tif file
MERGED_RESAMPLED_DIR = os.path.join(
    INTERIM_POP_DIR, "GHSL_2020_merged_200.tif"
)

# write to GeoTIFF raster file
xds_merged_resampled.rio.to_raster(MERGED_RESAMPLED_DIR)

# %%
# use merged dataset herein
# open data and clip to the above geometry, using from disk (more performant)
xds_clip = rioxarray.open_rasterio(MERGED_RESAMPLED_DIR, masked=True).rio.clip(
    geometries, from_disk=True, all_touched=True
)

# set the variable name of the data to be population
xds_clip.name = "population"

# plot data and show resolution
xds_clip.plot()

# %%
# use geocube to conver raster to geopandas df
gdf = vectorize(xds_clip.squeeze().astype(np.float32))

# %%
# TODO: need to functionise
# visualise the results
m = gdf[gdf["population"] >= MIN_PLOT_THRESH].explore(
    "population",
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr=(
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMa'
        'p</a> contributors &copy; <a href="https://carto.com/attributions">CA'
        "RTO</a>"
    ),
    control_scale=True,
    zoom_control=True,
)

# define and outputs directory
POP_OUTPUTS_DIR = os.path.join(here(), "outputs", "population")

# make one if it does not exist
if not os.path.exists(POP_OUTPUTS_DIR):
    os.mkdir(POP_OUTPUTS_DIR)

# save to file
m.save(os.path.join(POP_OUTPUTS_DIR, f"{AREA_OF_INTEREST}.html"))

# %%
# TODO: need to functionise
# get OpenStreetMap tile layer
map_tile = cimgt.OSM(desired_tile_form="L")

# build plot axis and add map tile
ax = plt.axes(projection=map_tile.crs)
ax.figure.set_size_inches(8, 10)
data_crs = ccrs.Mollweide()
ax.add_image(map_tile, 12, cmap="gray")

# build a colormap and add pcolormesh plot
cmap = colormaps.get_cmap("viridis")
cmap.set_under(alpha=0)

# build a colormap and add pcolormesh plot
plt_resampled_rst = np.copy(xds_clip.squeeze())
plt_resampled_rst[plt_resampled_rst < MIN_PLOT_THRESH] = 0
ctf = ax.pcolormesh(
    xds_clip.squeeze().x.to_numpy(),
    xds_clip.squeeze().y.to_numpy(),
    plt_resampled_rst,
    cmap=cmap,
    vmin=MIN_PLOT_THRESH,
    transform=data_crs,
)

# add a colorbar - set base to notify not displaying 0 population
cbar = plt.colorbar(ctf, ax=ax, fraction=0.034, pad=0.04)
cbar.ax.set_ylabel("Population count per cell", rotation=270, labelpad=20)
cbar.set_ticks(
    np.concatenate([np.array([MIN_PLOT_THRESH]), cbar.get_ticks()[1:-1]])
)

# create an attribution string and add it to the axis
grid_res = xds_clip.rio.resolution()
attribution = f"""
Generated on: {datetime.strftime(datetime.now(), "%Y-%m-%d")}
Population data: {POPULATION_ATTR}
Grid Size: {abs(grid_res[0])}m x {abs(grid_res[1])}m
Base map: {BASE_MAP_ATTR}"""
ax.text(
    0.01,
    0.01,
    attribution,
    transform=ax.transAxes,
    size=8,
    wrap=True,
    fontdict={"name": "Arial", "color": "#000000"},
    va="bottom",
    ha="left",
)

# show plot
plt.tight_layout()
plt.show()

# %%
# check equivalence of orignal method
# note ignoring last row/col since BBOX isn't always divisible by 200m when
# reading in 100x100m grid to resample it effectively (undersums that row/col)
assert xds_clip.rio.transform() == xds_resampled.rio.transform()
assert xds_clip.rio.bounds() == xds_resampled.rio.bounds()
assert xds_clip.shape == xds_resampled.shape
assert np.array_equal(
    xds_resampled[:, :-1, :-1].to_numpy(),
    xds_clip[:, :-1, :-1].to_numpy(),
    equal_nan=True,
)

# %%
# sense check values against mid-year estimates
# following cells can be run independently after loading in the window for the
# area of interest

# %%
# open data and clip to the above geometry, using from disk (more performant)
xds_sc = rioxarray.open_rasterio(MERGED_SRC_DIR, masked=True).rio.clip(
    geometries, from_disk=True, all_touched=True
)

# set the variable name of the data to be population
xds_sc.name = "population"

# plot data and show resolution
xds_sc.plot()

# %%


# read in nomis population data for area of interest
def get_nomis_myes(nomis_url: str, obs_col_name: str = "mye") -> pd.DataFrame:
    """Get mid-year population estimates from nomis.

    Parameters
    ----------
    nomis_url : str
        url to nomis end point for the data
    obs_col_name : str, optional
        name of mye column, by default "mye"

    Returns
    -------
    pd.DataFrame
        nomis myes as a pandas dataframe

    """
    try:
        response = requests.get(nomis_url)
        response.raise_for_status()
    except HTTPError as he:
        logger.error(f"A HTTP error occured: {he}")
    except Exception as e:
        logger.error(f"An error occured when reading nomis data {e}")

    pops = []
    for obs in response.json()["obs"]:
        pops.append([obs["geography"]["geogcode"], obs["obs_value"]["value"]])

    pop_df = pd.DataFrame(pops, columns=["OA11CD", obs_col_name])

    return pop_df


# get 2020 mid-year estimates
logger.info("Retrieving MYEs 2020...")
pop_df = get_nomis_myes(NOMIS_URL, "mye")

# get 2011 census estimates
logger.info("Retrieving 2011 census estimates...")
pop_df = pop_df.merge(
    get_nomis_myes(NOMIS_URL.replace("date=latest", "date=2011"), "census"),
    on="OA11CD",
)

# %%
# read in ONS geoportal boundaries
active_response = True
result_offset = 0
oa_bounds = []

# repeat calls untill all the boundaries are retrieved - 2000 per batch
while active_response:
    try:
        response = requests.get(
            "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/service"
            "s/Output_Areas_Dec_2011_Boundaries_EW_BFC_2022/FeatureServer/0/qu"
            "ery?where=1%3D1&outFields=OA11CD&geometry="
            f"{BBOX[0]}%2C{BBOX[1]}%2C{BBOX[2]}%2C{BBOX[3]}&geometryType=esriG"
            "eometryEnvelope&inSR=4326&spatialRel=esriSpatialRelContains&outSR"
            f"=4326&f=geojson&resultOffset={result_offset}"
        )
        response.raise_for_status()
    except HTTPError as he:
        logger.error(f"A HTTP error occured: {he}")
    except Exception as e:
        logger.error(f"An error occured when reading nomis data {e}")

    if response.json()["features"] != []:
        oa_bounds.append(
            gpd.GeoDataFrame.from_features(response.json(), crs="EPSG:4326")
        )
        result_offset += 2000
    else:
        active_response = False

# %%
# connvert response to geodataframe
oa_bounds_gdf = pd.concat(oa_bounds)

# match the crs with the raster data
oa_bounds_gdf = oa_bounds_gdf.to_crs(src.crs.to_string())
oa_bounds_gdf.plot()

# %%
# merege on population data and filter to OAs only in pop data
oa_gdf = oa_bounds_gdf.merge(pop_df, on="OA11CD", how="right")
assert len(pop_df) == len(oa_gdf)

oa_gdf.plot("mye", legend=True)

# %%


def clip_sum(geometry: Polygon, rst: DataArray) -> float:
    """Clip rastered data to geometry and estimate population.

    Parameters
    ----------
    geometry : Polygon
        Polygon representing the bounds to clip to. Must be in the same CRS are
        the raster data.
    rst : DataArray
        Raster data array

    Returns
    -------
    float
        Sum values within clip, excluding nan values.

    """
    try:
        rst_clipped = rst.rio.clip([geometry]).to_numpy()
    except NoDataInBounds:
        return 0
    return np.nansum(rst_clipped)


def clip_sum_overlap(geometry: Polygon, rst: DataArray) -> float:
    """Clip raster data to geometry and estimate population.

    Method includes all cells that touch the provided geometry. Then scales
    the cell value based on the fraction of overlap between the cell and the
    geometry. The total is then then combined sum of these fractional cell
    values.

    Parameters
    ----------
    geometry : Polygon
        Polygon representing the bounds to clip to. Must be in the same CRS are
        the raster data.
    rst : DataArray
        Raster data array

    Returns
    -------
    float
        Sum values within clip, excluding nan values, and adjusting for
        fractional overlap of cells/geometry.

    """
    # clip to geometry and include all cells that touch
    try:
        rst_clipped = rst.rio.clip([geometry], all_touched=True)
    except NoDataInBounds:
        return 0

    # vectorise and drop masked/null cells (nodata regions)
    try:
        clipped_gdf = vectorize(rst_clipped.squeeze().astype(np.float32))
    except ValueError:
        # TODO: use this approach, should work for all conditions
        clipped_gdf = vectorize(rst_clipped.squeeze(axis=0).astype(np.float32))
    clipped_gdf.dropna(subset=["population"], inplace=True)

    # calculate the fractional overlap within the OA
    clipped_gdf["overlap"] = (
        clipped_gdf.intersection(geometry).area / clipped_gdf.area
    )

    # calculate the OA estimate by scaling the cell population by the
    # fractional overlap
    oa_estimate = (clipped_gdf["overlap"] * clipped_gdf["population"]).sum()

    return oa_estimate


# %%
# apply the clip_sum_overlap func to all OAs and calculate the difference
oa_gdf["ghs"] = oa_gdf.geometry.apply(clip_sum_overlap, args=[xds_sc])
oa_gdf["pop_diff"] = oa_gdf["mye"] - oa_gdf["ghs"]

# %%
# repeat above, but exclude cells below a threshold
threshs = [1, 3, 5, 7, 10]
for thresh in threshs:
    logger.info(f"Calculating GHS-POP estimate using {thresh}...")
    xds_thresh = xds_sc.copy()
    xds_thresh = xds_thresh.where(xds_thresh > thresh)
    oa_gdf[f"ghs_thresh_{thresh}"] = oa_gdf.geometry.apply(
        clip_sum_overlap, args=[xds_thresh]
    )
    oa_gdf[f"pop_diff_thresh_{thresh}"] = (
        oa_gdf["mye"] - oa_gdf[f"ghs_thresh_{thresh}"]
    )

# %%
# build an interactive folium map displaying OA pop estimate difference
m = oa_gdf.explore(
    "pop_diff",
    legend=True,
    cmap="RdBu",
    vmin=-600,
    vmax=600,
    tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attr=(
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMa'
        'p</a> contributors &copy; <a href="https://carto.com/attributions">CA'
        "RTO</a>"
    ),
    legend_kwds={"backgroundcolor": "white"},
)

m.save(os.path.join(here(), "outputs", "mye", f"{AREA_OF_INTEREST}.html"))
del m

# %%
# build a beeswarn to display distribution of OA differences
fig, ax = plt.subplots(figsize=(16, 9))

# for visualisation purposes, do not display extreme outlies
plot_lim = 1000
plot_col = "pop_diff"
plot_df = oa_gdf[
    (oa_gdf[plot_col] < plot_lim) & ((oa_gdf[plot_col] > -plot_lim))
]

# add a colored region to show 10-90 percentile range
perc_low, perc_high = np.percentile(oa_gdf[plot_col], [5, 95])
p_range = perc_high - perc_low
ax.axvspan(
    perc_low,
    perc_high,
    color="gold",
    alpha=0.5,
    label=(
        f"Percentile Range (10-90):\n{p_range:.0f} [{perc_low:.0f} to "
        f"{perc_high:.0f}]"
    ),
)

# add a colored region to show IQR
iqr_low, iqr_high = np.percentile(oa_gdf[plot_col], [25, 75])
iqr = iqr_high - iqr_low
ax.axvspan(
    iqr_low,
    iqr_high,
    color="darkorange",
    alpha=0.5,
    label=f"IQR: {iqr:.0f} [{iqr_low:.0f} to {iqr_high:.0f}]",
)

# add a line to show the median difference
median = oa_gdf[plot_col].median()
ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.0f}")

# add beswarm plot - note it can take 10/20s to plot
# TODO: automatically asign size
sns.swarmplot(plot_df, x=plot_col, size=3, ax=ax, label="OA Pop. Difference")

# fix the scales to the nearest 1000
limit = np.abs(ax.get_xlim()).max()
limit = round(limit / 1e3) * 1e3
ax.set_xticks(np.arange(-limit, limit + 200, 200))

# add minor grids and adjust formatting of y axis
ax.xaxis.grid()
minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)
ax.xaxis.grid(which="minor", linestyle="--", alpha=0.5)
ax.set_yticks([])

# add an indicator to show the 'side' of the greater estimate
indicator = ax.text(
    0,
    -0.45,
    "Greater  Estimate\nGHS-POP ◀---       ---▶ ONS-MYE",
    va="center",
    ha="center",
)
indicator.set_bbox(dict(facecolor="white", edgecolor="black", alpha=0.9))

# create an attribution string and add it to the axis
grid_res = xds_sc.rio.resolution()
attribution = f"""
Generated on: {datetime.strftime(datetime.now(), "%Y-%m-%d")}
Gridded pop. estimates: {POPULATION_ATTR}
Grid Size: {abs(grid_res[0])}m x {abs(grid_res[1])}m
Mid year pop. estimates: {ONS_MYE_ATTR}"""
ax.text(
    0.01,
    0.01,
    attribution,
    transform=ax.transAxes,
    size=8,
    wrap=True,
    fontdict={"name": "Arial", "color": "#000000"},
    va="bottom",
    ha="left",
)

# rename x axis
ax.set_xlabel(
    "Difference between ONS MYE and GHS-POP Output Area Population Estimate"
)

# reorder legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

# if extereme outlier exist (positive) add text to show values
if oa_gdf[oa_gdf[plot_col] >= plot_lim][plot_col].any():
    pos_fliers = oa_gdf[oa_gdf[plot_col] >= plot_lim][plot_col].to_list()
    pos_fliers = [f"{x:.0f}" for x in sorted(pos_fliers)]
    pos_fliers_text = ", ".join(pos_fliers)
    pos_indicator = ax.text(
        0.98,
        0.65,
        f"Not displayed ---▶\n{textwrap.fill(pos_fliers_text, 24)}",
        va="center",
        ha="right",
        transform=ax.transAxes,
    )
    pos_indicator.set_bbox(
        dict(facecolor="white", edgecolor="white", alpha=0.5)
    )

# if extereme outlier exist (negative) add text to show values
if oa_gdf[oa_gdf[plot_col] <= -plot_lim][plot_col].any():
    neg_fliers = oa_gdf[oa_gdf[plot_col] <= -plot_lim][plot_col].to_list()
    neg_fliers = [f"{x:.0f}" for x in sorted(neg_fliers)]
    neg_fliers_text = ", ".join(neg_fliers)
    neg_indicator = ax.text(
        0.02,
        0.65,
        f"◀--- Not displayed\n{neg_fliers_text}",
        va="center",
        ha="left",
        transform=ax.transAxes,
    )
    neg_indicator.set_bbox(
        dict(facecolor="white", edgecolor="white", alpha=0.5)
    )

plt.show()

# %%
# build boxplots for displaying effect of pop threshold - melt data first
pop_diff_cols = [col for col in oa_gdf.columns if col.startswith("pop_diff")]
plot_df = oa_gdf[["OA11CD"] + pop_diff_cols].melt(
    id_vars="OA11CD",
    value_vars=pop_diff_cols,
    var_name="Method",
    value_name="Difference",
)

fig, ax = plt.subplots(figsize=(16, 9))
sns.boxplot(plot_df, x="Difference", y="Method", ax=ax, orient="h")

ax.set_xlabel(
    "Difference between ONS MYE and GHS-POP Output Area Population Estimate"
)
plt.show()

# %%
# build choropleth maps for MYE and GHS estimates
fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# get consitent min and max to ensure consistent colorscales
vmin = min(oa_gdf["mye"].min(), oa_gdf["ghs"].min())
vmax = max(oa_gdf["mye"].max(), oa_gdf["ghs"].max())

# plot both estimates
for ax, col in zip(axes.flat, ["mye", "ghs"]):
    oa_gdf.to_crs("EPSG:4326").plot(
        col,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    ax.axis("off")
    ax.set_title(col.upper())

# make a common colorbar at the bottom of the figure
fig = ax.get_figure()
cax = fig.add_axes([0.2, 0.2, 0.6, 0.03])
sm = plt.cm.ScalarMappable(
    cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.ax.set_xlabel("OA Population")

plt.show()

# %%
# build a choropleth map to display the difference
fig, ax = plt.subplots(figsize=(16, 9))

cbar = oa_gdf.to_crs("EPSG:4326").plot(
    "pop_diff",
    ax=ax,
    cmap="RdBu",
    vmin=-600,
    vmax=600,
    legend=True,
    legend_kwds={
        "label": "Difference (MYE-GHS)",
    },
)

# set the background to grey since the mid point of the colorscale is white
ax.set_facecolor("grey")

# hide all the tick marks and labels
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])

# add a note to show the colorscale is saturated
note = (
    "Note: Color scale is saturated to +/- 600 to improve interpretability"
    "in OAs where the difference is small. Differences may lie outside these"
    "limits"
)
ax.text(
    0.01,
    0.01,
    note,
    transform=ax.transAxes,
    size=8,
    wrap=True,
    fontdict={"name": "Arial", "color": "#000000"},
    va="bottom",
    ha="left",
)

plt.show()
# %%
# open data and clip to the above geometry (2015 data)
xds_sc_2015 = rioxarray.open_rasterio(
    MERGED_SRC_DIR_2015, masked=True
).rio.clip(geometries, from_disk=True, all_touched=True)

# set the variable name of the data to be population
xds_sc_2015.name = "population"

# plot data and show resolution
xds_sc_2015.plot()

# %%
# apply the clip_sum_overlap func to all OAs and calculate the difference
oa_gdf["ghs_2015"] = oa_gdf.geometry.apply(
    clip_sum_overlap, args=[xds_sc_2015]
)

# calulcate the diff with census
oa_gdf["census_diff_2020"] = oa_gdf["census"] - oa_gdf["ghs"]
oa_gdf["census_diff_2015"] = oa_gdf["census"] - oa_gdf["ghs_2015"]

# %%
# melt the df
melt_cols = ["OA11CD", "pop_diff", "census_diff_2020", "census_diff_2015"]
plot_df = oa_gdf[melt_cols].melt(
    id_vars="OA11CD",
    value_vars=["pop_diff", "census_diff_2020", "census_diff_2015"],
    var_name="Estimate",
    value_name="Difference",
)

fig, ax = plt.subplots(figsize=(16, 9))
sns.boxplot(plot_df, x="Difference", y="Estimate", ax=ax, orient="h")

ax.set_xlabel(
    "Difference between ONS and GHS-POP Output Area Population Estimates"
)

ax.set_yticklabels(
    [
        "MYE 2020 vs GHS-POP 2020",
        "Census 2011 vs GHS-POP 2020",
        "Census 2011 vs GHS-POP 2015",
    ]
)
ax.grid(axis="x")

plt.show()

# %%
# build choropleth maps for MYE and census differences estimates
fig, axes = plt.subplots(1, 3, figsize=(16, 9), facecolor=(0, 0, 0))

# get consitent min and max to ensure consistent colorscales
vmin = -600
vmax = 600

# build a title map
title_map = {
    "pop_diff": "MYE 2020 vs GHS-POP 2020",
    "census_diff_2020": "Census 2011 vs GHS-POP 2020",
    "census_diff_2015": "Census 2011 vs GHS-POP 2015",
}

# plot both estimates
for ax, col in zip(axes.flat, title_map.keys()):
    oa_gdf.to_crs("EPSG:4326").plot(
        col, ax=ax, vmin=vmin, vmax=vmax, cmap="RdBu"
    )
    ax.axis("off")
    ax.set_title(title_map[col], color="white")

# make a common colorbar at the bottom of the figure
fig = ax.get_figure()
cax = fig.add_axes([0.2, 0.2, 0.6, 0.03])
sm = plt.cm.ScalarMappable(
    cmap="RdBu", norm=plt.Normalize(vmin=vmin, vmax=vmax)
)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.ax.set_xlabel("OA Population Difference", color="White")
cbar.ax.xaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white")

plt.tight_layout()
plt.show()
# %%
