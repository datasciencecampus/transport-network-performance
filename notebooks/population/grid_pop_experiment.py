# %%
"""[EXPERIMENTAL] This notebook aims to explore gridded population data.

It uses GHSL - Global Human Settlement Layer - for the population input.
Specifically, GHS-POP-R2023A for the year 2020 and 100x100m population grids
[1]. Coverage, in this case, is most of the UK [2]. The data is cropped to a
region of interest (Newport) and then resampled to 200x200m grids.

It requires the dataset [2] to be unzipped, with the .tif file sorted within
the data/external/population/ directory, named GHSL_2020_UK.tif.

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

"""

# %%
import rasterio as rio
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import folium

from datetime import datetime
from shapely.geometry import box
from rasterio.mask import raster_geometry_mask
from rasterio.warp import calculate_default_transform
from pyproj import Transformer
from pyprojroot import here
from matplotlib import colormaps
from rasterio.warp import reproject, Resampling
from folium.raster_layers import ImageOverlay


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
SRC_DIR = os.path.join(here(), "data", "external", "population", SRC_FILENAME)

# logger name, level and set-up
LOGGER_NAME = "pop-exploration"
LOGGER_LEVEL = logging.INFO
logger = setup_logger(LOGGER_NAME, level=LOGGER_LEVEL)

# bounding box around area of interest in EPGS:4326
BBOX = (-3.205023, 51.503789, -2.725744, 51.680820)

# resammpling scaling factor
RESAMPLING_SCALE_FACTOR = 2

# attributions - used during plotting
POPULATION_ATTR = "GHSL 2020 (R2023)"
BASE_MAP_ATTR = "(C) OpenSteetMap contributors"

# %%
# open raster file and extract meta-data
with rio.open(SRC_DIR) as src:
    logger.info(
        f"Metadata:\n\tCRS: {src.crs}\n\tNo. bands: {src.count}\n\tBounds: "
        f"{src.bounds}"
    )

# %%
# read in whole raster
with rio.open(SRC_DIR) as src:

    # get metadata
    whole_transform = src.transform
    whole_width = src.width
    whole_height = src.height
    whole_bounds = src.bounds

    # read in whole raster
    whole_raster = src.read(1)

# %%
# visualise whole raster - use matplotlib since lots of data
cmap = colormaps.get_cmap("viridis")
cmap.set_under(alpha=0)
plt.imshow(whole_raster, cmap=cmap, vmin=0)

# %%
# read in a cropped window of raster
with rio.open(SRC_DIR) as src:
    # get src metadata
    src_crs = src.crs
    src_nodata = src.nodata

    # reporoject bounding box to raster CRS
    window = reproj_bbox(BBOX, "EPSG:4326", src_crs.to_string())

    # get mask, affine transform and window of cropped area
    windowed_mask, windowed_affine, windowed_loc = raster_geometry_mask(
        src, [box(*window)], crop=True
    )

    # read in raster cropped to window
    windowed_rst = src.read(1, window=windowed_loc)

# %%
# create a destination ndarray to hold reprojected data
destination = np.zeros(
    tuple(int(size / RESAMPLING_SCALE_FACTOR) for size in windowed_rst.shape)
)

# resample scaling by a factor
resampled_rst, resampled_affine = reproject(
    windowed_rst,
    destination,
    src_transform=windowed_affine,
    src_crs=src_crs,
    dst_transform=(
        windowed_affine * windowed_affine.scale(RESAMPLING_SCALE_FACTOR)
    ),
    dst_crs=src_crs,
    resampling=Resampling.sum,
    src_nodata=src_nodata,
    dst_nodata=src_nodata,
)

# %%
# get OpenStreetMap tile layer
map_tile = cimgt.OSM()

# build plot axis and add map tile
ax = plt.axes(projection=map_tile.crs)
ax.figure.set_size_inches(8, 10)
data_crs = ccrs.Mollweide()
ax.add_image(map_tile, 11)

# built a mesh grid of data - x, y are cell centroids in raster crs
height, width = resampled_rst.shape
columns, rows = np.meshgrid(np.arange(width), np.arange(height))
x, y = rio.transform.xy(resampled_affine, rows, columns)

# build a colormap and add pcolormesh plot
cmap = colormaps.get_cmap("hot")
cmap.set_under(alpha=0)
plt_resampled_rst = np.copy(resampled_rst)
plt_resampled_rst[plt_resampled_rst <= 0] = 0
ctf = ax.pcolormesh(
    x,
    y,
    plt_resampled_rst,
    cmap=cmap,
    vmin=1e-10,
    transform=data_crs,
)

# add a colorbar - set 1 at base to notify not displaying 0 population
cbar = plt.colorbar(ctf, ax=ax, fraction=0.034, pad=0.04)
cbar.ax.set_ylabel("Population count per cell", rotation=270, labelpad=20)
cbar.set_ticks(np.concatenate([np.array([1]), cbar.get_ticks()[1:-1]]))

# create an attribution string and add it to the axis
attribution = f"""
Generated on: {datetime.strftime(datetime.now(), "%Y-%m-%d")}
Population data: {POPULATION_ATTR}
Base map: {BASE_MAP_ATTR}"""
ax.text(
    0.01,
    0.01,
    attribution,
    transform=ax.transAxes,
    size=8,
    wrap=True,
    fontdict={"name": "Arial", "color": "#5A5A5A"},
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
    os.path.basename(SRC_DIR).replace(".tif", "_cropped_resampled.tif"),
)

# write to file
with rio.open(
    RESAMPLED_DIR,
    "w",
    driver="GTiff",
    height=resampled_rst.shape[0],
    width=resampled_rst.shape[1],
    count=1,
    dtype=resampled_rst.dtype,
    crs=src_crs,
    transform=resampled_affine,
    nodata=src_nodata,
) as dst:
    dst.write(resampled_rst, 1)

# %%
# get bounds of resampled data
with rio.open(RESAMPLED_DIR) as res:
    resampled_bounds = res.bounds

# %%
# change crs
trans_affine, trans_width, trans_height = calculate_default_transform(
    src_crs.to_string(),
    width=windowed_loc.width,
    height=windowed_loc.height,
    left=resampled_bounds.left,
    bottom=resampled_bounds.bottom,
    right=resampled_bounds.right,
    top=resampled_bounds.top,
    dst_crs="EPSG:4326",
    dst_width=resampled_rst.shape[1],
    dst_height=resampled_rst.shape[0],
)

# %%
# create a destination ndarray to hold reprojected data
trans_dst = np.zeros((trans_height, trans_width))

# resample scaling by a factor
trans_rst, _ = reproject(
    resampled_rst,
    trans_dst,
    src_transform=resampled_affine,
    src_crs=src_crs,
    dst_transform=trans_affine,
    dst_crs="EPSG:4326",
    resampling=Resampling.nearest,
    src_nodata=src_nodata,
    dst_nodata=src_nodata,
)

# %%
# write to file
REPROJ_DIR = os.path.join(
    INTERIM_DIR,
    os.path.basename(SRC_DIR).replace(".tif", "_cropped_resampled_reproj.tif"),
)
with rio.open(
    REPROJ_DIR,
    "w",
    driver="GTiff",
    height=trans_rst.shape[0],
    width=trans_rst.shape[1],
    count=1,
    dtype=trans_rst.dtype,
    crs="EPSG:4326",
    transform=trans_affine,
    nodata=src_nodata,
) as dst:
    dst.write(trans_rst, 1)

# %%
# read reprojected data back in
with rio.open(REPROJ_DIR) as reproj:
    reproj_bounds = reproj.bounds
    reproj_rst = reproj.read(1)

    # filter out those less than 10
    reproj_rst[reproj_rst < 10] = reproj.nodata

# %%
# build a folium basemap
m = folium.Map(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr=(
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMa'
        'p</a> contributors &copy; <a href="https://carto.com/attributions">CA'
        "RTO</a>"
    ),
    control_scale=True,
    zoom_control=True,
)

# create a colormap, setting everthing less than a value to be transparent
cmap = colormaps.get_cmap("viridis")
cmap.set_under(alpha=0)

# add the image overlay
m.add_child(
    ImageOverlay(
        # scale relative to max for cmap
        reproj_rst / reproj_rst.max(),
        bounds=[
            [reproj_bounds.bottom, reproj_bounds.left],
            [reproj_bounds.top, reproj_bounds.right],
        ],
        colormap=cmap,
        opacity=0.7,
        vmin=0,
    )
)

# fit the map to the bounds of the image
m.fit_bounds(m.get_bounds())

# visualise the map
m
# %%
