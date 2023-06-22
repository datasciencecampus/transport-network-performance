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
import plotly.express as px
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from datetime import datetime
from shapely.geometry import box
from rasterio.windows import from_bounds
from rasterio.mask import raster_geometry_mask
from pyproj import Transformer
from pyprojroot import here
from skimage.measure import block_reduce
from matplotlib import colormaps
from rasterio.warp import reproject, Resampling


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

    return tuple(round(x / 100) * 100 for x in window)


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

# no data values (to be set to zero)
NO_DATA_VALS = [-200]

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
# reporoject bounding box to raster CRS
window = reproj_bbox(BBOX, "EPSG:4326", src.crs)

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
# read in crop of raster
with rio.open(SRC_DIR) as src:

    # get window meta data
    window_meta = from_bounds(
        left=window[0],
        bottom=window[1],
        right=window[2],
        top=window[3],
        transform=src.transform,
    )

    # read in crop
    window_raster = src.read(1, window=window_meta)

# %%
with rio.open(SRC_DIR) as src:
    CRS = src.crs
    masked, affine, win = raster_geometry_mask(src, [box(*window)], crop=True)

    win_raster = src.read(1, window=win)

# %%
# create a destination ndarray to hold reprojected data
destination = np.zeros(tuple(int(size / 2) for size in win_raster.shape))

# resample scaling by 2
# TODO: remove hard coding of scale and nodata
resampled, resampled_affine = reproject(
    win_raster,
    destination,
    src_transform=affine,
    src_crs=CRS,
    dst_transform=affine * affine.scale(2),
    dst_crs=CRS,
    resampling=Resampling.sum,
    src_nodata=-200,
    dst_nodata=-200,
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
height, width = resampled.shape
columns, rows = np.meshgrid(np.arange(width), np.arange(height))
x, y = rio.transform.xy(resampled_affine, rows, columns)

# build a colormap and add pcolormesh plot
cmap = colormaps.get_cmap("hot")
cmap.set_under(alpha=0)
ctf = ax.pcolormesh(
    x,
    y,
    resampled,
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
# read in windowed raster, fill in no data with 0 for plottint
# use plotly for intereactivity
fiiled_window_raster = np.copy(window_raster)
fiiled_window_raster[fiiled_window_raster <= 0] = 0
px.imshow(fiiled_window_raster)

# %%
# calculate affine transform for windowed region by adjusting the offet of the
# whole datasets affine transform

# asset the row and column adjustments are integers
assert (window_meta.row_off).is_integer()
assert (window_meta.col_off).is_integer()

# get row and column adjustments relative to whole raster
row_adjust = int(window_meta.row_off)
col_adjust = int(window_meta.col_off)

# create affine transform for window_raster by adjusting the offset
window_transform = rio.Affine(
    whole_transform.a,
    whole_transform.b,
    whole_transform.c + (whole_transform.a * col_adjust),
    whole_transform.d,
    whole_transform.e,
    whole_transform.f + (whole_transform.e * row_adjust),
)

# %%
# show xy coords of bbox top left - these should be the same
whole_transformer = rio.transform.AffineTransformer(whole_transform)
window_transformer = rio.transform.AffineTransformer(window_transform)
logger.info(
    f"{whole_transformer.xy(row_adjust, col_adjust)}, "
    f"{window_transformer.xy(0, 0)}"
)

# %%
# check popoulation and positions are the same between widow and whole
row_test = 100
col_test = 150

# xy centroid - should be the same
whole_xy = whole_transformer.xy(row_adjust + row_test, col_adjust + col_test)
window_xy = window_transformer.xy(row_test, col_test)
logger.info(f"{whole_xy}, {window_xy}")

# population value - should be the same
whole_pop = whole_raster[row_adjust + row_test][col_adjust + col_test]
window_pop = window_raster[row_test][col_test]
logger.info(f"{whole_pop}, {window_pop}")

# %%
# resample data into 200x200m grid by summing pop values within larger grid
# TODO: use mask here instead of setting to 0
zero_window_raster = np.copy(window_raster)
zero_window_raster[zero_window_raster <= 0] = 0
resampled_raster = block_reduce(window_raster, block_size=(2, 2), func=np.sum)
resampled_raster.shape

# %%
# plot resampled raster layer
px.imshow(resampled_raster)

# %%
# create affine transform for window_raster by scaling a and e coeffs
resampled_transform = rio.Affine(
    whole_transform.a * 2,
    whole_transform.b,
    whole_transform.c + (whole_transform.a * col_adjust),
    whole_transform.d,
    whole_transform.e * 2,
    whole_transform.f + (whole_transform.e * row_adjust),
)

# %%
# display shapes of rastered layers
logger.info(f"{window_raster.shape}, {resampled_raster.shape}")

# %%
# check resampling of population is correct
start_row = 40  # 45
start_col = 10  # 83

# sum population of neighbouring 100x100m grids
total_window_pop = (
    window_raster[(start_row * 2)][(start_col * 2)]
    + window_raster[(start_row * 2) + 1][(start_col * 2)]
    + window_raster[(start_row * 2)][(start_col * 2) + 1]
    + window_raster[(start_row * 2) + 1][(start_col * 2) + 1]
)

# corresponding resampled population
resampled_pop = resampled_raster[start_row][start_col]
logger.info(f" {total_window_pop}, {resampled_pop}")

# %%
# check affine transform - top left of cell since centroids are different
start_row = 60
start_col = 21

window_loc = window_transform * (start_row * 2, start_col * 2)
resampled_loc = resampled_transform * (start_row, start_col)
logger.info(f"{window_loc}, {resampled_loc}")

# %%
