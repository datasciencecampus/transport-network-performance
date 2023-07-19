"""Functions to calculate urban centres following Eurostat definition."""
import rasterio
import geopandas as gpd
import numpy as np
from scipy.ndimage import label, generic_filter
from rasterio.mask import raster_geometry_mask
import numpy.ma as ma
from collections import Counter
from geocube.vector import vectorize
import xarray as xr
import affine
from pyproj import Transformer
from rasterio.transform import rowcol


def filter_cells(file: str, bbox: gpd.GeoDataFrame, band_n: int = 1) -> tuple:
    """Open file, load band and apply mask.

    Parameters
    ----------
    file : str
        Path to geoTIFF file.
    bbox : gpd.GeoDataFrame
        A GeoPandas GeoDataFrame containing boundaries to filter the
        raster. If the boundaries are a bounding box, the raster is
        clipped to the box. If it is an irregular shape (e.g. LA boundaries)
        the raster is clipped to the extent of the shape, and a mask is
        applied to the shape.
    band_n : int
        Band number to load from the geoTIFF.

    Returns
    -------
        tuple[0]: ma.core.MaskedArray: raster, clipped to the extent of the
        bbox and masked if extent does not match the boundaries provided.
        tuple[1]: Affine: transform matrix for the loaded raster.
        tuple[2]: crs string from the raster.

    """
    if not isinstance(file, str):
        raise TypeError(
            "`file` expected string, " f"got {type(file).__name__}."
        )
    if not isinstance(bbox, gpd.GeoDataFrame):
        raise TypeError(
            "`bbox` expected GeoDataFrame, " f"got {type(bbox).__name__}."
        )
    if not isinstance(band_n, int):
        raise TypeError(
            "`band_n` expected integer, " f"got {type(band_n).__name__}"
        )

    with rasterio.open(file) as src:
        if src.crs != bbox.crs:
            raise ValueError("Raster and bounding box crs do not match")

        masked, affine, win = raster_geometry_mask(
            src, bbox.geometry.values, crop=True, all_touched=True
        )

        # band is clipped to extent of bbox
        rst = src.read(band_n, window=win)
        # pixels that are not within and do not touch
        # bbox boundaries are masked
        rst_masked = ma.masked_array(rst, masked)

        return (rst_masked, affine, src.crs)


def flag_cells(
    masked_rst: np.ndarray, cell_pop_thres: int = 1500
) -> np.ndarray:
    """Flag cells that are over the threshold.

    Parameters
    ----------
    masked_rst : np.ndarray
        Clipped (and potentially masked) array.
    cell_pop_thres: int
        A cell is flagged if its value is equal or
        higher than the threshold.

    Returns
    -------
        ma.core.MaskedArray: boolean array where cells over
        the threshold are flagged as True.

    """
    if not isinstance(masked_rst, np.ndarray):
        raise TypeError(
            "`masked_rst` expected numpy array, "
            f"got {type(masked_rst).__name__}."
        )
    if not isinstance(cell_pop_thres, int):
        raise TypeError(
            "`cell_pop_threshold` expected integer, "
            f"got {type(cell_pop_thres).__name__}."
        )

    flag_array = masked_rst >= cell_pop_thres
    return flag_array


def cluster_cells(flag_array: np.ndarray, diag: bool = False) -> tuple:
    """Cluster cells based on adjacency.

    Parameters
    ----------
    flag_array : ma.core.MaskedArray
        Boolean array.
    diag: bool
        If True, diagonals are considered as adjacent.

    Returns
    -------
        tuple[0]: array including all clusters, each with an unique label.
        tuple[1]: number of clusters identified.

    """
    if not isinstance(flag_array, np.ndarray):
        raise TypeError(
            "`masked_rst` expected numpy array, "
            f"got {type(flag_array).__name__}."
        )
    if not isinstance(diag, bool):
        raise TypeError("`diag` must be a boolean.")

    if diag is False:
        s = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif diag is True:
        s = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    labelled_array, num_clusters = label(flag_array, s)

    return (labelled_array, num_clusters)


def check_cluster_pop(
    band: np.ndarray,
    labelled_array: np.ndarray,
    num_clusters: int,
    pop_threshold: int = 50000,
):
    """Filter clusters based on total population.

    Checks whether clusters have more than the threshold population
    and changes label for those that don't to 0.

    Parameters
    ----------
    band : np.ndarray or ma.core.MaskedArray
        Original clipped raster with population values.
    labelled_array: np.ndarray
        Array with clusters, each with unique labels.
    num_clusters: int
        Number of unique clusters in the labelled array.
    pop_threshold: int
        Threshold to consider inclusion of cluster. If
        total population in cluster is lower than
        threshold, the cluster label is set to 0.

    Returns
    -------
        np.ndarray: array including only clusters with
        population over the threshold.

    """
    if not isinstance(band, np.ndarray):
        raise TypeError(
            "`masked_rst` expected numpy array, " f"got {type(band).__name__}."
        )
    if not isinstance(labelled_array, np.ndarray):
        raise TypeError(
            "`labelled_array` expected numpy array, "
            f"got {type(labelled_array).__name__}."
        )
    if not isinstance(num_clusters, int):
        raise TypeError(
            "`num_clusters` expected integer, "
            f"got {type(num_clusters).__name__}"
        )
    if not isinstance(pop_threshold, int):
        raise TypeError(
            "`pop_threshold` expected integer, "
            f"got {type(pop_threshold).__name__}"
        )

    urban_centres = labelled_array.copy()
    for n in range(1, num_clusters + 1):
        total_pop = ma.sum(ma.masked_where(urban_centres != n, band))
        # print(n, total_pop)

        if total_pop < pop_threshold:
            urban_centres[urban_centres == n] = 0

    return urban_centres


def custom_filter(win: np.ndarray, threshold: int) -> int:
    """Check gap filling criteria.

    Counts non-zero values within window and if
    higher than threshold and cell is zero
    returns mode, else returns value of origin cell.

    Parameters
    ----------
    win : np.ndarray
        1-D flattened array of a 3x3 grid, where the
        centre is win[len(win) // 2]. Note that cells
        outside of the edges are filled with 0.
    threshold: int
        Number of cells that need to be filled to change
        the value of the central cell.

    Returns
    -------
        int: value to impute to the central cell.

    """
    if not isinstance(win, np.ndarray):
        raise TypeError(
            "`win` expected numpy array, " f"got {type(win).__name__}."
        )
    if not isinstance(threshold, int):
        raise TypeError(
            "`threshold` expected integer, " f"got {type(threshold).__name__}"
        )

    counter = Counter(win)
    mode_count = counter.most_common(1)[0]
    if (mode_count[1] >= threshold) & (win[len(win) // 2] == 0):
        r = mode_count[0]
    else:
        r = win[len(win) // 2]
    return r


def fill_gaps(urban_centres: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Fill gaps in urban clusters.

    For empty cells, checks if at least 5 adjacent cells belong to cluster,
    and if so fills with cluster value.

    Parameters
    ----------
    urban_centres : np.ndarray
        Array including urban centres, i.e. clusters over the population
        threshold.
    threshold: int
        If the number of cells adjacent to any empty cell belonging to
        a cluster is higher than the threshold, the cell is filled with
        the cluster value.

    Returns
    -------
        np.ndarray: array including urban centres with gaps filled.

    """
    if not isinstance(urban_centres, np.ndarray):
        raise TypeError(
            "`urban_centres` expected numpy array, "
            f"got {type(urban_centres).__name__}."
        )
    if not isinstance(threshold, int):
        raise TypeError(
            "`threshold` expected integer, " f"got {type(threshold).__name__}"
        )

    filled = urban_centres.copy()
    n = 0
    while True:
        n += 1
        check = filled.copy()
        filled = generic_filter(
            filled,
            function=custom_filter,
            size=3,
            mode="constant",
            extra_keywords={"threshold": threshold},
        )
        if np.array_equal(filled, check):
            print("iter", n)
            break
    return filled


def get_x_y(coords: tuple, aff: affine.Affine, crs: rasterio.crs.CRS) -> tuple:
    """Get array index for given coordinates.

    Parameters
    ----------
    coords: tuple
        Tuple with coordinates to convert.
        Must be in format (lat, long) and EPSG: 4326.
    aff: affine.Affine
        Affine transform.
    crs: rasterio.crs.CRS
        valid rasterio crs string.

    Returns
    -------
        tuple: (x, y) coordinates in provided crs.

    """
    transformer = Transformer.from_crs("EPSG:4326", crs)
    x, y = transformer.transform(*coords)
    row, col = rowcol(aff, x, y)

    return row, col


def vectorize_uc(
    uc_array: np.ndarray,
    aff: affine.Affine,
    crs: rasterio.crs.CRS,
    centre: tuple,
    nodata: int = -200,
    type: str = "int32",
) -> gpd.GeoDataFrame:
    """Vectorize raster with urban centre polygon.

    Parameters
    ----------
    uc_array : np.ndarray
        Array including filled urban centres.
    aff: affine.Affine
        Affine transform of the masked raster.
    crs: rasterio.crs.CRS
        crs string of the masked raster.
    centre: tuple
        Tuple with coordinates for city centre, used to filter cluster.
        Must be in format (lat, long) and EPSG: 4326.
    nodata: int
        Value to fill empty cells.
    type: str
        Type for the xarray values.

    Returns
    -------
        gpd.GeoDataFrame: GeoDataFrame with polygon boundaries.

    """
    if not isinstance(uc_array, np.ndarray):
        raise TypeError(
            "`uc_array` expected numpy array, "
            f"got {type(uc_array).__name__}."
        )
    if not isinstance(centre, tuple):
        raise TypeError(
            "`centre` expected tuple, " f"got {type(centre).__name__}"
        )
    if not isinstance(aff, affine.Affine):
        raise TypeError("`aff` must be a valid Affine object")
    if not isinstance(crs, rasterio.crs.CRS):
        raise TypeError("`crs` must be a valid rasterio.crs.CRS object")
    if not isinstance(nodata, int):
        raise TypeError(
            "`nodata` expected integer, " f"got {type(nodata).__name__}"
        )
    if not isinstance(type, str):
        raise TypeError(
            "`type` expected string, " f"got {type(type).__name__}"
        )

    row, col = get_x_y(centre, aff, crs)
    if row > uc_array.shape[0] or col > uc_array.shape[1]:
        raise IndexError("Coordinates fall outside of raster window.")

    cluster_num = uc_array[row, col]
    if cluster_num == 0:
        raise ValueError(
            "Coordinates provided are not included " "within any cluster."
        )

    filt_array = uc_array == cluster_num

    x_array = (
        xr.DataArray(filt_array)
        .astype(type)
        .rio.write_nodata(nodata)
        .rio.write_transform(aff)
        .rio.set_crs(crs, inplace=True)
    )

    gdf = vectorize(x_array)
    gdf.columns = ["label", "geometry"]
    gdf = gdf[gdf["label"] == 1]

    return gdf


def add_buffer(gdf: gpd.GeoDataFrame, size: int = 10000) -> gpd.GeoDataFrame:
    """Add buffer around urban centre polygon.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame including boundaries of urban centre.
    size: int
        Size of the buffer, in metres.

    Returns
    -------
        gpd.GeoDataFrame: GeoDataFrame with buffer boundaries.

    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            "`gdf` expected GeoPandas GeoDataFrame, "
            f"got {type(gdf).__name__}."
        )
    if not isinstance(size, int):
        raise TypeError(
            "`size` expected integer, " f"got {type(size).__name__}"
        )

    b = gdf.buffer(size)
    return b
