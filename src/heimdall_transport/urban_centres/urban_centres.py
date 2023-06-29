# %%
import rasterio
import geopandas as gpd
import numpy as np
from scipy.ndimage import label, generic_filter
from rasterio.mask import raster_geometry_mask
import numpy.ma as ma
from scipy.stats import mode


# %%
def filter_cells(file: str, bbox: gpd.GeoDataFrame,
                 band_n: int = 1) -> ma.core.MaskedArray:
    """
    Opens file, loads band and applies mask

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
        ma.core.MaskedArray: raster, clipped to the extent of the bbox
        and masked if extent does not match the boundaries provided.
    """

    with rasterio.open(file) as src:
        masked, affine, win = raster_geometry_mask(
            src,
            bbox.geometry.values,
            crop=True,
            all_touched=True
        )

        # band is clipped to extent of bbox
        rst = src.read(band_n, window=win)
        # pixels that are not within and do not touch 
        # bbox boundaries are masked
        rst_masked = ma.masked_array(rst, masked)

        return rst_masked


def flag_cells(masked_rst: ma.core.MaskedArray,
               cell_pop_thres: int = 1500) -> ma.core.MaskedArray:
    """
    Flags cells that are over the threshold.

    Parameters
    ----------
    masked_rst : ma.core.MaskedArray
        Masked array.
    cell_pop_thres: int
        A cell is flagged if its value is equal or
        higher than the threshold.

    Returns
    -------
        ma.core.MaskedArray: boolean array where cells over
        the threshold are flagged as True.
    """
    flag_array = masked_rst >= cell_pop_thres
    return flag_array


def cluster_cells(flag_array: np.array,
                  diag: bool = False) -> tuple:
    """
    Clusters cells based on adjacency.

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

    if diag is False:
        s = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif diag is True:
        s = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    labelled_array, num_clusters = label(flag_array, s)

    return (labelled_array, num_clusters)


def check_cluster_pop(band: np.array,
                      labelled_array: np.array,
                      num_clusters: int,
                      pop_threshold: int = 50000):
    """
    Checks whether clusters have more than the threshold population
    and changes label for those that don't to 0.

    Parameters
    ----------
    band : np.array or ma.core.MaskedArray
        Original clipped raster with population values.
    labelled_array: np.array
        Array with clusters, each with unique labels.
    num_clusters: int
        Number of unique clusters in the labelled array.
    pop_threshold: int
        Threshold to consider inclusion of cluster. If
        total population in cluster is lower than
        threshold, the cluster label is set to 0.

    Returns
    -------
        np.array: array including only clusters with 
        population over the threshold.
    """
    urban_centres = labelled_array.copy()
    for n in range(1, num_clusters + 1):
        total_pop = ma.sum(ma.masked_where(urban_centres != n, band))
        # print(n, total_pop)

        if total_pop < pop_threshold:
            urban_centres[urban_centres == n] = 0

    return urban_centres


def custom_filter(win, threshold):
    """
    Auxiliary function to apply in generic_filter. Counts non-zero
    values within window and if higher than threshold and cell is zero
    returns mode, else returns value of origin cell.
    """
    if ((np.count_nonzero(win) >= threshold)
            & (win[len(win) // 2] == 0)):
        r = max(mode(win, axis=None, keepdims=True).mode)
    else:
        r = win[len(win) // 2]
    return r


def fill_gaps(urban_centres: np.array, threshold: int = 5) -> np.array:
    """
    For empty cells, checks if at least 5 adjacent cells belong to cluster,
    and if so fills with cluster value.

    Parameters
    ----------
    urban_centres : np.array
        Array including urban centres, i.e. clusters over the population
        threshold.
    threshold: int
        If the number of cells adjacent to any empty cell belonging to
        a cluster is higher than the threshold, the cell is filled with
        the cluster value.

    Returns
    -------
        np.array: array including urban centres with gaps filled

    TODO: need to account for cases where a cell is surrounded by multiple
    clusters.
    """
    gf = urban_centres.copy()
    n = 0
    while True:
        n += 1
        check = gf.copy()
        gf = generic_filter(gf, function=custom_filter, size=3,
                            mode='constant',
                            extra_keywords={'threshold': threshold})
        if np.array_equal(gf, check):
            print('iter', n)
            break
    return gf
