"""Functions to calculate urban centres following Eurostat definition."""
from collections import Counter

import affine
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import pandas as pd
import pathlib
import rasterio
import xarray as xr

from geocube.vector import vectorize
from pyproj import Transformer
from rasterio.mask import raster_geometry_mask
from rasterio.transform import rowcol
from scipy.ndimage import generic_filter, label
from transport_performance.utils.defence import _is_expected_filetype
from typing import Union


class UrbanCentre:
    """Object to create and store urban centres.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
        Path to the raw raster file

    exp_ext: list = [".tif", ".tiff", ".tff"], optional
        List of acceptable raster file extensions, defaults to
        [".tif", ".tiff", ".tff"].

    Attributes
    ----------
    file : Union[str, pathlib.Path]
        Path to the raw raster file.
    aff : affine.Affine
        Affine transform matrix for the raster file.
    crs : rasterio.crs.CRS
        CRS string for the raster.
    output : gpd.GeoDataFrame
        GeoDataFrame including vector information for the urban centre, buffer
        and bounding box limits.

    Methods
    -------
    get_urban_centre
        Calculates urban centre from population raster and returns vectorised
        geography for urban centre, buffer and bounding box.

    Raises
    ------
    TypeError
        When `file` is not either of string or pathlib.Path.
    FileNotFoundError
        When `file` does not exist on disk.
    ValueError
        When `file` does not have the expected file extension(s).

    Notes
    -----
    Intermediate results for the urban centre calculations are stored as
    internal (mangled) attributes, but can be accessed for examination.

    """

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        exp_ext: list = [".tif", ".tiff", ".tff"],
    ):

        # check that path is str or PosixPath
        _is_expected_filetype(path, "file", exp_ext=exp_ext)
        self.file = path

    def get_urban_centre(
        self,
        bbox: gpd.GeoDataFrame,
        centre: tuple,
        centre_crs: str = None,
        band_n: int = 1,
        cell_pop_threshold: int = 1500,
        diag: bool = False,
        cluster_pop_threshold: int = 50000,
        cell_fill_threshold: int = 5,
        vector_nodata: int = -200,
        buffer_size: int = 10000,
    ) -> gpd.GeoDataFrame:
        """Get urban centre.

        Wrapper for functions to get urban centre, buffer and bbox.

        Parameters
        ----------
        bbox : gpd.GeoDataFrame
            A GeoPandas GeoDataFrame containing boundaries to filter the
            raster. If the boundaries are a bounding box, the raster is
            clipped to the box. If it is an irregular shape (e.g. LA
            boundaries) the raster is clipped to the extent of the shape,
            and a mask is applied to the shape.
        centre : tuple
            Tuple with coordinates for city centre. Urban centres that do not
            contain these coordinates will be filtered out.
        centre_crs : str = None, optional
            crs string of the centre coordinates. If None, it will default to
            raster_crs.
        band_n : int = 1, optional
            Band number to load from the geoTIFF.
        cell_pop_threshold : int = 1500, optional
            When calculating urban centre, keep cells whose value is equal or
            higher than the threshold.
        diag : bool = False, optional
            When calculating clusters, if True, diagonals are considered as
            adjacent and included in the cluster.
        cluster_pop_threshold : int = 50000, optional
            Threshold to consider inclusion of cluster. If total population in
            cluster is lower than threshold, the cluster label is filtered out.
        cell_fill_threshold : int = 5, optional
            Defines gap filling behaviour. If the number of cells adjacent to
            any empty cell belonging to a cluster is higher than the threshold,
            the cell is filled with the cluster value. Needs to be between 5
            and 8.
        vector_nodata : int = -200, optional
            Value to fill empty cells. Select a negative value that you would
            not expect to encounter within the raster population data.
        buffer_size : int = 10000, optional
            Size of the buffer around the urban centre, in the distance units
            of the `centre_crs`. Defaults to 10,000 metres.

        Returns
        -------
        output : gpd.GeoDataFrame
            GeoDataFrame with urban centre, buffer and bbox vector polygon
            boundaries.

        """
        # window raster based on bbox
        self.__windowed_array, self.aff, self.crs = self._window_raster(
            self.file, bbox, band_n
        )

        # cells over pop threshold
        self.__pop_filt_array = self._flag_cells(
            self.__windowed_array, cell_pop_threshold
        )

        # clusters
        self.__cluster_array, self.__num_clusters = self._cluster_cells(
            self.__pop_filt_array, diag
        )

        # clusters over pop threshold
        self.__urban_centres_array = self._check_cluster_pop(
            self.__windowed_array,
            self.__cluster_array,
            self.__num_clusters,
            cluster_pop_threshold,
        )

        # smoothed clusters
        self.__filled_array = self._fill_gaps(
            self.__urban_centres_array, cell_fill_threshold
        )

        # vectorized urban centre
        self.__vectorized_uc = self._vectorize_uc(
            self.__filled_array,
            self.aff,
            self.crs,
            centre,
            centre_crs,
            vector_nodata,
        )

        # buffer
        if not isinstance(buffer_size, int):
            raise TypeError(
                "`buffer_size` expected int, "
                f"got {type(buffer_size).__name__}."
            )
        if buffer_size <= 0:
            raise ValueError(
                "`buffer_size` expected positive non-zero integer"
            )

        self.__buffer = gpd.GeoDataFrame(
            geometry=self.__vectorized_uc.buffer(buffer_size), crs=self.crs
        )

        # bbox
        self.__uc_buffer_bbox = gpd.GeoDataFrame(
            geometry=self.__buffer.envelope, crs=self.crs
        )

        # single GeoDataFrame containing all labelled outputs
        self.output = pd.concat(
            [self.__vectorized_uc, self.__buffer, self.__uc_buffer_bbox],
            axis=0,
        ).reset_index(drop=True)
        self.output["label"] = ["vectorized_uc", "buffer", "bbox"]

        return self.output

    def _window_raster(
        self, file: str, bbox: gpd.GeoDataFrame, band_n: int = 1
    ) -> tuple:
        """Open file, load band and apply mask.

        Parameters
        ----------
        file : str
            Path to geoTIFF file.
        bbox : gpd.GeoDataFrame
            A GeoPandas GeoDataFrame containing boundaries to filter the
            raster.
        band_n : int = 1, optional
            Band number to load from the geoTIFF.

        Returns
        -------
        tuple[0] : numpy.ndarray
            raster, clipped to the extent of the bbox and masked if extent
            does not match the boundaries provided.
        tuple[1] : Affine
            transform matrix for the loaded raster.
        tuple[2] : rasterio.crs.CRS
            crs string from the raster.

        """
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

            _, affine, win = raster_geometry_mask(
                src, bbox.geometry.values, crop=True, all_touched=True
            )

            # band is clipped to extent of bbox
            rst = src.read(band_n, window=win)

            return (rst, affine, src.crs)

    def _flag_cells(
        self, masked_rst: np.ndarray, cell_pop_threshold: int = 1500
    ) -> np.ndarray:
        """Flag cells that are over the threshold.

        Parameters
        ----------
        masked_rst : np.ndarray
            Clipped (and potentially masked) array.
        cell_pop_threshold : int = 1500, optional
            A cell is flagged if its value is equal or higher than the
            threshold.

        Returns
        -------
        flag_array : np.ndarray
            boolean array where cells over the threshold are flagged as
            True.

        Raises
        ------
        ValueError
            If cell_pop_threshold is too high and all cells are filtered out.

        """
        if not isinstance(masked_rst, np.ndarray):
            raise TypeError(
                "`masked_rst` expected numpy array, "
                f"got {type(masked_rst).__name__}."
            )
        if not isinstance(cell_pop_threshold, int):
            raise TypeError(
                "`cell_pop_threshold` expected integer, "
                f"got {type(cell_pop_threshold).__name__}."
            )

        flag_array = masked_rst >= cell_pop_threshold

        if np.sum(flag_array) == 0:
            raise ValueError(
                "`cell_pop_threshold` value too high, "
                "no cells over threshold"
            )

        return flag_array

    def _cluster_cells(
        self, flag_array: np.ndarray, diag: bool = False
    ) -> tuple:
        """Cluster cells based on adjacency.

        Parameters
        ----------
        flag_array : np.ndarray
            Boolean array.
        diag : bool = False, optional
            If True, diagonals are considered as adjacent.

        Returns
        -------
        tuple[0] : np.ndarray
            Array including all clusters, each with an unique label.
        tuple[1] : int
            Number of clusters identified.

        """
        if not isinstance(flag_array, np.ndarray):
            raise TypeError(
                "`flag_array` expected numpy array, "
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

    def _check_cluster_pop(
        self,
        band: np.ndarray,
        labelled_array: np.ndarray,
        num_clusters: int,
        cluster_pop_threshold: int = 50000,
    ) -> np.ndarray:
        """Filter clusters based on total population.

        Checks whether clusters have more than the threshold population and
        changes label for those that don't to 0.

        Parameters
        ----------
        band : np.ndarray
            Original clipped raster with population values.
        labelled_array : np.ndarray
            Array with clusters, each with unique labels.
        num_clusters : int
            Number of unique clusters in the labelled array.
        cluster_pop_threshold : int = 50000, optional
            Threshold to consider inclusion of cluster. If total population in
            cluster is lower than threshold, the cluster label is set to 0.

        Returns
        -------
        urban_centres : np.ndarray
            Array including only clusters with population over the threshold.

        """
        if not isinstance(band, np.ndarray):
            raise TypeError(
                "`band` expected numpy array, " f"got {type(band).__name__}."
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
        if not isinstance(cluster_pop_threshold, int):
            raise TypeError(
                "`cluster_pop_threshold` expected integer, "
                f"got {type(cluster_pop_threshold).__name__}"
            )

        urban_centres = labelled_array.copy()
        for n in range(1, num_clusters + 1):
            total_pop = ma.sum(ma.masked_where(urban_centres != n, band))

            if total_pop < cluster_pop_threshold:
                urban_centres[urban_centres == n] = 0

        if len(urban_centres[urban_centres != 0]) == 0:
            raise ValueError(
                "`cluster_pop_threshold` value too high, "
                "no clusters over threshold"
            )

        return urban_centres

    def _custom_filter(self, win: np.ndarray, threshold: int) -> int:
        """Check gap filling criteria.

        Counts non-zero values within window and if higher than threshold and
        cell is zero returns mode, else returns value of origin cell.

        Parameters
        ----------
        win : np.ndarray
            1-D flattened array of a 3x3 grid, where the centre is
            win[len(win) // 2]. Note that cells outside of the edges are
            filled with 0.
        threshold : int
            Number of cells that need to be filled to change the value of the
            central cell.

        Returns
        -------
        int
            Value to impute to the central cell.

        """
        counter = Counter(win)
        mode_count = counter.most_common(1)[0]
        if (mode_count[1] >= threshold) & (win[len(win) // 2] == 0):
            r = mode_count[0]
        else:
            r = win[len(win) // 2]
        return r

    def _fill_gaps(
        self, urban_centres: np.ndarray, cell_fill_threshold: int = 5
    ) -> np.ndarray:
        """Fill gaps in urban clusters.

        For empty cells, checks if at least 5 adjacent cells belong to cluster,
        and if so fills with cluster value.

        Parameters
        ----------
        urban_centres : np.ndarray
            Array including urban centres, i.e. clusters over the population
            threshold.
        cell_fill_threshold : int = 5, optional
            If the number of cells adjacent to any empty cell belonging to
            a cluster is higher than the threshold, the cell is filled with
            the cluster value.  Needs to be between 5 and 8.

        Returns
        -------
        filled : np.ndarray
            Array including urban centres with gaps filled.

        """
        if not isinstance(urban_centres, np.ndarray):
            raise TypeError(
                "`urban_centres` expected numpy array, "
                f"got {type(urban_centres).__name__}."
            )
        if not isinstance(cell_fill_threshold, int):
            raise TypeError(
                "`cell_fill_threshold` expected integer, "
                f"got {type(cell_fill_threshold).__name__}"
            )
        if not (5 <= cell_fill_threshold <= 8):
            raise ValueError(
                "Wrong value for `cell_fill_threshold`, "
                "please enter value between 5 and 8"
            )

        filled = urban_centres.copy()
        n = 0
        while True:
            n += 1
            check = filled.copy()
            filled = generic_filter(
                filled,
                function=self._custom_filter,
                size=3,
                mode="constant",
                extra_keywords={"threshold": cell_fill_threshold},
            )
            if np.array_equal(filled, check):
                break
        return filled

    def _get_x_y(
        self,
        coords: tuple,
        aff: affine.Affine,
        raster_crs: rasterio.crs.CRS,
        coords_crs: str,
    ) -> tuple:
        """Get array index for given coordinates.

        Parameters
        ----------
        coords : tuple
            Tuple with coordinates to convert.
        aff : affine.Affine
            Affine transform.
        raster_crs : rasterio.crs.CRS
            Valid rasterio crs string.
        coords_crs : str
            CRS code for coordinates provided.

        Returns
        -------
        tuple
            (row, col) position for provided parameters.

        """
        if len(coords) != 2:
            raise ValueError("`coords` expected a tuple of lenght 2.")

        if (not isinstance(coords[0], float)) and (
            not isinstance(coords[1], float)
        ):
            raise TypeError("Elements of `coords` need to be float.")

        transformer = Transformer.from_crs(coords_crs, raster_crs)
        x, y = transformer.transform(*coords)
        row, col = rowcol(aff, x, y)

        return row, col

    def _vectorize_uc(
        self,
        uc_array: np.ndarray,
        aff: affine.Affine,
        raster_crs: rasterio.crs.CRS,
        centre: tuple,
        centre_crs: str = None,
        nodata: int = -200,
    ) -> gpd.GeoDataFrame:
        """Vectorize raster with urban centre polygon.

        Parameters
        ----------
        uc_array : np.ndarray
            Array including filled urban centres.
        aff : affine.Affine
            Affine transform of the masked raster.
        raster_crs : rasterio.crs.CRS
            crs string of the masked raster.
        centre : tuple
            Tuple with coordinates for city centre, used to filter cluster.
        centre_crs : str = None, optional
            crs string of the centre coordinates. If None, it will default
            to raster_crs.
        nodata : int = -200, optional
            Value to fill empty cells.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with polygon boundaries.

        Raises
        ------
        IndexError
            If centre coordinates provided fall outside the raster window.
            This can be caused if coordinates are provided in a tuple in the
            wrong order, or if a wrong crs is provided.
        ValueError
            If centre coordinates are not included within any cluster.

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
        if not isinstance(raster_crs, rasterio.crs.CRS):
            raise TypeError(
                "`raster_crs` must be a valid rasterio.crs.CRS " "object"
            )
        if not isinstance(nodata, int):
            raise TypeError(
                "`nodata` expected integer, " f"got {type(nodata).__name__}"
            )

        if centre_crs is None:
            centre_crs = raster_crs

        row, col = self._get_x_y(centre, aff, raster_crs, centre_crs)
        if row > uc_array.shape[0] or col > uc_array.shape[1]:
            raise IndexError(
                "Coordinates fall outside of raster window. "
                "Did you use the correct x, y order?"
            )

        cluster_num = uc_array[row, col]
        if cluster_num == 0:
            raise ValueError(
                "Coordinates provided are not included within any cluster."
            )

        filt_array = uc_array == cluster_num

        x_array = (
            xr.DataArray(filt_array)
            .astype("int32")
            .rio.write_nodata(nodata)
            .rio.write_transform(aff)
            .rio.set_crs(raster_crs, inplace=True)
        )

        gdf = vectorize(x_array)
        gdf.columns = ["label", "geometry"]
        gdf = gdf[gdf["label"] == 1]

        return gdf
