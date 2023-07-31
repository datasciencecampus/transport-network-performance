"""Classes to handle population data."""

import geopandas as gpd
import os
import numpy as np
import rasterio as rio
import xarray
import rioxarray

from geocube.vector import vectorize
from typing import Union, Type
from shapely.geometry.polygon import Polygon


class RasterPop:
    """Prepare raster population inputs for trasport analysis.

    This class is suited to working with rastered population data (e.g.
    gridded population data).

    Parameters
    ----------
    filepath : Union[str, bytes, os.PathLike]
        file path to population data

    Methods
    -------
    get_pop
        Read and preprocess population estimates into a geopandas dataframe.
    plot
        Build static and interactive visualisations of population data. Can
        only use this method once `get_pop` has been called.

    """

    def __init__(self, filepath: Union[str, bytes, os.PathLike]) -> None:

        # defend against cases where input is not a file and does not exist
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{filepath} is not a file.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Unable to find {filepath}.")

        self.__filepath = filepath

        # record the crs of the data source without reading in data
        with rio.open(filepath) as src:
            self.__crs = src.crs.to_string()

    def get_pop(
        self,
        aoi_bounds: Type[Polygon],
        aoi_crs: str = None,
        round: bool = False,
        threshold: int = None,
        var_name: str = "population",
    ) -> None:
        """Get population data.

        Parameters
        ----------
        aoi_bounds : Type[Polygon]
            A shapely polygon defining the boundary of the area of interest.
            Assumed to be in the same CRS as the rastered population data. If
            it is different, set `aoi_crs` to the CRS of the boundary.
        aoi_crs : str, optional
            CRS string for `aoi_bounds` (e.g. "EPSG:4326"), by default None
            which means it is assumed to have the same CRS as `aoi_bounds`.
        round : bool, optional
            Round population estimates to the nearest whole integer.
        threshold : int, optional
            Threshold population estimates, where values below the set
            threshold will be set to nan, by default None which means no
            thresholding will occur.
        var_name : str, optional
            The variable name corresponding to the data's measurement, by
            default "population"

        """
        # read and clip population data to area of interest
        self._xds = self._read_and_clip(aoi_bounds, aoi_crs, var_name)

        # round population estimates, if requested
        self.__round = round
        if round:
            self._round_population()

        # threshold the population data, if requested
        if threshold is not None:
            self._threshold_population(threshold)

        self._to_geopandas()

    def plot(self) -> None:
        """Plot population data."""
        pass

    def _read_and_clip(
        self,
        aoi_bounds: Type[Polygon],
        aoi_crs: str = None,
        var_name: str = "population",
    ) -> xarray.DataArray:
        """Open data and clip to the area of interest boundary.

        Read and clip raster file from disk (more performant). Mask the data
        (such that no data values are nan) and read in all grids that touch
        the aoi boundary.

        Parameters
        ----------
        aoi_bounds : Type[Polygon]
            A shapely polygon defining the boundary of the area of interest.
        aoi_crs : str, optional
            CRS string for `aoi_bounds` (e.g. "EPSG:4326"), by default None
            which means it is assumed to have the same CRS as `aoi_bounds`.
        var_name : str, optional
            The variable name corresponding to the data's measurement, by
            default "population"

        Returns
        -------
        xarray.DataArray
            Clipped data to area of interest.

        """
        # defend against case where aoi_bounds is not a shapely polygon
        if not isinstance(aoi_bounds, Polygon):
            raise TypeError(
                f"Expected type {Polygon.__name__} for `aoi_bounds`, "
                f"got {type(aoi_bounds).__name__}."
            )

        # convert aoi bounds CRS if needed
        if aoi_crs is not None:
            gdf = gpd.GeoDataFrame(geometry=[aoi_bounds], crs=aoi_crs)
            gdf = gdf.to_crs(self.__crs)
            aoi_bounds = gdf.loc[0, "geometry"]

        xds = rioxarray.open_rasterio(self.__filepath, masked=True).rio.clip(
            [aoi_bounds], from_disk=True, all_touched=True
        )

        # set the variable name
        xds.name = var_name
        self.__var_name = var_name

        return xds

    def _round_population(self) -> None:
        """Round population data."""
        self._xds = np.rint(self._xds)

    def _threshold_population(self, threshold: int) -> None:
        """Threshold population data."""
        self._xds = self._xds.where(self._xds >= threshold)

    def _to_geopandas(self) -> None:
        """Convert to geopandas dataframe."""
        # vectorise to geopandas dataframe, converting datatype for `vectorize`
        if self.__round:
            set_type = np.int32
        else:
            set_type = np.float32
        # squeeze needed since shape is (1xnxm) (1 band in tiff file)
        self.var_gdf = vectorize(self._xds.squeeze(axis=0).astype(set_type))

        # dropna to remove nodata regions and those below threshold (if set)
        self.var_gdf = self.var_gdf.dropna(subset=self.__var_name).reset_index(
            drop=True
        )

        # add an id for each cell
        self.var_gdf["id"] = np.arange(0, len(self.var_gdf.index))

        # re-order columns for consistency
        self.var_gdf = self.var_gdf[["id", self.__var_name, "geometry"]]

        # create centroid geodataframe using only necessary columns
        self.centroid_gdf = self.var_gdf[["id", "geometry"]].copy()
        self.centroid_gdf["centroid"] = self.centroid_gdf.geometry.centroid
        self.centroid_gdf.set_geometry("centroid", inplace=True)
        self.centroid_gdf.drop("geometry", axis=1, inplace=True)

        # convert centroids to EPSG:4326 for use in r5py
        if self.__crs != "EPSG:4326":
            self.centroid_gdf = self.centroid_gdf.to_crs("EPSG:4326")


class VectorPop:
    """Prepare vector population inputs for trasport analysis.

    This class is suited to working with vectored population data (e.g.
    population data defined within administitive boundaries).

    TODO: add methods and updated documentation.
    """

    def __init__(self) -> None:
        raise NotImplementedError("This class has not yet been implemented.")
