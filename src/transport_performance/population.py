"""Classes to handle population data."""

import os
import rasterio as rio
import xarray
import rioxarray

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

        # record the crs of the data source
        with rio.open(filepath) as src:
            self.__crs = src.crs.to_string()

    def get_pop(self, aoi_bounds: Type[Polygon]) -> None:
        """Get population data.

        Parameters
        ----------
        aoi_bounds : Type[Polygon]
            A shapely polygon defining the boundary of the area of interest.

        """
        # read and clip population data to area of interest
        self._xds = self._read_and_clip(aoi_bounds)

    def plot(self) -> None:
        """Plot population data."""
        pass

    def _read_and_clip(self, aoi_bounds: Type[Polygon]) -> xarray.DataArray:
        """Open data and clip to the area of interest boundary.

        Read and clip raster file from disk (more performant). Mask the data
        (such that no data values are nan) and read in all grids that touch
        the aoi boundary.

        Parameters
        ----------
        aoi_bounds : Type[Polygon]
            A shapely polygon defining the boundary of the area of interest.

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

        xds = rioxarray.open_rasterio(self.__filepath, masked=True).rio.clip(
            [aoi_bounds], from_disk=True, all_touched=True
        )
        return xds


class VectorPop:
    """Prepare vector population inputs for trasport analysis.

    This class is suited to working with vectored population data (e.g.
    population data defined within administitive boundaries).

    TODO: add methods and updated documentation.
    """

    def __init__(self) -> None:
        pass
