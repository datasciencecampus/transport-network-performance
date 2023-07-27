"""Classes to handle population data."""

import os
import rasterio as rio

from typing import Union


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

    def get_pop(self) -> None:
        """Get population data."""
        pass

    def plot(self) -> None:
        """Plot population data."""
        pass


class VectorPop:
    """Prepare vector population inputs for trasport analysis.

    This class is suited to working with vectored population data (e.g.
    population data defined within administitive boundaries).

    TODO: add methods and updated documentation.
    """

    pass
