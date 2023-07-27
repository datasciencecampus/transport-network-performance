"""Classes to handle population data."""


class RasterPop:
    """Prepare raster population inputs for trasport analysis.

    This class is suited to working with rastered population data (e.g.
    gridded population data).

    Parameters
    ----------
    filepath : str
        file path to population data

    Methods
    -------
    get_pop
        Read and preprocess population estimates into a geopandas dataframe.
    plot
        Build static and interactive visualisations of population data. Can
        only use this method once `get_pop` has been called.

    """

    def __init__(self, filepath: str) -> None:
        self.__filepath = filepath

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
