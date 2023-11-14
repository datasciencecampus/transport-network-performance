"""Metrics utility functions."""

import pathlib

from typing import Type, Union

from transport_performance.population.rasterpop import RasterPop
from transport_performance.utils.io import from_pickle


def _retrieve_rasterpop(
    pop_or_path: Union[Type[RasterPop], str, pathlib.Path]
) -> Type[RasterPop]:
    """Resolve a pickled/unpickled `RasterPop` instance.

    Parameters
    ----------
    pop_or_path : Union[Type[RasterPop], str, pathlib.Path]
        A `RasterPop` instance, either the raw object or a path to a pickled
        `RasterPop` instance.

    Returns
    -------
    Type[RasterPop]
        The `RasterPop` instance, retrieved from the pickled path if provided.

    """
    # read pickle file if path provided, else return the object back.
    if not isinstance(pop_or_path, RasterPop):
        rp = from_pickle(pop_or_path)
        # handle case when pickled object is not a `RasterPop` instance
        if not isinstance(rp, RasterPop):
            raise TypeError(
                f"Object unpickled from {pop_or_path} is not type `RasterPop`."
                f" Got {type(rp)}."
            )
        return rp

    return pop_or_path
