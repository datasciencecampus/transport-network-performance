"""Metrics to assess the performance of transport networks."""

import pandas as pd
import pathlib

from typing import Type, Union

from transport_performance.population.rasterpop import RasterPop
from transport_performance._metrics.metrics_utils import _retrieve_rasterpop
from transport_performance._metrics.tp_utils import (
    _transport_performance_pandas,
)


def transport_performance(
    travel_times_path: Union[str, pathlib.Path],
    population_or_picklepath: Union[Type[RasterPop], str, pathlib.Path],
    travel_time_threshold: int = 45,
    distance_threshold: float = 11.25,
    sources_col: str = "from_id",
    destinations_col: str = "to_id",
    backend: str = "pandas",
) -> pd.DataFrame:
    """Calculate the transport performance.

    Parameters
    ----------
    travel_times_path : Union[str, pathlib.Path]
        File path or directory path to `analyse_network` output(s). Files must
        be in '.parquet' format.
    population_or_picklepath : Union[Type[RasterPop], str, pathlib.Path]
        Population data, either a `RasterPop` instance or a path to a pickled
        `RasterPop` instance. If the `RasterPop` instance is pickled, the file
        must have a '.pkl' or '.pickle' file extension.
    travel_time_threshold : int, optional
        Maximum threshold for travel times, by default 45 (minutes). Used when
        calculating accessibility.
    distance_threshold : float, optional
        Maximum threshold for source/desintiation distance, by default 11.25
        (Km). Used when calculating accessibility and proximity.
    sources_col : str, optional
        The sources column name in the travel time data, by default "from_id".
    destinations_col : str, optional
        The destinations column name in the travel time data, by default
        "to_id".
    backend : str, optional
        The 'backend' to use to calculate transport performance, by default
        "pandas". Must be one of: {"pandas"}.

    Returns
    -------
    pd.DataFrame
        Transport performance metrics, grouped by destination column IDs.

    Raises
    ------
    ValueError
        When an invalid backend is provided.

    """
    # record valid transport performance backends
    VALID_TP_BACKENDS = ["pandas"]

    # parse input into `RasterPop` object
    rp = _retrieve_rasterpop(population_or_picklepath)

    if backend == "pandas":
        # calculate transport performance using pandas
        tp_df = _transport_performance_pandas(
            travel_times_path,
            rp.centroid_gdf,
            rp.pop_gdf,
            travel_time_threshold=travel_time_threshold,
            distance_threshold=distance_threshold,
            sources_col=sources_col,
            destinations_col=destinations_col,
        )
    else:
        # raise an error if an unexpected backend is provided
        raise ValueError(
            f"Got `backend`={backend}. Expected one of: {VALID_TP_BACKENDS}"
        )

    return tp_df
