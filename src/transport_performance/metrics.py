"""Metrics to assess the performance of transport networks."""

import os
import pandas as pd
import geopandas as gpd
import pathlib

from glob import glob
from typing import Optional, Tuple, Union

from transport_performance._metrics.tp_utils import (
    _transport_performance_pandas,
    _transport_performance_stats,
)
from transport_performance.utils.defence import (
    _type_defence,
    _is_expected_filetype,
)


def transport_performance(
    travel_times_path: Union[str, pathlib.Path],
    centroid_gdf: gpd.GeoDataFrame,
    pop_gdf: gpd.GeoDataFrame,
    travel_time_threshold: int = 45,
    distance_threshold: Union[int, float] = 11.25,
    sources_col: str = "from_id",
    destinations_col: str = "to_id",
    backend: str = "pandas",
    descriptive_stats: bool = True,
    urban_centre_name: str = None,
    urban_centre_country: str = None,
    urban_centre_gdf: gpd.GeoDataFrame = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Calculate the transport performance.

    Parameters
    ----------
    travel_times_path : Union[str, pathlib.Path]
        File path or directory path to `analyse_network` output(s). Files must
        be in '.parquet' format.
    centroid_gdf : gpd.GeoDataFrame
        Cell centroid geometry. Output from `RasterPop.get_pop()`.
    pop_gdf : gpd.GeoDataFrame
        Cell geometry with population estimates. Output from
        `RasterPop.get_pop()`.
    travel_time_threshold : int, optional
        Maximum threshold for travel times, by default 45 (minutes). Used when
        calculating accessibility.
    distance_threshold : Union[int, float], optional
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
    descriptive_stats : bool, optional
        Calculate transport performance descriptive statistics and return them
        in a seperate dataframe. By default True, means descriptive statistics
        will be calculated and returned.
    urban_centre_name : str, optional
        The urban centre name, by default None meaning the name will not be
        set. Only considered when `descriptive_stats` is True.
    urban_centre_country: str, optional
        The country in which the urban centre resides, by default None meaning
        the country will not be set. Only considered when `descriptive_stats`
        is True.
    urban_centre_gdf : gpd.GeoDataFrame, optional
        Output from `UrbanCentre`, containg the urban centre geometry
        information. By default None meaning the urban centre area will not be
        calcuated. Only considered when `descriptive_stats` is True.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        The first element of the tuple is the Transport performance metrics
        dataframe, grouped by destination column IDs. When `descriptive_stats`
        is `True` the second element will be the descriptive statistics output,
        otherwise this is `None`.

    Raises
    ------
    ValueError
        When an invalid backend is provided.

    """
    # record valid transport performance backends
    VALID_TP_BACKENDS = ["pandas"]

    # type defences
    type_dict = {
        "travel_times_path": [travel_times_path, (str, pathlib.Path)],
        "centroid_gdf": [centroid_gdf, gpd.GeoDataFrame],
        "pop_gdf": [pop_gdf, gpd.GeoDataFrame],
        "travel_time_threshold": [travel_time_threshold, int],
        "distance_threshold": [distance_threshold, (int, float)],
        "sources_col": [sources_col, str],
        "destinations_col": [destinations_col, str],
        "backend": [backend, str],
        "descriptive_stats": [descriptive_stats, bool],
        "urban_centre_name": [urban_centre_name, (type(None), str)],
        "urban_centre_country": [urban_centre_country, (type(None), str)],
        "urban_centre_gdf": [urban_centre_gdf, (type(None), gpd.GeoDataFrame)],
    }
    for k, v in type_dict.items():
        _type_defence(v[0], k, v[-1])

    # handle travel_times_path file extension checks
    if os.path.splitext(travel_times_path)[1] == "":
        if not os.path.exists(travel_times_path):
            raise FileNotFoundError(f"{travel_times_path} does not exist.")
        files_in_dir = glob(os.path.join(travel_times_path, "*"))
        if len(files_in_dir) == 0:
            raise FileNotFoundError(
                f"No files detected in {travel_times_path}"
            )
        for file in files_in_dir:
            _is_expected_filetype(file, f"{file}", exp_ext=".parquet")
    else:
        _is_expected_filetype(
            travel_times_path, "travel_times_path", exp_ext=".parquet"
        )

    if backend == "pandas":
        # calculate transport performance using pandas
        tp_df = _transport_performance_pandas(
            travel_times_path,
            centroid_gdf,
            pop_gdf,
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

    # handle stats generation, if requested
    if descriptive_stats:
        stats_df = _transport_performance_stats(
            tp_df,
            urban_centre_name=urban_centre_name,
            urban_centre_country=urban_centre_country,
            urban_centre_gdf=urban_centre_gdf,
        )
        return tp_df, stats_df

    return tp_df, None
