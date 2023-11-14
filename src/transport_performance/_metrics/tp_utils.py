"""Transport performance helper functions."""

import geopandas as gpd
import pandas as pd
import pathlib

from haversine import haversine_vector
from typing import Union


def _transport_performance_pandas(
    filepath_or_dirpath: Union[str, pathlib.Path],
    centroids: gpd.GeoDataFrame,
    populations: gpd.GeoDataFrame,
    travel_time_threshold: int = 45,
    distance_threshold: float = 11.25,
    sources_col: str = "from_id",
    destinations_col: str = "to_id",
) -> pd.DataFrame:
    """Calculate transport performance using pandas.

    Parameters
    ----------
    filepath_or_dirpath : Union[str, pathlib.Path]
        File path or directory path to `analyse_network` output(s). Files must
        be in '.parquet' format.
    centroids : gpd.GeoDataFrame
        Populations geodataframe containing centroid geometries
    populations : gpd.GeoDataFrame
        Populations geodataframe containing population data and cell
        geometries.
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

    Returns
    -------
    pd.DataFrame
        Transport performance metrics, grouped by destination column IDs.

    """
    # convert centroid shapley object to tuple
    centroids["centroid_tuple"] = centroids["centroid"].apply(
        lambda coord: (coord.y, coord.x)
    )

    # read in travel time matrix
    tts = pd.read_parquet(filepath_or_dirpath).reset_index(drop=True)

    # merge on centroid coordinates tuples
    tts = (
        tts.merge(
            centroids[["id", "centroid_tuple"]],
            left_on=sources_col,
            right_on="id",
            how="left",
        )
        .drop(["id"], axis=1)
        .rename(columns={"centroid_tuple": "from_centroid_tuple"})
        .merge(
            centroids[["id", "centroid_tuple"]],
            left_on=destinations_col,
            right_on="id",
            how="left",
        )
        .drop(["id"], axis=1)
        .rename(columns={"centroid_tuple": "to_centroid_tuple"})
        .merge(
            populations[["id", "population"]],
            left_on=sources_col,
            right_on="id",
            how="left",
        )
        .drop(["id"], axis=1)
        .rename(columns={"population": "from_population"})
    )

    # calculate distance between centroids
    tts["inter_centroid_distance"] = haversine_vector(
        tts.from_centroid_tuple.to_list(),
        tts.to_centroid_tuple.to_list(),
    )

    # calculate the accessibility - total pop within time and distance
    accessibility = (
        tts[
            (tts.inter_centroid_distance <= distance_threshold)
            & (tts.travel_time <= travel_time_threshold)
        ]
        .groupby(destinations_col)["from_population"]
        .sum()
        .reset_index()
        .rename(columns={"from_population": "accessible_population"})
    )

    # calculate the proximity - total pop within distance
    proximity = (
        tts[(tts.inter_centroid_distance <= distance_threshold)]
        .groupby(destinations_col)["from_population"]
        .sum()
        .reset_index()
        .rename(columns={"from_population": "proximity_population"})
    )

    # merge together to start forming the results
    perf_df = accessibility.merge(
        proximity, on=destinations_col, validate="one_to_one"
    )

    # calculate the transport performance
    perf_df["transport_performance"] = (
        perf_df.accessible_population.divide(perf_df.proximity_population)
        * 100
    )

    # merge on population geospatial data
    perf_gdf = populations.merge(
        perf_df,
        left_on="id",
        right_on=destinations_col,
        how="right",
    ).drop([destinations_col], axis=1)

    return perf_gdf
