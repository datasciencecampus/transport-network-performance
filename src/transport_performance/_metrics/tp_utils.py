"""Transport performance helper functions."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pathlib
import warnings

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
        Populations geodataframe containing centroid geometries.
    populations : gpd.GeoDataFrame
        Populations geodataframe containing population data and cell
        geometries.
    travel_time_threshold : int, optional
        Maximum threshold for travel times, by default 45 (minutes). Used when
        calculating accessibility.
    distance_threshold : float, optional
        Maximum threshold for source/destination distance, by default 11.25
        (km). Used when calculating accessibility and proximity.
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
    # create local binding before manipulation since `centroids` is a mutable
    # dtype - create pass-by-value effect and won't impact input variable.
    centroids_df = centroids

    # convert centroid shapely object to tuple
    centroids_df["centroid_tuple"] = centroids_df["centroid"].apply(
        lambda coord: (coord.y, coord.x)
    )

    # read in travel time matrix
    tts = pd.read_parquet(filepath_or_dirpath).reset_index(drop=True)

    # merge on centroid coordinates tuples
    tts = (
        tts.merge(
            centroids_df[["id", "centroid_tuple"]],
            left_on=sources_col,
            right_on="id",
            how="left",
        )
        .drop(["id"], axis=1)
        .rename(columns={"centroid_tuple": "from_centroid_tuple"})
        .merge(
            centroids_df[["id", "centroid_tuple"]],
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


def _transport_performance_stats(
    tp_df: pd.DataFrame,
    urban_centre_name: str = None,
    urban_centre_country: str = None,
    urban_centre_gdf: gpd.GeoDataFrame = None,
) -> pd.DataFrame:
    """Calculate transport performance descriptive statistics.

    Parameters
    ----------
    tp_df : pd.DataFrame
        Transport performance dataframe, output from
        `_transport_performance_pandas()`, or similar.
    urban_centre_name : str, optional
        The urban centre name, by default None meaning the name will not be
        set.
    urban_centre_country: str, optional
        The country in which the urban centre resides, by default None meaning
        the country will not be set.
    urban_centre_gdf : gpd.GeoDataFrame, optional
        Output from `UrbanCentre`, containg the urban centre geometry
        information. By default None meaning the urban centre area will not be
        calculated.

    Returns
    -------
    pd.DataFrame
        Transport performance descriptive statistics.

    Raises
    ------
    UserWarning
        When the CRS unit of `urban_centre_gdf` is not in meters, and
        reprojection is required in order to calculate the urban centre area.

    """
    # describe columns to include
    DESCRIBE_COLS = ["min", "25%", "50%", "75%", "max"]
    UC_AREA_COL = "urban centre area"
    UC_COUNTRY_COL = "urban centre country"
    UC_NAME_COL = "urban centre name"
    UC_LABEL = "vectorized_uc"

    # instantiate an output columns list - columns will be inserted here
    select_cols = ["urban centre population"]

    # get results dataframe and transpose - reset index to drop column name
    tp_results = (
        pd.DataFrame(tp_df.transport_performance.describe())
        .T[DESCRIBE_COLS]
        .reset_index(drop=True)
    )

    # calculate the urban centre area
    if urban_centre_gdf is not None:
        # copy urban centre geodataframe and set label as axis to simplify
        # area calcuation step (only urban centre, not other geometries)
        uc = urban_centre_gdf.copy()
        uc.set_index("label", inplace=True)

        # handle case where CRS is not in an equal area projection
        crs_units = uc.crs.axis_info[0].unit_name
        uc_crs = uc.crs.to_string()
        if crs_units != "metre":
            warnings.warn(
                f"Unable to calculate the ubran centre area in CRS {uc_crs} "
                f"with units {crs_units}. Reprojecting `urban_centre` onto an "
                "equal area projection (ESRI:54009, mollweide) for the area "
                "calculation step."
            )
            uc.to_crs("ESRI:54009", inplace=True)

        # calculate the urban centre
        tp_results[UC_AREA_COL] = uc.loc[UC_LABEL].geometry.area * 1e-6
        select_cols.insert(0, UC_AREA_COL)

    # add the urban centre country
    if urban_centre_country is not None:
        tp_results.loc[0, UC_COUNTRY_COL] = urban_centre_country
        select_cols.insert(0, UC_COUNTRY_COL)

    # add in a name column - do last, such that it is the first column
    if urban_centre_name is not None:
        tp_results.loc[0, UC_NAME_COL] = urban_centre_name
        select_cols.insert(0, UC_NAME_COL)

    # calculate the total population - set to int64 type (default pandas int)
    tp_results["urban centre population"] = (
        tp_df.population.sum().round().astype(np.int64)
    )

    # reorder columns to improve readability
    tp_results = tp_results[select_cols + DESCRIBE_COLS]

    return tp_results
