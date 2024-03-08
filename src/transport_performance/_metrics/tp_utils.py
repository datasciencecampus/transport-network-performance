"""Transport performance helper functions."""
import pathlib
import warnings
from typing import Union

import polars as pl
import geopandas as gpd
import numpy as np
import pandas as pd
from haversine import haversine_vector
from numpy import arccos, cos, radians, sin


def _transport_performance_pandas(
    filepath_or_dirpath: Union[str, pathlib.Path],
    centroids: gpd.GeoDataFrame,
    populations: gpd.GeoDataFrame,
    travel_time_threshold: int = 45,
    distance_threshold: float = 11.25,
    sources_col: str = "from_id",
    destinations_col: str = "to_id",
) -> gpd.GeoDataFrame:
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
    gpd.GeoDataFrame
        Transport performance metrics, grouped by destination column IDs.

    """
    # create local copy before manipulation since `centroids` is a mutable
    # dtype - create pass-by-value effect and won't impact input variable.
    centroids_df = centroids.copy()

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


def _transport_performance_polars(
    filepath_or_dirpath: pathlib.Path,
    centroids: gpd.GeoDataFrame,
    populations: gpd.GeoDataFrame,
    travel_time_threshold: int = 45,
    distance_threshold: float = 11.25,
    sources_col: str = "from_id",
    destinations_col: str = "to_id",
) -> gpd.GeoDataFrame:
    """Calculate transport performance using polars (and some pandas).

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
    gpd.GeoDataFrame
        Transport performance metrics, grouped by destination column IDs.

    """

    def _haversine(lat1, lon1, lat2, lon2):
        """Haversine function for use with polars.

        Description
        -----------
        Return an array of the haversine distance in KM. Assumes coordinates
        are in degrees.
        """
        return 6371 * arccos(
            (sin(radians(lat1)) * sin(radians(lat2)))
            + cos(radians(lat1))
            * cos(radians(lat2))
            * cos(radians(lon2) - radians(lon1))
        )

    # create local copy before manipulation since `centroids` is a mutable
    # dtype - create pass-by-value effect and won't impact input variable.
    centroids_gdf = centroids.copy()
    # make centroid coords individual columns
    centroids_gdf["centroid_x"] = centroids_gdf["centroid"].apply(
        lambda coord: coord.x
    )
    centroids_gdf["centroid_y"] = centroids_gdf["centroid"].apply(
        lambda coord: coord.y
    )
    centroids_gdf.drop("centroid", axis=1, inplace=True)
    # create relevant polars LazyFrame's
    batch_lf = (
        pl.scan_parquet(filepath_or_dirpath)
        .select([sources_col, destinations_col, "travel_time"])
        .lazy()
    )
    pop_lf = pl.from_pandas(populations[["population", "id"]]).lazy()
    centroids_lf = pl.from_pandas(centroids_gdf).lazy()
    # combine for a faster join
    cent_pop_lf = centroids_lf.join(
        pop_lf.select({"id", "population"}), on="id", how="left"
    )
    # merge all datasetss
    merged = (
        batch_lf.select(pl.exclude("within_urban_centre"))
        .join(
            cent_pop_lf.select(
                ["id", "centroid_x", "centroid_y", "population"]
            ),
            left_on=sources_col,
            right_on="id",
            how="left",
        )
        .rename(
            {
                "centroid_x": "from_centroid_x",
                "centroid_y": "from_centroid_y",
                "population": "from_population",
            }
        )
        .join(
            centroids_lf.select(["id", "centroid_x", "centroid_y"]),
            left_on=destinations_col,
            right_on="id",
            how="left",
        )
        .rename({"centroid_x": "to_centroid_x", "centroid_y": "to_centroid_y"})
        .with_columns(
            _haversine(
                pl.col("from_centroid_y"),
                pl.col("from_centroid_x"),
                pl.col("to_centroid_y"),
                pl.col("to_centroid_x"),
            ).alias("dist")
        )
    )
    # calculate accessible and proximity populations for TP
    accessibility = (
        merged.filter(pl.col("dist") <= distance_threshold)
        .filter(pl.col("travel_time") <= travel_time_threshold)
        .group_by(destinations_col)
        .sum()
        .select(destinations_col, "from_population")
        .rename({"from_population": "accessible_population"})
    )

    proximity = (
        merged.filter(pl.col("dist") <= 11.25)
        .group_by(destinations_col)
        .sum()
        .select(destinations_col, "from_population")
        .rename({"from_population": "proximity_population"})
    )
    # calculate TP and covert back to pandas
    perf_df = (
        (
            accessibility.join(
                proximity, on=destinations_col, validate="1:1"
            ).with_columns(
                (
                    pl.col("accessible_population").truediv(
                        pl.col("proximity_population")
                    )
                )
                .mul(100)
                .alias("transport_performance")
            )
        )
        .collect()
        .to_pandas()
    )
    # re-join geometry
    perf_df = perf_df.merge(
        populations[["id", "within_urban_centre", "geometry"]],
        how="left",
        left_on=destinations_col,
        right_on="id",
    )
    return perf_df


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
                f"Unable to calculate the urban centre area in CRS {uc_crs} "
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
