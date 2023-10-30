"""A set of functions that clean the gtfs data."""
from typing import Union
import warnings

import numpy as np
import pandas as pd
from gtfs_kit.cleaners import (
    clean_ids as clean_ids_gk,
    clean_route_short_names as clean_route_short_names_gk,
    clean_times as clean_times_gk,
    drop_zombies as drop_zombies_gk,
)

from transport_performance.gtfs.gtfs_utils import (
    _get_validation_warnings,
    _remove_validation_row,
)
from transport_performance.utils.defence import (
    _gtfs_defence,
    _check_iterable,
    _type_defence,
    _check_attribute,
)


def drop_trips(gtfs, trip_id: Union[str, list, np.ndarray]) -> None:
    """Drop trip{s} from a GtfsInstance object.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance object to drop the trip(s) from
    trip_id : Union[str, list, np.ndarray]
        The trip ID(s) of the trip to be dropped from the gtfs data.

    Returns
    -------
    None

    """
    # defences
    _gtfs_defence(gtfs, "gtfs")
    if not isinstance(trip_id, (str, list, np.ndarray)):
        raise TypeError(
            f"'trip_id' received type: {type(trip_id)}. "
            "Expected types: [str, list, np.ndarray]"
        )
    # ensure trip ID is an iterable
    if isinstance(trip_id, str):
        trip_id = [trip_id]

    # ensure trip ids are string
    _check_iterable(
        iterable=trip_id,
        param_nm="trip_id",
        iterable_type=type(trip_id),
        check_elements=True,
        exp_type=str,
    )

    # warn users if passed one of the passed trip_id's is not present in the
    # GTFS.
    for _id in trip_id:
        if _id not in gtfs.feed.trips.trip_id.unique():
            warnings.warn(UserWarning(f"trip_id '{_id}' not found in GTFS"))

    # drop relevant records from tables
    gtfs.feed.trips = gtfs.feed.trips[
        ~gtfs.feed.trips["trip_id"].isin(trip_id)
    ]
    gtfs.feed.stop_times = gtfs.feed.stop_times[
        ~gtfs.feed.stop_times["trip_id"].isin(trip_id)
    ]

    # finish cleaning up deleted trips
    gtfs.feed = gtfs.feed.drop_zombies()

    # re-run so that summaries can be updated
    gtfs.pre_processed_trips = gtfs._preprocess_trips_and_routes()
    return None


def _clean_fast_travel_preperation(gtfs, warning_re: str) -> pd.DataFrame:
    """Prepare to clean fast travel errors.

    At the beggining of both of the fast travel cleaners, the gtfs is type
    checked, attr checked and then warnings are obtained. Because of this, this
     has been functionalised

    Parameters
    ----------
    gtfs : _type_
        The GtfsInstance.
    warning_re : str
        Regex used to obtain warnings.

    Returns
    -------
    pd.DataFrame
        A dataframe containing warnings.

    """
    _gtfs_defence(gtfs, "gtfs")
    _type_defence(warning_re, "warning_re", str)
    _check_attribute(
        gtfs,
        "validity_df",
        message=(
            "The gtfs has not been validated, therefore no"
            "warnings can be identified. You can pass "
            "validate=True to this function to validate the "
            "gtfs."
        ),
    )
    needed_warning = _get_validation_warnings(gtfs, warning_re)
    return needed_warning


def clean_consecutive_stop_fast_travel_warnings(gtfs) -> None:
    """Clean 'Fast Travel Between Consecutive Stops' warnings from validity_df.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance to clean warnings within

    Returns
    -------
    None

    """
    # defences
    needed_warning = _clean_fast_travel_preperation(
        gtfs, "Fast Travel Between Consecutive Stops"
    )

    if len(needed_warning) < 1:
        return None

    trip_ids = gtfs.full_stop_schedule.loc[
        needed_warning[0][3]
    ].trip_id.unique()

    # drop trips from tables
    drop_trips(gtfs=gtfs, trip_id=trip_ids)
    gtfs.full_stop_schedule = gtfs.full_stop_schedule[
        ~gtfs.full_stop_schedule["trip_id"].isin(trip_ids)
    ]
    return None


def clean_multiple_stop_fast_travel_warnings(gtfs) -> None:
    """Clean 'Fast Travel Over Multiple Stops' warnings from validity_df.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance to clean warnings within

    Returns
    -------
    None

    """
    needed_warning = _clean_fast_travel_preperation(
        gtfs, "Fast Travel Over Multiple Stops"
    )
    if len(needed_warning) < 1:
        return None

    trip_ids = gtfs.multiple_stops_invalid.loc[
        needed_warning[0][3]
    ].trip_id.unique()

    # drop trips from tables
    drop_trips(gtfs=gtfs, trip_id=trip_ids)
    gtfs.multiple_stops_invalid = gtfs.multiple_stops_invalid[
        ~gtfs.multiple_stops_invalid["trip_id"].isin(trip_ids)
    ]
    return None


def core_cleaners(
    gtfs,
    clean_ids: bool = True,
    clean_times: bool = True,
    clean_route_short_names: bool = True,
    drop_zombies: bool = True,
) -> None:
    """Clean the gtfs with the core cleaners of gtfs-kit.

    The source code for the cleaners, along with detailed descriptions of the
    cleaning they are performing can be found here:
    https://github.com/mrcagney/gtfs_kit/blob/master/gtfs_kit/cleaners.py

    All credit for these cleaners goes to the creators of the gtfs_kit package.
    HOMEPAGE:  https://github.com/mrcagney/gtfs_kit

    Parameters
    ----------
    gtfs : GtfsInstance
        The gtfs to clean
    clean_ids : bool, optional
        Whether or not to use clean_ids, by default True
    clean_times : bool, optional
        Whether or not to use clean_times, by default True
    clean_route_short_names : bool, optional
        Whether or not to use clean_route_short_names, by default True
    drop_zombies : bool, optional
        Whether or not to use drop_zombies, by default True

    Returns
    -------
    None

    """
    # defences
    _gtfs_defence(gtfs, "gtfs")
    _type_defence(clean_ids, "clean_ids", bool)
    _type_defence(clean_times, "clean_times", bool)
    _type_defence(clean_route_short_names, "clean_route_short_names", bool)
    _type_defence(drop_zombies, "drop_zombies", bool)
    # cleaning
    if clean_ids:
        clean_ids_gk(gtfs.feed)
    if clean_times:
        clean_times_gk(gtfs.feed)
    if clean_route_short_names:
        clean_route_short_names_gk(gtfs.feed)
    if drop_zombies:
        try:
            drop_zombies_gk(gtfs.feed)
        except KeyError:
            warnings.warn(
                UserWarning(
                    "The drop_zombies cleaner was unable to operate on "
                    "clean_feed as the trips table has no shape_id column"
                )
            )
    return None


def clean_unrecognised_column_warnings(gtfs) -> None:
    """Clean warnings for unrecognised columns.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance to clean warnings from

    Returns
    -------
    None

    """
    _gtfs_defence(gtfs, "gtfs")
    warnings = _get_validation_warnings(
        gtfs=gtfs, message="Unrecognized column .*"
    )
    for warning in warnings:
        tbl = gtfs.table_map[warning[2]]
        # parse column from warning message
        column = warning[1].split("column")[1].strip()
        tbl.drop(column, inplace=True, axis=1)
        _remove_validation_row(gtfs, warning[1])
    return None


def clean_duplicate_stop_times(gtfs) -> None:
    """Clean duplicates from stop_times with repeated pair (trip_id, ...

    departure_time.

    Parameters
    ----------
    gtfs : GtfsInstance
        The gtfs to clean

    Returns
    -------
    None

    """
    _gtfs_defence(gtfs, "gtfs")
    warning_re = r".* \(trip_id, departure_time\)"
    # we are only expecting one warning here
    warning = _get_validation_warnings(gtfs, warning_re)
    if len(warning) == 0:
        return None
    warning = warning[0]
    # drop from actual table
    gtfs.table_map[warning[2]].drop_duplicates(
        subset=["arrival_time", "departure_time", "trip_id", "stop_id"],
        inplace=True,
    )
    _remove_validation_row(gtfs, message=warning_re)
    # re-validate with gtfs-kit validator
    gtfs.is_valid({"core_validation": None})
    return None
