"""A set of functions that clean the gtfs data."""
from typing import Union

import numpy as np
import warnings

from gtfs_kit.cleaners import (
    clean_ids as clean_ids_gk,
    clean_route_short_names as clean_route_short_names_gk,
    clean_times as clean_times_gk,
    drop_zombies as drop_zombies_gk,
)

from transport_performance.gtfs.gtfs_utils import _get_validation_warnings
from transport_performance.utils.defence import (
    _gtfs_defence,
    _check_iterable,
    _type_defence,
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

    # _check_iterable only takes lists, therefore convert numpy arrays
    if isinstance(trip_id, np.ndarray):
        trip_id = list(trip_id)

    # ensure trip ids are string
    _check_iterable(
        iterable=trip_id,
        param_nm="trip_id",
        iterable_type=list,
        check_elements=True,
        exp_type=str,
    )

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


def clean_consecutive_stop_fast_travel_warnings(
    gtfs, validate: bool = False
) -> None:
    """Clean 'Fast Travel Between Consecutive Stops' warnings from validity_df.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance to clean warnings within
    validate : bool, optional
        Whether or not to validate the gtfs before carrying out this cleaning
        operation

    Returns
    -------
    None

    """
    # defences
    _gtfs_defence(gtfs, "gtfs")
    if "validity_df" not in gtfs.__dict__.keys() and not validate:
        raise AttributeError(
            "The gtfs has not been validated, therefore no"
            "warnings can be identified. You can pass "
            "validate=True to this function to validate the "
            "gtfs."
        )

    if validate:
        gtfs.is_valid()

    needed_warning = _get_validation_warnings(
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


def clean_multiple_stop_fast_travel_warnings(
    gtfs, validate: bool = False
) -> None:
    """Clean 'Fast Travel Over Multiple Stops' warnings from validity_df.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance to clean warnings within
    validate : bool, optional
        Whether or not to validate the gtfs before carrying out this cleaning
        operation

    Returns
    -------
    None

    """
    # defences
    _gtfs_defence(gtfs, "gtfs")
    if "validity_df" not in gtfs.__dict__.keys() and not validate:
        raise AttributeError(
            "The gtfs has not been validated, therefore no"
            "warnings can be identified. You can pass "
            "validate=True to this function to validate the "
            "gtfs."
        )

    if validate:
        gtfs.is_valid()

    needed_warning = _get_validation_warnings(
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
        clean_ids_gk(gtfs)
    if clean_times:
        clean_times_gk(gtfs)
    if clean_route_short_names:
        clean_route_short_names_gk(gtfs)
    if drop_zombies:
        try:
            drop_zombies_gk(gtfs)
        except KeyError:
            warnings.warn(
                UserWarning(
                    "The drop_zombies cleaner was unable to operate on "
                    "clean_feed as the trips table ahs no sape_id column"
                )
            )
    return None
