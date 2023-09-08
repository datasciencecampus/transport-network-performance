"""A set of functions that clean the gtfs data."""
from typing import Union

import numpy as np

from transport_performance.utils.defence import _gtfs_defence


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
            f"'trip_id recieved type: {type(trip_id)}."
            "Expected types: [str, list, np.ndarray]"
        )

    # ensure trip ID is an iterable
    if isinstance(trip_id, str):
        trip_id = [trip_id]

    # obtain all releveant IDs
    trip_info = gtfs.feed.trips[gtfs.feed.trips["trip_id"].isin(trip_id)]
    route_id = list(trip_info["route_id"])
    service_id = list(trip_info["service_id"])
    shape_id = list(trip_info["shape_id"])
    stop_ids = gtfs.feed.stop_times[
        gtfs.feed.stop_times["trip_id"].isin(trip_id)
    ]["stop_id"].unique()

    # drop relevant records from tables
    gtfs.feed.trips = gtfs.feed.trips[gtfs.feed.trips["trip_id"] != trip_id]
    gtfs.feed.stops = gtfs.feed.stops[
        ~gtfs.feed.stops["stop_id"].isin(stop_ids)
    ]
    gtfs.feed.calendar = gtfs.feed.calendar[
        ~gtfs.feed.calendar["service_id"].isin(service_id)
    ]
    gtfs.feed.stop_times = gtfs.feed.stop_times[
        ~gtfs.feed.stop_times["trip_id"].isin(trip_id)
    ]
    gtfs.feed.routes = gtfs.feed.routes[
        ~gtfs.feed.routes["route_id"].isin(route_id)
    ]
    gtfs.feed.shapes = gtfs.feed.shapes[
        ~gtfs.feed.shapes["shape_id"].isin(shape_id)
    ]

    # re-run so that summaries can be updated
    gtfs.pre_processed_trips = gtfs._preprocess_trips_and_routes()
    return None


# def clean_consecutive_stop_fast_travel_warnings(gtfs):
#     pass


# def clean_multiple_stop_fast_travel_warnings(gtfs):
#     pass
