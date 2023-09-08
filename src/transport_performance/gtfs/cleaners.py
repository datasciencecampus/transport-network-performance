"""A set of functions that clean the gtfs data."""


def drop_trip(gtfs, trip_id: str) -> None:
    """Drop a trip from a GtfsInstance object.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance object to drop the trip from
    trip_id : str
        The trip ID of the trip to be dropped from the gtfs data.

    Returns
    -------
    None

    """
    # obtain all releveant IDs
    trip_info = gtfs.feed.trips[gtfs.feed.trips["trip_id"] == trip_id]
    route_id = list(trip_info["route_id"])[0]
    service_id = list(trip_info["service_id"])[0]
    shape_id = list(trip_info["shape_id"])[0]
    stop_ids = gtfs.feed.stop_times[
        gtfs.feed.stop_times["trip_id"] == trip_id
    ]["stop_id"].unique()

    # drop relevant records from tables
    gtfs.feed.trips = gtfs.feed.trips[gtfs.feed.trips["trip_id"] != trip_id]
    gtfs.feed.stops = gtfs.feed.stops[
        ~gtfs.feed.stops["stop_id"].isin(stop_ids)
    ]
    gtfs.feed.calendar = gtfs.feed.calendar[
        gtfs.feed.calendar["service_id"] != service_id
    ]
    gtfs.feed.stop_times = gtfs.feed.stop_times[
        gtfs.feed.stop_times["trip_id"] != trip_id
    ]
    gtfs.feed.routes = gtfs.feed.routes[
        gtfs.feed.routes["route_id"] != route_id
    ]
    gtfs.feed.shapes = gtfs.feed.shapes[
        gtfs.feed.shapes["shape_id"] != shape_id
    ]

    # re-run so that summaries can be updated
    gtfs.pre_processed_trips = gtfs._preprocess_trips_and_routes()
    return None
