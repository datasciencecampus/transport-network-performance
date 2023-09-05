"""A set of functions that validate the GTFS data."""
import numpy as np
from haversine import Unit, haversine_vector

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.gtfs_utils import _add_validation_row

# from transport_performance.utils.defence import _gtfs_instance_defence


# a constant containing the max acceptable speed of a route type (vehicle type)
VEHICLE_SPEED_BOUNDS = {
    0: 100,
    1: 150,
    2: 500,
    3: 150,
    4: 80,
    5: 30,
    6: 50,
    7: 50,
    11: 150,
    12: 150,
    200: 120,
}


def validate_travel_between_consecutive_stops(gtfs: GtfsInstance):
    """Validate the travel between consecutive stops in the GTFS data.

    Ensures that a trip is valid by examining the duration and distance
    of a trip. If a vehicle is travelling at an unusual speed, the trip can
    be deemed invalid.
    """
    # defences
    if not isinstance(gtfs, GtfsInstance):
        raise TypeError(
            f"'gtfs' expected type {type(GtfsInstance)} " f"Got {type(gtfs)}"
        )

    gtfs.feed.full_stop_schedule = gtfs.feed.stop_times.merge(
        gtfs.feed.stops[["stop_id", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    )

    stop_sched = gtfs.feed.full_stop_schedule

    # create lat_lon tuple (rationale: np.dstack is more efficient than zip)
    stop_sched["lat_lon"] = [
        (x[0], x[1])
        for x in list(
            np.dstack((stop_sched["stop_lat"], stop_sched["stop_lon"]))[0]
        )
    ]

    # create a new column with the lonn/lat of the following stop
    stop_sched["lead_lat_lon"] = stop_sched["lat_lon"].shift(-1, fill_value=0)

    # TODO: Decide how to handle these cases
    stop_sched["lead_lat_lon"] = stop_sched["lead_lat_lon"].apply(
        lambda x: x if x != 0 else (0, 0)
    )

    # sort values for correct calculations
    stop_sched.sort_values(
        ["trip_id", "stop_sequence"], ascending=True, inplace=True
    )

    # calculate the distance between two stops in meters
    stop_sched["distance"] = haversine_vector(
        list(stop_sched["lat_lon"]),
        list(stop_sched["lead_lat_lon"]),
        unit=Unit.KILOMETERS,
    )

    # flag last stops on trips
    stop_sched["lead_trip_id"] = stop_sched["trip_id"].shift(-1)
    stop_sched["trip_end"] = np.array(
        stop_sched["trip_id"] != stop_sched["lead_trip_id"]
    )

    # calculate time taken to reach next stop
    stop_sched["arrival_time_s"] = (
        stop_sched["arrival_time"]
        .str.split(":")
        .apply(lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
    )

    stop_sched["departure_time_s"] = (
        stop_sched["departure_time"]
        .str.split(":")
        .apply(lambda x: int(x[0]) * 3600 + int(x[1]) * 60 + int(x[2]))
    )

    stop_sched["duration"] = np.subtract(
        stop_sched["departure_time_s"], stop_sched["arrival_time_s"].shift(-1)
    ).abs()

    # clean dataframe
    stop_sched.drop(
        [
            "lat_lon",
            "lead_lat_lon",
            "lead_trip_id",
            "departure_time_s",
            "arrival_time_s",
        ],
        axis=1,
        inplace=True,
    )

    # calculate speed
    stop_sched["speed"] = np.multiply(
        (np.divide(stop_sched["distance"], stop_sched["duration"])), 3600
    )

    # clean rows for last stops of a trip
    stop_sched.loc[
        stop_sched.trip_end, ("speed", "distance", "duration")
    ] = np.nan

    # handle rows with infinite speed
    stop_sched["speed"] = stop_sched["speed"].replace(
        [np.inf, -np.inf], np.nan
    )

    # create a route_type lookup table for trips
    route_type_lkp = gtfs.feed.trips[["route_id", "trip_id"]].merge(
        gtfs.feed.routes[["route_id", "route_type"]], on="route_id", how="left"
    )[["trip_id", "route_type"]]

    stop_sched = stop_sched.merge(route_type_lkp, on="trip_id", how="left")

    # determine whether the speed is within the bounds for that transport mode
    def _join_max_speed(r_type: int) -> int:
        try:
            return VEHICLE_SPEED_BOUNDS[r_type]
        except KeyError:
            return 200

    stop_sched["speed_bound"] = stop_sched["route_type"].apply(
        lambda x: _join_max_speed(r_type=int(x))
    )

    # find the stops that exceed the speed boundary
    invalid_stops = stop_sched[stop_sched["speed"] > stop_sched["speed_bound"]]

    # check if the validity_df table exists
    if "validity_df" not in gtfs.__dict__.keys():
        raise AttributeError(
            "The validity_df does not exist in as an "
            "attribute of your GtfsInstance object, \n"
            "Did you forget to run the .is_valid() method?"
        )

    # check if the impacted rows are 0
    if len(invalid_stops) == 0:
        return invalid_stops

    # add the error to the validation table
    # TODO: After merge add full_stop_schedule to HTML output table keys
    _add_validation_row(
        gtfs=gtfs,
        _type="warning",
        message="Fast Travel Between Consecutive Stops",
        table="full_stop_schedule",
        rows=list(invalid_stops.index),
    )

    return invalid_stops


def validate_travel_over_multiple_stops(gtfs: GtfsInstance) -> None:
    """Validate travel over multiple stops in the GTFS data."""
    # defences
    if not isinstance(gtfs, GtfsInstance):
        raise TypeError(
            f"'gtfs' expected type {type(GtfsInstance)} " f"Got {type(gtfs)}"
        )

    if "full_stop_schedule" not in gtfs.feed.__dict__.keys():
        print(
            "'full_stops_schedule' table not found. Passing GtfsInstance to"
            "validate_travel_between_consecutive_stops() first."
        )
        validate_travel_between_consecutive_stops(gtfs)

    stop_sched = gtfs.feed.full_stop_schedule
    trip_ids = stop_sched.trip_id.unique()

    # sequences = []
    # stop_ids = []
    # departure_times = []

    for trip_id in trip_ids:
        pass
        # trip_schedule = stop_sched[stop_sched.trip_id == trip_id]

    return None
