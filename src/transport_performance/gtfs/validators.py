"""A set of functions that validate the GTFS data."""
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector

from transport_performance.gtfs.gtfs_utils import _add_validation_row
from transport_performance.utils.defence import _gtfs_defence

if TYPE_CHECKING:
    from transport_performance.gtfs.validation import GtfsInstance

# a constant containing the max acceptable speed of a route type (vehicle type)
GTFS_VEHICLE_SPEED_BOUNDS = {
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


def validate_travel_between_consecutive_stops(gtfs: "GtfsInstance"):
    """Validate the travel between consecutive stops in the GTFS data.

    Ensures that a trip is valid by examining the duration and distance
    of a trip. If a vehicle is travelling at an unusual speed, the trip can
    be deemed invalid.
    """
    _gtfs_defence(gtfs, "gtfs")
    # check if the validity_df table exists
    if "validity_df" not in gtfs.__dict__.keys():
        raise AttributeError(
            "The validity_df does not exist in as an "
            "attribute of your GtfsInstance object, \n"
            "Did you forget to run the .is_valid() method?"
        )

    stops = gtfs.feed.stops[["stop_id", "stop_lat", "stop_lon"]].copy()
    stops["lat_lon"] = [
        (x[0], x[1])
        for x in list(np.dstack((stops["stop_lat"], stops["stop_lon"]))[0])
    ]
    stops.drop(["stop_lat", "stop_lon"], axis=1, inplace=True)

    stop_sched = gtfs.feed.stop_times.merge(
        stops,
        on="stop_id",
        how="left",
    )

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
            return GTFS_VEHICLE_SPEED_BOUNDS[r_type]
        except KeyError:
            return 200

    stop_sched["speed_bound"] = stop_sched["route_type"].apply(
        lambda x: _join_max_speed(r_type=int(x))
    )

    gtfs.full_stop_schedule = stop_sched
    # find the stops that exceed the speed boundary
    invalid_stops = stop_sched[stop_sched["speed"] > stop_sched["speed_bound"]]

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


def validate_travel_over_multiple_stops(gtfs: "GtfsInstance") -> None:
    """Validate travel over multiple stops in the GTFS data."""
    # defences
    _gtfs_defence(gtfs, "gtfs")
    if "full_stop_schedule" not in gtfs.__dict__.keys():
        print(
            "'full_stops_schedule' table not found. Passing GtfsInstance to"
            "validate_travel_between_consecutive_stops() first."
        )
        validate_travel_between_consecutive_stops(gtfs)

    stop_sched = gtfs.full_stop_schedule

    # take a list of unique trip ids that have speeds greater than the limit
    invalid_rows = stop_sched[stop_sched["speed"] > stop_sched["speed_bound"]]
    trip_ids = invalid_rows.trip_id.unique()

    needed_sched = stop_sched[stop_sched.trip_id.isin(trip_ids)]

    # create lists to store relevant information
    sequences = []
    durations = []
    avg_speeds = []
    ovr_distances = []
    all_trip_ids = []

    for trip_id in trip_ids:
        # take a subset of the data
        trip = needed_sched[needed_sched.trip_id == trip_id]
        trip = trip[
            [
                "stop_sequence",
                "distance",
                "speed",
                "speed_bound",
                "duration",
                "stop_id",
            ]
        ].values
        FAR_DISTANCE_KM = 10
        MAX_SPEED_KPH = trip[0, 3]

        num_rows = len(trip)

        for end_idx in range(1, num_rows):

            # initialise variable
            cur_avg_speed = -1

            # reset values
            distance_to_end_idx = 0
            overall_duration = 0
            max_distance_hit = False

            # loop through each row above the end index
            for start_idx in range(end_idx - 1, -1, -1):
                distance_to_end_idx += trip[start_idx, 1]
                overall_duration += trip[start_idx, 4]

                # if the cumulative threshold is exceeded
                if distance_to_end_idx >= FAR_DISTANCE_KM:
                    max_distance_hit = True
                    cur_avg_speed = (
                        distance_to_end_idx / overall_duration
                    ) * 3600
                    break

            # ensure that the distance limit is exceeded and the overall speed
            # across the stops is out of bounds
            if cur_avg_speed >= MAX_SPEED_KPH and max_distance_hit:
                sequences.append(tuple([start_idx, end_idx]))
                all_trip_ids.append(trip_id)
                durations.append(overall_duration)
                avg_speeds.append(cur_avg_speed)
                ovr_distances.append(distance_to_end_idx)
                break

    far_stops_df = pd.DataFrame(
        {
            "trip_id": all_trip_ids,
            "stop_sequences": sequences,
            "overall_distance": ovr_distances,
            "avg_speed": avg_speeds,
            "overall_duration": durations,
        }
    )

    # TODO: Add this table to the lookup once gtfs HTML is merged
    gtfs.multiple_stops_invalid = far_stops_df

    if len(gtfs.multiple_stops_invalid) > 0:
        _add_validation_row(
            gtfs=gtfs,
            _type="warning",
            message="Fast Travel Over Multiple Stops",
            table="multiple_stops_invalid",
            rows=list(gtfs.multiple_stops_invalid.index),
        )

    return far_stops_df
