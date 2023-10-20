"""Constants to be used throughout the transport-performance package."""
import transport_performance
from importlib import resources as pkg_resources

PKG_PATH = pkg_resources.files(transport_performance)

# GTFS

# Constant to remove non needed columns from repeated
# pair error information.
# This is a messy method however it is the only
# way to ensure that the error report remains
# dynamic and can adadpt to different tables
# in the GTFS file.
GTFS_UNNEEDED_COLUMNS = {
    "routes": [],
    "agency": ["agency_phone", "agency_lang"],
    "stop_times": [
        "stop_headsign",
        "pickup_type",
        "drop_off_type",
        "shape_dist_traveled",
        "timepoint",
    ],
    "stops": [
        "wheelchair_boarding",
        "location_type",
        "parent_station",
        "platform_code",
    ],
    "calendar_dates": [],
    "calendar": [],
    "trips": [
        "trip_headsign",
        "block_id",
        "shape_id",
        "wheelchair_accessible",
    ],
    "shapes": [],
}

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

# POPULATION

# make attribution location dictionary
ATTR_LOC = {
    "bottom_left": {
        "x": 0.01,
        "y": 0.01,
        "ha": "left",
        "va": "bottom",
    },
    "bottom_right": {
        "x": 0.99,
        "y": 0.01,
        "ha": "right",
        "va": "bottom",
    },
    "top_right": {"x": 0.99, "y": 0.99, "ha": "right", "va": "top"},
    "top_left": {"x": 0.01, "y": 0.99, "ha": "left", "va": "top"},
}
