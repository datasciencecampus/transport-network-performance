"""Validation of invalid IDs whilst joining GTFS sub-tables."""

# %%
# imports
import gtfs_kit as gk
from pyprojroot import here
import pandas as pd
import numpy as np

# %%
# initialise my feed from GTFS test data
feed = gk.read_feed(
    here("tests/data/newport-20230613_gtfs.zip"), dist_units="m"
)
feed.validate()

# %%
# calendar test
feed.calendar = pd.concat(
    [
        feed.calendar,
        pd.DataFrame(
            {
                "service_id": [101],
                "monday": [0],
                "tuesday": [0],
                "wednesday": [0],
                "thursday": [0],
                "friday": [0],
                "saturday": [0],
                "sunday": [0],
                "start_date": ["20200104"],
                "end_date": ["20230301"],
            }
        ),
    ],
    axis=0,
)

feed.validate()

# %%
# trips test
feed.trips = pd.concat(
    [
        feed.trips,
        pd.DataFrame(
            {
                "service_id": ["101023"],
                "route_id": ["2030445"],
                "trip_id": ["VJbedb4cfd0673348e017d42435abbdff3ddacbf89"],
                "trip_headsign": ["Newport"],
                "block_id": [np.nan],
                "shape_id": ["RPSPc4c99ac6aff7e4648cbbef785f88427a48efa80f"],
                "wheelchair_accessible": [0],
                "trip_direction_name": [np.nan],
                "vehicle_journey_code": ["VJ109"],
            }
        ),
    ],
    axis=0,
)

feed.validate()

# %%
# routes test
feed.routes = pd.concat(
    [
        feed.routes,
        pd.DataFrame(
            {
                "route_id": ["20304"],
                "agency_id": ["OL5060"],
                "route_short_name": ["X145"],
                "route_long_name": [np.nan],
                "route_type": [200],
            }
        ),
    ],
    axis=0,
)

feed.validate()

# OUTCOME
# It appears that 'errors' are recognised when there is an attempt to validate
# the gtfs data using the pre-built gtfs_kit functions.
# This suggests that if the GTFS data is flawed, it will be identified within
# the pipeline and therefore the user will be made aware. It is also flagged
# as an error which means that 'the GTFS is violated'
# (https://mrcagney.github.io/gtfs_kit_docs/).

# %%
