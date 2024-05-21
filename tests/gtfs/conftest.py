"""Test fixtures used throughout GTFS tests."""
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def _EXPECTED_NEWPORT_VALIDITY_DF():
    validity_df = pd.DataFrame(
        {
            "type": [
                "warning",
                "warning",
                "warning",
                "warning",
                "warning",
                "warning",
            ],
            "message": [
                "Unrecognized column agency_noc",
                "Feed expired",
                "Unrecognized column platform_code",
                "Unrecognized column vehicle_journey_code",
                "Fast Travel Between Consecutive Stops",
                "Fast Travel Over Multiple Stops",
            ],
            "table": [
                "agency",
                "calendar",
                "stops",
                "trips",
                "full_stop_schedule",
                "multiple_stops_invalid",
            ],
            "rows": [
                [],
                [],
                [],
                [],
                [457, 458, 4596, 4597, 5788, 5789],
                [0, 1, 2],
            ],
        }
    )
    return validity_df


@pytest.fixture(scope="session")
def _EXPECTED_CHESTER_VALIDITY_DF():
    validity_df = pd.DataFrame(
        {
            "type": [
                "warning",
                "warning",
                "warning",
                "warning",
                "warning",
                "warning",
            ],
            "message": [
                "Unrecognized column agency_noc",
                "Feed expired",
                "Unrecognized column platform_code",
                "Unrecognized column vehicle_journey_code",
                "Fast Travel Between Consecutive Stops",
                "Fast Travel Over Multiple Stops",
            ],
            "table": [
                "agency",
                "calendar",
                "stops",
                "trips",
                "full_stop_schedule",
                "multiple_stops_invalid",
            ],
            "rows": [
                [],
                [],
                [],
                [],
                [457, 458, 4596, 4597, 5788, 5789],
                [0, 1, 2],
            ],
        }
    )
    return validity_df
