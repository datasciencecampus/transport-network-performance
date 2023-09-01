"""Tests for inserting invalid data into various GTFS tables."""
import pytest
import pandas as pd
import numpy as np

from transport_performance.gtfs.validation import GtfsInstance


@pytest.fixture(scope="function")
def gtfs_fixture():
    """Test GtfsInstance() fixture."""
    gtfs = GtfsInstance()
    return gtfs


@pytest.mark.runinteg
class TestUmatchedIDWarnings(object):
    """Tests for unmatch ID warnings in GTFS data."""

    def test_unmatched_id_warnings_calendar(self, gtfs_fixture):
        """Tests for unmatched IDs in the calendar table."""
        # insert invalid data
        gtfs_fixture.feed.calendar = pd.concat(
            [
                gtfs_fixture.feed.calendar,
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

        expected_message = {
            "type": "error",
            "message": "Invalid service_id; maybe has extra space characters",
            "table": "calendar",
            "rows": [0],
        }

        # ensure raised errors match expected errors
        assert expected_message == gtfs_fixture.is_valid().iloc[0].to_dict(), (
            "GTFS validation failed to identify an unmatched ID in "
            "the calendar table"
        )

    @pytest.mark.runinteg
    def test_unmatched_id_warnings_trips(self, gtfs_fixture):
        """Tests for unmatched IDs in the trips table."""
        # insert invalid data
        gtfs_fixture.feed.trips = pd.concat(
            [
                gtfs_fixture.feed.trips,
                pd.DataFrame(
                    {
                        "service_id": [101],
                        "route_id": [20304],
                        "trip_id": [
                            "VJbedb4cfd0673348e017d42435abbdff3ddacbf89"
                        ],
                        "trip_headsign": ["Newport"],
                        "block_id": [np.nan],
                        "shape_id": [
                            "RPSPc4c99ac6aff7e4648cbbef785f88427a48efa80f"
                        ],
                        "wheelchair_accessible": [0],
                        "trip_direction_name": [np.nan],
                        "vehicle_journey_code": ["VJ109"],
                    }
                ),
            ],
            axis=0,
        )

        expected_errors = {
            "type": {1: "error", 2: "error", 9: "warning"},
            "message": {
                1: "Undefined route_id",
                2: "Undefined service_id",
                9: "Trip has no stop times",
            },
            "table": {1: "trips", 2: "trips", 9: "trips"},
            "rows": {1: [0], 2: [0], 9: [0]},
        }

        found_errors = (
            gtfs_fixture.is_valid()
            .reset_index(drop=True)
            .iloc[[1, 2, 9]]
            .to_dict()
        )

        # ensure raised errors match expected errors
        assert expected_errors == found_errors, (
            "GTFS validation failed to identify an unmatched IDs and "
            "invalid data in the trips table"
        )

    @pytest.mark.runinteg
    def test_unmatched_id_warnings_routes(self, gtfs_fixture):
        """Tests for unmatched IDs in the routes table."""
        # insert invalid data
        gtfs_fixture.feed.routes = pd.concat(
            [
                gtfs_fixture.feed.routes,
                pd.DataFrame(
                    {
                        "service_id": [101],
                        "route_id": [20304],
                        "agency_id": ["OL5060"],
                        "route_short_name": ["X145"],
                        "route_long_name": [np.nan],
                        "route_type": [200],
                    }
                ),
            ],
            axis=0,
        )

        expected_errors = {
            "type": {0: "error", 1: "error", 6: "warning"},
            "message": {
                0: "Invalid route_id; maybe has extra space characters",
                1: "Undefined agency_id",
                6: "Route has no trips",
            },
            "table": {0: "routes", 1: "routes", 6: "routes"},
            "rows": {0: [0], 1: [0], 6: [0]},
        }

        found_errors = (
            gtfs_fixture.is_valid()
            .reset_index(drop=True)
            .iloc[[0, 1, 6]]
            .to_dict()
        )

        # ensure raised errors match expected errors
        assert expected_errors == found_errors, (
            "GTFS validation failed to identify an unmatched IDs and invalid "
            "data in the routes table"
        )
