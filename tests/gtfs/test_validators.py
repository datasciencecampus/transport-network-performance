"""Tests for validation module."""
from pyprojroot import here
import pytest
import re

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.validators import (
    validate_travel_between_consecutive_stops,
    validate_travel_over_multiple_stops,
)


@pytest.fixture(scope="function")
def gtfs_fixture():
    """Fixture for test funcs expecting a valid feed object."""
    gtfs = GtfsInstance(here("tests/data/chester-20230816-small_gtfs.zip"))
    return gtfs


class Test_ValidateTravelBetweenConsecutiveStops(object):
    """Tests for the validate_travel_between_consecutive_stops function()."""

    def test_validate_travel_between_consecutive_stops_defences(
        self, gtfs_fixture
    ):
        """Defensive tests for validating travel between consecutive stops."""
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "The validity_df does not exist in as an "
                "attribute of your GtfsInstance object, \n"
                "Did you forget to run the .is_valid() method?"
            ),
        ):
            validate_travel_between_consecutive_stops(gtfs_fixture)
        pass

    def test_validate_travel_between_consecutive_stops(self, gtfs_fixture):
        """General tests for validating travel between consecutive stops."""
        gtfs_fixture.is_valid(far_stops=False)
        validate_travel_between_consecutive_stops(gtfs=gtfs_fixture)

        expected_validation = {
            "type": {
                0: "warning",
                1: "warning",
                2: "warning",
                3: "warning",
                4: "warning",
            },
            "message": {
                0: "Unrecognized column agency_noc",
                1: "Feed expired",
                2: "Unrecognized column platform_code",
                3: "Unrecognized column vehicle_journey_code",
                4: "Fast Travel Between Consecutive Stops",
            },
            "table": {
                0: "agency",
                1: "calendar",
                2: "stops",
                3: "trips",
                4: "full_stop_schedule",
            },
            "rows": {
                0: [],
                1: [],
                2: [],
                3: [],
                4: [457, 458, 4596, 4597, 5788, 5789],
            },
        }

        found_dataframe = gtfs_fixture.validity_df
        assert (
            expected_validation == found_dataframe.to_dict()
        ), "validity_df not as expected."


class Test_ValidateTravelOverMultipleStops(object):
    """Tests for validate_travel_over_multiple_stops()."""

    def test_validate_travel_over_multiple_stops(self, gtfs_fixture):
        """General tests for validate_travel_over_multiple_stops()."""
        gtfs_fixture.is_valid(far_stops=False)
        validate_travel_over_multiple_stops(gtfs=gtfs_fixture)

        expected_validation = {
            "type": {
                0: "warning",
                1: "warning",
                2: "warning",
                3: "warning",
                4: "warning",
                5: "warning",
            },
            "message": {
                0: "Unrecognized column agency_noc",
                1: "Feed expired",
                2: "Unrecognized column platform_code",
                3: "Unrecognized column vehicle_journey_code",
                4: "Fast Travel Between Consecutive Stops",
                5: "Fast Travel Over Multiple Stops",
            },
            "table": {
                0: "agency",
                1: "calendar",
                2: "stops",
                3: "trips",
                4: "full_stop_schedule",
                5: "multiple_stops_invalid",
            },
            "rows": {
                0: [],
                1: [],
                2: [],
                3: [],
                4: [457, 458, 4596, 4597, 5788, 5789],
                5: [0, 1, 2],
            },
        }

        found_dataframe = gtfs_fixture.validity_df

        assert (
            expected_validation == found_dataframe.to_dict()
        ), "validity_df not as expected."
