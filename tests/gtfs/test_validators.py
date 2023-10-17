"""Tests for validation module."""
from pyprojroot import here
import pytest
import re

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.validators import (
    validate_travel_between_consecutive_stops,
    validate_travel_over_multiple_stops,
    validate_route_type_warnings,
)
from transport_performance.gtfs.gtfs_utils import _get_validation_warnings


@pytest.fixture(scope="function")
def chest_gtfs_fixture():
    """Fixture for test funcs expecting a valid feed object."""
    gtfs = GtfsInstance(here("tests/data/chester-20230816-small_gtfs.zip"))
    return gtfs


@pytest.fixture(scope="function")
def newp_gtfs_fixture():
    """Fixture for test funcs expecting a valid feed object."""
    gtfs = GtfsInstance(here("tests/data/gtfs/newport-20230613_gtfs.zip"))
    return gtfs


class Test_ValidateTravelBetweenConsecutiveStops(object):
    """Tests for the validate_travel_between_consecutive_stops function()."""

    def test_validate_travel_between_consecutive_stops_defences(
        self, chest_gtfs_fixture
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
            validate_travel_between_consecutive_stops(chest_gtfs_fixture)
        pass

    def test_validate_travel_between_consecutive_stops(
        self, chest_gtfs_fixture
    ):
        """General tests for validating travel between consecutive stops."""
        chest_gtfs_fixture.is_valid(far_stops=False)
        validate_travel_between_consecutive_stops(gtfs=chest_gtfs_fixture)

        expected_validation = {
            "type": {0: "warning", 1: "warning", 2: "warning", 3: "warning"},
            "message": {
                0: "Unrecognized column agency_noc",
                1: "Unrecognized column platform_code",
                2: "Unrecognized column vehicle_journey_code",
                3: "Fast Travel Between Consecutive Stops",
            },
            "table": {
                0: "agency",
                1: "stops",
                2: "trips",
                3: "full_stop_schedule",
            },
            "rows": {
                0: [],
                1: [],
                2: [],
                3: [457, 458, 4596, 4597, 5788, 5789],
            },
        }

        found_dataframe = chest_gtfs_fixture.validity_df
        assert expected_validation == found_dataframe.to_dict(), (
            "'_validate_travel_between_consecutive_stops()' failed to raise "
            "warnings in the validity df"
        )


class Test_ValidateTravelOverMultipleStops(object):
    """Tests for validate_travel_over_multiple_stops()."""

    def test_validate_travel_over_multiple_stops(self, chest_gtfs_fixture):
        """General tests for validate_travel_over_multiple_stops()."""
        chest_gtfs_fixture.is_valid(far_stops=False)
        validate_travel_over_multiple_stops(gtfs=chest_gtfs_fixture)

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
                1: "Unrecognized column platform_code",
                2: "Unrecognized column vehicle_journey_code",
                3: "Fast Travel Between Consecutive Stops",
                4: "Fast Travel Over Multiple Stops",
            },
            "table": {
                0: "agency",
                1: "stops",
                2: "trips",
                3: "full_stop_schedule",
                4: "multiple_stops_invalid",
            },
            "rows": {
                0: [],
                1: [],
                2: [],
                3: [457, 458, 4596, 4597, 5788, 5789],
                4: [0, 1, 2],
            },
        }

        found_dataframe = chest_gtfs_fixture.validity_df

        assert expected_validation == found_dataframe.to_dict(), (
            "'_validate_travel_over_multiple_stops()' failed to raise "
            "warnings in the validity df"
        )


class TestValidateRouteTypeWarnings(object):
    """Tests for valdate_route_type_warnings."""

    def test_validate_route_type_warnings_defence(self, newp_gtfs_fixture):
        """Tests for validate_route_type_warnings on fail."""
        with pytest.raises(
            TypeError, match=r".* expected a GtfsInstance object. Got .*"
        ):
            validate_route_type_warnings(1)
        with pytest.raises(
            AttributeError, match=r".* has no attribute validity_df"
        ):
            validate_route_type_warnings(newp_gtfs_fixture)

    def test_validate_route_type_warnings_on_pass(self, newp_gtfs_fixture):
        """Tests for validate_route_type_warnings on pass."""
        newp_gtfs_fixture.is_valid(False)
        route_errors = _get_validation_warnings(
            newp_gtfs_fixture, message="Invalid route_type"
        )
        assert len(route_errors) == 1, "No route_type warnings found"
        # clean the route_type errors
        newp_gtfs_fixture.is_valid()
        validate_route_type_warnings(newp_gtfs_fixture)
        new_route_errors = _get_validation_warnings(
            newp_gtfs_fixture, message="Invalid route_type"
        )
        assert (
            len(new_route_errors) == 0
        ), "Found route_type errors after cleaning"
