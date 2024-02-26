"""Tests for validation module."""
from pyprojroot import here
import pytest
import re
import shutil
import os
import zipfile
import pathlib

import numpy as np

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.validators import (
    validate_travel_between_consecutive_stops,
    validate_travel_over_multiple_stops,
    validate_route_type_warnings,
    validate_gtfs_files,
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
        chest_gtfs_fixture.is_valid(validators={"core_validation": None})
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

    def test__join_max_speed(self, newp_gtfs_fixture):
        """Tests for the _join_max_speed function."""
        newp_gtfs_fixture.is_valid(validators={"core_validation": None})
        # assert route_type's beforehand
        existing_types = newp_gtfs_fixture.feed.routes.route_type.unique()
        assert np.array_equal(
            existing_types, [3, 200]
        ), "Existing route types not as expected."
        # replace 3 with an invlid route_type
        newp_gtfs_fixture.feed.routes.route_type = (
            newp_gtfs_fixture.feed.routes.route_type.apply(
                lambda x: 12345 if x == 200 else x
            )
        )
        new_types = newp_gtfs_fixture.feed.routes.route_type.unique()
        assert np.array_equal(
            new_types, [3, 12345]
        ), "Route types of 200 not replaced correctly"
        # validate and assert a speed bound of 150 is set for these cases
        validate_travel_between_consecutive_stops(newp_gtfs_fixture)
        cases = newp_gtfs_fixture.full_stop_schedule[
            newp_gtfs_fixture.full_stop_schedule.route_type == 12345
        ]
        print(cases.speed_bound)
        assert np.array_equal(
            cases.route_type.unique(), [12345]
        ), "Dataframe filter to cases did not work"
        assert np.array_equal(
            cases.speed_bound.unique(), [200]
        ), "Obtaining max speed for unrecognised route_type failed"


class Test_ValidateTravelOverMultipleStops(object):
    """Tests for validate_travel_over_multiple_stops()."""

    def test_validate_travel_over_multiple_stops(self, chest_gtfs_fixture):
        """General tests for validate_travel_over_multiple_stops()."""
        chest_gtfs_fixture.is_valid(validators={"core_validation": None})
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
        newp_gtfs_fixture.is_valid(validators={"core_validation": None})
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
        ), "Found route_type warnings after cleaning"

    def test_validate_route_type_warnings_creates_warnings(
        self, newp_gtfs_fixture
    ):
        """Test validate_route_type_warnings re-raises route_type warnings."""
        newp_gtfs_fixture.feed.routes[
            "route_type"
        ] = newp_gtfs_fixture.feed.routes["route_type"].apply(
            lambda x: 310030 if x == 200 else 200
        )
        newp_gtfs_fixture.is_valid(
            {"core_validation": None, "validate_route_type_warnings": None}
        )
        new_route_errors = _get_validation_warnings(
            newp_gtfs_fixture, message="Invalid route_type"
        )
        assert (
            len(new_route_errors) == 1
        ), "route_type warnings not found after cleaning"


@pytest.fixture(scope="function")
def create_test_zip(tmp_path) -> pathlib.Path:
    """Create a gtfs zip with invalid files."""
    gtfs_pth = here("tests/data/chester-20230816-small_gtfs.zip")
    # create dir for unzipped gtfs contents
    gtfs_contents_pth = os.path.join(tmp_path, "gtfs_contents")
    os.mkdir(gtfs_contents_pth)
    # extract unzipped gtfs file to new dir
    with zipfile.ZipFile(gtfs_pth, "r") as gtfs_zip:
        gtfs_zip.extractall(gtfs_contents_pth)
    # write some dummy files with test cases
    with open(os.path.join(gtfs_contents_pth, "not_in_spec.txt"), "w") as f:
        f.write("test_date")
    with open(os.path.join(gtfs_contents_pth, "routes.invalid"), "w") as f:
        f.write("test_date")
    # zip contents
    new_zip_pth = os.path.join(tmp_path, "gtfs_zip")
    shutil.make_archive(new_zip_pth, "zip", gtfs_contents_pth)
    full_zip_pth = pathlib.Path(new_zip_pth + ".zip")
    return full_zip_pth


class TestValidateGtfsFile(object):
    """Tests for validate_gtfs_files."""

    def test_validate_gtfs_files_defence(self):
        """Defensive tests for validate_gtfs_files."""
        with pytest.raises(
            TypeError, match="'gtfs' expected a GtfsInstance object.*"
        ):
            validate_gtfs_files(False)

    def test_validate_gtfs_files_on_pass(self, create_test_zip):
        """General tests for validte_gtfs_files."""
        gtfs = GtfsInstance(create_test_zip)
        gtfs.is_valid(validators={"core_validation": None})
        validate_gtfs_files(gtfs)
        # tests for invalid extensions
        warnings = _get_validation_warnings(
            gtfs, r".*files not of type .*txt.*", return_type="dataframe"
        )
        assert (
            len(warnings) == 1
        ), "More warnings than expected for invalid extension"
        assert warnings.loc[3]["message"] == (
            "GTFS zip includes files not of type '.txt'. These files include "
            "['routes.invalid']"
        ), "Warnings not appearing as expected"

        # tests for unrecognised files
        warnings = _get_validation_warnings(
            gtfs,
            r".*files that aren't recognised by the GTFS.*",
            return_type="dataframe",
        )
        assert (
            len(warnings) == 1
        ), "More warnings than expected for not implemented tables"
        assert warnings.loc[4]["message"] == (
            "GTFS zip includes files that aren't recognised by the GTFS "
            "spec. These include ['not_in_spec.txt']"
        )
