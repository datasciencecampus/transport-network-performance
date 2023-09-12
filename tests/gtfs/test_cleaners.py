"""Tests for the transport_performance.gtfs.cleaners.py module."""
import pytest
import os
import re

import numpy as np

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.cleaners import (
    drop_trips,
    clean_consecutive_stop_fast_travel_warnings,
    clean_multiple_stop_fast_travel_warnings,
)


@pytest.fixture(scope="function")
def gtfs_fixture():
    """Fixture for tests expecting GtfsInstance()."""
    gtfs = GtfsInstance(
        gtfs_pth=os.path.join(
            "tests", "data", "chester-20230816-small_gtfs.zip"
        )
    )
    return gtfs


class Test_DropTrips(object):
    """Tests for drop_trips()."""

    def test_drop_trips_defences(self, gtfs_fixture):
        """Defensive tests for drop_trips()."""
        # test with int type
        with pytest.raises(
            TypeError,
            match=re.escape(
                "'trip_id' received type: <class 'int'>. "
                "Expected types: [str, list, np.ndarray]"
            ),
        ):
            drop_trips(gtfs_fixture, trip_id=5)
        # test with invalid iterable type
        with pytest.raises(
            TypeError,
            match=re.escape(
                "'trip_id' received type: <class 'tuple'>. "
                "Expected types: [str, list, np.ndarray]"
            ),
        ):
            drop_trips(gtfs_fixture, trip_id=("A7HEF", "AHTESTDATA"))
        # pass a list of non strings
        with pytest.raises(
            TypeError,
            match=re.escape(
                "`trip_id` must contain <class 'str'> only. "
                "Found <class 'bool'> : False"
            ),
        ):
            drop_trips(gtfs_fixture, trip_id=["test", False, "test2"])

    def test_drop_trips_on_pass(self, gtfs_fixture):
        """General tests for drop_trips()."""
        fixture = gtfs_fixture
        trips_to_drop = [
            "VJ48dbdedf89131f2468c4a8d750d45c06dcd3cbf9",
            "VJ15cb559c814f0e81bf12d90c3368f041714a372e",
        ]
        # test that the trip summary is correct
        og_mon_sum_exp = np.array(
            ["monday", 3, 795, 795.0, 795.0, 794], dtype="object"
        )
        og_mon_sum_fnd = fixture.summarise_trips().values[0]
        assert np.array_equal(
            og_mon_sum_exp, og_mon_sum_fnd.astype("object")
        ), ("Test data summary does not " "match expected summary.")
        # ensure that the trips to drop are in the gtfs
        assert (
            len(
                fixture.feed.trips[
                    gtfs_fixture.feed.trips.trip_id.isin(trips_to_drop)
                ]
            )
            == 2
        ), "Trips subject to testing not found in the test data."
        # drop trips
        drop_trips(gtfs_fixture, trips_to_drop)
        # test that the trips have been dropped
        assert (
            len(
                gtfs_fixture.feed.trips[
                    gtfs_fixture.feed.trips.trip_id.isin(trips_to_drop)
                ]
            )
            == 0
        ), "Failed to drop trips."
        # test that the trip summary has updated
        updated_mon_sum_exp = np.array(
            ["monday", 3, 793, 793.0, 793.0, 792], dtype="object"
        )
        found_monday_sum = fixture.summarise_trips().values[0]
        assert np.array_equal(updated_mon_sum_exp, found_monday_sum), (
            "Test data summary does not "
            "match expected summary after "
            "dropping trips. "
        )


class Test_CleanConsecutiveStopFastTravelWarnings(object):
    """Tests for clean_consecutive_stop_fast_travel_warnings()."""

    def test_clean_consecutive_stop_fast_travel_warnings_defence(
        self, gtfs_fixture
    ):
        """Defensive tests forclean_consecutive_stop_fast_travel_warnings()."""
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "The gtfs has not been validated, therefore no"
                "warnings can be identified. You can pass "
                "validate=True to this function to validate the "
                "gtfs."
            ),
        ):
            clean_consecutive_stop_fast_travel_warnings(
                gtfs=gtfs_fixture, validate=False
            )

    def test_clean_consecutive_stop_fast_travel_warnings_on_pass(
        self, gtfs_fixture
    ):
        """General tests for clean_consecutive_stop_fast_travel_warnings()."""
        gtfs_fixture.is_valid(far_stops=True)
        original_validation = {
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
        expected_validation = {
            "type": {0: "warning", 1: "warning", 2: "warning"},
            "message": {
                0: "Unrecognized column agency_noc",
                1: "Unrecognized column platform_code",
                2: "Unrecognized column vehicle_journey_code",
            },
            "table": {0: "agency", 1: "stops", 2: "trips"},
            "rows": {0: [], 1: [], 2: []},
        }

        assert (
            original_validation == gtfs_fixture.validity_df.to_dict()
        ), "Original validity df is not as expected"
        clean_consecutive_stop_fast_travel_warnings(
            gtfs=gtfs_fixture, validate=False
        )
        gtfs_fixture.is_valid()
        assert expected_validation == gtfs_fixture.validity_df.to_dict(), (
            "Validation table is not as expected after cleaning consecutive "
            "stop fast travel warnings"
        )
        # test validation; test gtfs with no warnings
        clean_consecutive_stop_fast_travel_warnings(
            gtfs=gtfs_fixture, validate=True
        )


class Test_CleanMultipleStopFastTravelWarnings(object):
    """Tests for clean_multiple_stop_fast_travel_warnings()."""

    def test_clean_multiple_stop_fast_travel_warnings_defence(
        self, gtfs_fixture
    ):
        """Defensive tests for clean_multiple_stop_fast_travel_warnings()."""
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "The gtfs has not been validated, therefore no"
                "warnings can be identified. You can pass "
                "validate=True to this function to validate the "
                "gtfs."
            ),
        ):
            clean_multiple_stop_fast_travel_warnings(
                gtfs=gtfs_fixture, validate=False
            )

    def test_clean_multiple_stop_fast_travel_warnings_on_pass(
        self, gtfs_fixture
    ):
        """General tests for clean_multiple_stop_fast_travel_warnings()."""
        gtfs_fixture.is_valid(far_stops=True)
        original_validation = {
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
        expected_validation = {
            "type": {0: "warning", 1: "warning", 2: "warning"},
            "message": {
                0: "Unrecognized column agency_noc",
                1: "Unrecognized column platform_code",
                2: "Unrecognized column vehicle_journey_code",
            },
            "table": {0: "agency", 1: "stops", 2: "trips"},
            "rows": {0: [], 1: [], 2: []},
        }

        assert (
            original_validation == gtfs_fixture.validity_df.to_dict()
        ), "Original validity df is not as expected"
        clean_multiple_stop_fast_travel_warnings(
            gtfs=gtfs_fixture, validate=False
        )
        gtfs_fixture.is_valid()
        assert expected_validation == gtfs_fixture.validity_df.to_dict(), (
            "Validation table is not as expected after cleaning consecutive "
            "stop fast travel warnings"
        )
        # test validation; test gtfs with no warnings
        clean_multiple_stop_fast_travel_warnings(
            gtfs=gtfs_fixture, validate=True
        )
