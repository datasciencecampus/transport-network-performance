"""Tests for the transport_performance.gtfs.cleaners.py module."""
import pytest
import os
import re

import numpy as np

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.cleaners import (
    drop_trips,
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
