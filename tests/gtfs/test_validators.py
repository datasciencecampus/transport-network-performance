"""Tests for validation module."""
import re
import pytest
from pyprojroot import here

import pandas as pd

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

    def test_validate_travel_between_consecutive_stops(
        self, gtfs_fixture, _EXPECTED_CHESTER_VALIDITY_DF
    ):
        """General tests for validating travel between consecutive stops."""
        gtfs_fixture.is_valid(far_stops=False)
        validate_travel_between_consecutive_stops(gtfs=gtfs_fixture)
        # This assertion should not contain the final row of the chester
        # fixture, which is created on validate_travel_over_multiple_stops()
        _expected_chester_valid_df = _EXPECTED_CHESTER_VALIDITY_DF.loc[
            "Fast Travel Over Multiple Stops"
            != _EXPECTED_CHESTER_VALIDITY_DF["message"]
        ]
        pd.testing.assert_frame_equal(
            _expected_chester_valid_df, gtfs_fixture.validity_df
        )


class Test_ValidateTravelOverMultipleStops(object):
    """Tests for validate_travel_over_multiple_stops()."""

    def test_validate_travel_over_multiple_stops(
        self, gtfs_fixture, _EXPECTED_CHESTER_VALIDITY_DF
    ):
        """General tests for validate_travel_over_multiple_stops()."""
        gtfs_fixture.is_valid(far_stops=False)
        validate_travel_over_multiple_stops(gtfs=gtfs_fixture)
        pd.testing.assert_frame_equal(
            _EXPECTED_CHESTER_VALIDITY_DF, gtfs_fixture.validity_df
        )
