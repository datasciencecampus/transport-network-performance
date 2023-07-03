"""Tests for validation module."""
import pytest
from pyprojroot import here
import gtfs_kit as gk
import pandas as pd

from heimdall_transport.gtfs.validation import Gtfs_Instance


class TestGtfsInstance(object):
    """Tests related to the Gtfs_Instance class."""

    def test_init_defensive_behaviours(self):
        """Testing parameter validation on class initialisation."""
        with pytest.raises(
            TypeError,
            match=r"`gtfs_pth` expected a path-like, found <class 'int'>",
        ):
            Gtfs_Instance(gtfs_pth=1)
        with pytest.raises(
            FileExistsError, match=r"doesnt/exist not found on file."
        ):
            Gtfs_Instance(gtfs_pth="doesnt/exist")
        #  a case where file is found but not a zip directory
        with pytest.raises(
            ValueError,
            match=r"`gtfs_pth` expected a zip file extension. Found .pbf",
        ):
            Gtfs_Instance(
                gtfs_pth=here("tests/data/newport-2023-06-13.osm.pbf")
            )
        # handling units
        with pytest.raises(
            TypeError, match=r"`units` expected a string. Found <class 'bool'>"
        ):
            Gtfs_Instance(units=False)
        # non metric units
        with pytest.raises(
            ValueError, match=r"`units` accepts metric only. Found: miles"
        ):
            Gtfs_Instance(units="Miles")  # imperial units not implemented

    def test_init_on_pass(self):
        """Assertions about the feed attribute."""
        gtfs = Gtfs_Instance()
        assert isinstance(gtfs.feed, gk.feed.Feed)
        assert gtfs.feed.dist_units == "m"
        # can coerce to correct distance unit?
        gtfs1 = Gtfs_Instance(units="kilometers")
        assert gtfs1.feed.dist_units == "km"
        gtfs2 = Gtfs_Instance(units="metres")
        assert gtfs2.feed.dist_units == "m"

    def test_is_valid(self):
        """Assertions about validity_df table."""
        gtfs = Gtfs_Instance()
        gtfs.is_valid()
        assert isinstance(gtfs.validity_df, pd.core.frame.DataFrame)
        assert gtfs.validity_df.shape == (7, 4)
        exp_cols = pd.Index(["type", "message", "table", "rows"])
        assert (gtfs.validity_df.columns == exp_cols).all()

    def test_print_alerts_defence(self):
        """Check defensive behaviour of print_alerts()."""
        with pytest.raises(
            AttributeError,
            match=r"is None, did you forget to use `self.is_valid()`?",
        ):
            gtfs = Gtfs_Instance()
            gtfs.print_alerts()
