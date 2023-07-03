"""Tests for validation module."""
import pytest
from pyprojroot import here

from heimdall_transport.gtfs.validation import Gtfs_Instance


class TestGtfsInstance(object):
    """Tests related to the Gtfs_Instance class."""

    def test_init_defensive_behaviours(
        self, fix_gtfs_pth=here("tests/data/newport-20230613_gtfs.zip")
    ):
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
        with pytest.raises(
            TypeError, match=r"`units` expected a string. Found <class 'bool'>"
        ):
            Gtfs_Instance(gtfs_pth=fix_gtfs_pth, units=False)
        # non metric units
        with pytest.raises(
            ValueError, match=r"`units` accepts metric only. Found: miles"
        ):
            Gtfs_Instance(gtfs_pth=fix_gtfs_pth, units="Miles")
