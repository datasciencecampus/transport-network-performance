"""Test GTFS utility functions."""

from pyprojroot import here
import os
import pytest

from transport_performance.gtfs.utils import bbox_filter_gtfs
from transport_performance.gtfs.validation import GtfsInstance


class TestBboxFilterGtfs(object):
    """Test bbox_filter_gtfs."""

    def test_bbox_filter_gtfs_defence(self):
        """Check defensive behaviour for bbox_filter_gtfs."""
        with pytest.raises(
            TypeError, match="Expected string. Found <class 'bool'> : False"
        ):
            bbox_filter_gtfs(units=False)

    def test_bbox_filter_gtfs_writes_as_expected(self, tmpdir):
        """Test bbox_filter_gtfs writes out a filtered GTFS archive."""
        tmp_out = os.path.join(tmpdir, "newport-train-station_gtfs.zip")
        bbox_filter_gtfs(
            in_pth=here("tests/data/newport-20230613_gtfs.zip"),
            out_pth=tmp_out,
            bbox_list=[
                -3.0017783334,
                51.5874718209,
                -2.9964692194,
                51.5907034241,
            ],  # tiny bounding box over newport train station
        )
        assert os.path.exists(
            tmp_out
        ), f"Expected {tmp_out} to exist but it did not."
        # check the output gtfs can be read
        feed = GtfsInstance(gtfs_pth=tmp_out)
        assert isinstance(
            feed, GtfsInstance
        ), f"Expected class `Gtfs_Instance but found: {type(feed)}`"
