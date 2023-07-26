"""Test osm_utils module."""
import pytest
from pyprojroot import here
from unittest.mock import patch
import os

from transport_performance.osm.osm_utils import filter_osm


class TestFilterOsm(object):
    """Testing filter_osm()."""

    @patch("builtins.print")
    def test_filter_osm_defense(self, mock_print, mocker):
        """Defensive behaviour for filter_osm."""
        with pytest.raises(
            FileExistsError, match="not/a/pbf/.nosiree not found on file."
        ):
            # file doesnt exist
            filter_osm(pbf_pth="not/a/pbf/.nosiree")
        with pytest.raises(
            ValueError,
            match="`pbf_pth` expected file extension .pbf. Found .zip",
        ):
            # file exists but is not a pbf
            filter_osm(pbf_pth=here("tests/data/newport-20230613_gtfs.zip"))
        with pytest.raises(
            TypeError,
            match="`out_pth` expected path-like, found <class 'bool'>.",
        ):
            # out_pth is not a path_like
            filter_osm(out_pth=False)
        with pytest.raises(
            TypeError, match="`tag_filter` expected boolean. Got <class 'int'>"
        ):
            # check for boolean defense
            filter_osm(tag_filter=1)
        with pytest.raises(
            TypeError,
            match="`install_osmosis` expected boolean. Got <class 'str'>",
        ):
            # check for boolean defense
            filter_osm(install_osmosis="False")
        with pytest.raises(
            ValueError,
            match="box longitude West 1.1 is not smaller than East 1.0",
        ):
            # check for bounding boxes that osmosis won't like - long problem
            filter_osm(bbox=[1.1, 0.0, 1.0, 0.1])
        with pytest.raises(
            ValueError,
            match="box latitude South 0.1 is not smaller than North 0.0",
        ):
            # lat problem
            filter_osm(bbox=[0.0, 0.1, 0.1, 0.0])
        with pytest.raises(
            TypeError,
            match="ox` must contain <class 'float'> only. Found <class 'int'>",
        ):
            # type problems with bbox
            filter_osm(bbox=[0, 1.1, 0.1, 1.2])
        with pytest.raises(
            Exception, match="`osmosis` is not found. Please install."
        ):
            # imitate missing osmosis and install_osmosis is False
            mock_missing_osmosis = mocker.patch(
                "transport_performance.osm.osm_utils.subprocess.run",
                side_effect=FileNotFoundError("No osmosis here..."),
            )
            filter_osm()
        assert (
            mock_missing_osmosis.called
        ), "`mock_missing_osmosis` was not called."

    @pytest.mark.runinteg
    def test_filter_osm_with_osmosis(self, tmpdir):
        """Assertions when osmosis is present."""
        target_pth = os.path.join(tmpdir, "test_output.osm.pbf")
        filter_osm(out_pth=target_pth, install_osmosis=True)
        # assert mock_missing_osmosis.called
        assert os.path.exists(target_pth)
