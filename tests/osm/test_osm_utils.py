"""Test osm_utils module."""
import pytest
from pyprojroot import here

from transport_performance.osm.osm_utils import filter_osm


class TestFilterOsm(object):
    """Testing filter_osm()."""

    def test_filter_osm_defense(self):
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
