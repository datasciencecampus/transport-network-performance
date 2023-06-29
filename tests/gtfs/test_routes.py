"""Testing routes module."""
import pytest

from heimdall_transport.gtfs.routes import scrape_route_type_lookup


class TestScrapeRouteTypeLookup(object):
    """Test scrape_route_type_lookup."""

    def test_defensive_exceptions(self):
        """Test the defensive checks raise as expected."""
        with pytest.raises(
            TypeError,
            match=r"url 1 expected string, instead got <class 'int'>",
        ):
            scrape_route_type_lookup(gtfs_url=1)
        with pytest.raises(
            TypeError,
            match=r"url False expected string, instead got <class 'bool'>",
        ):
            scrape_route_type_lookup(ext_spec_url=False)
        with pytest.raises(
            ValueError,
            match="url string expected protocol, instead found foobar",
        ):
            scrape_route_type_lookup(gtfs_url="foobar")
        with pytest.raises(
            TypeError,
            match=r"`extended_schema` expected boolean. Got <class 'str'>",
        ):
            scrape_route_type_lookup(extended_schema="True")
