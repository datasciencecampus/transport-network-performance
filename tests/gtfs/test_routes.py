"""Testing routes module."""
import pytest
import pandas as pd

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

    def test_table_without_extended_schema(self, mocker):
        """Check the return object when extended_schema = False."""
        mocker.patch(
            "heimdall_transport.gtfs.routes._get_response_text",
            return_value="<td> <br><code>0</code> - Tram.<br><code>1</code>",
        )
        result = scrape_route_type_lookup(extended_schema=False)
        assert isinstance(result, pd.core.frame.DataFrame)
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({"route_type": "0", "desc": "Tram."}, index=[0]),
        )
