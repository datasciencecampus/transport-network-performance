"""Testing routes module."""
import pytest
import pandas as pd

from heimdall_transport.gtfs.routes import scrape_route_type_lookup


def mocked__get_response_text(*args):
    """Mock _get_response_text.

    Returns
    -------
        str: Minimal text representation of url tables.

    """
    k1 = "https://gtfs.org/schedule/reference/"
    v1 = "<td> <br><code>0</code> - Tram."
    k2 = (
        "https://developers.google.com/transit/gtfs/reference/"
        "extended-route-types"
    )
    v2 = """<table class="nice-table">
    <tbody>
      <tr>
        <th>Code</th>
        <th>Description</th>
        <th>Supported</th>
        <th>Examples</th>
      </tr>
      <tr>
        <td><strong>100</strong></td>
        <td><strong>Railway Service</strong></td>
        <td>Yes</td>
        <td>Not applicable (N/A)</td>
      </tr>"""

    return_vals = {k1: v1, k2: v2}
    return return_vals[args[0]]


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
            side_effect=mocked__get_response_text,
        )
        result = scrape_route_type_lookup(extended_schema=False)
        assert isinstance(result, pd.core.frame.DataFrame)
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({"route_type": "0", "desc": "Tram."}, index=[0]),
        )

    def test_table_with_extended_schema(self, mocker):
        """Check return table when extended schema = True."""
        mocker.patch(
            "heimdall_transport.gtfs.routes._get_response_text",
            side_effect=mocked__get_response_text,
        )
        result = scrape_route_type_lookup()
        assert isinstance(result, pd.core.frame.DataFrame)
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame(
                {
                    "route_type": ["0", "100"],
                    "desc": ["Tram.", "Railway Service"],
                },
                index=[0, 1],
            ),
        )