"""test_integration.py.

Unit tests for testing integration with exernal resources or code. The
intention is these tests won't be part of the main test suite, and will only be
run to check dependencies have not changed.

This test module can be run with the pytest flag --runinteg.
"""
import pytest
import pandas as pd
from pyprojroot import here

from heimdall_transport.gtfs.routes import scrape_route_type_lookup

# import the expected fixtures
lookup_fix = pd.read_pickle(here("tests/data/gtfs/route_lookup.pkl"))


@pytest.mark.runinteg
class TestScrapeRouteTypeLookup(object):
    """Integration tests for scrape_route_type_lookup."""

    def test_lookup_is_stable(self):
        """Check if the tables at the urls have changed content."""
        lookup = scrape_route_type_lookup()
        pd.testing.assert_frame_equal(lookup, lookup_fix)
