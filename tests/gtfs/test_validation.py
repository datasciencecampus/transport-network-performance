"""Tests for validation module."""
import pytest
from pyprojroot import here
import gtfs_kit as gk
import pandas as pd
from unittest.mock import patch, call
import os
from geopandas import GeoDataFrame

from heimdall_transport.gtfs.validation import (
    Gtfs_Instance,
    _create_map_title_text,
)


@pytest.fixture(scope="function")  # some funcs expect cleaned feed others dont
def gtfs_fixture():
    """Fixture for test funcs expecting a valid feed object."""
    gtfs = Gtfs_Instance()
    return gtfs


class TestGtfsInstance(object):
    """Tests related to the Gtfs_Instance class."""

    def test_init_defensive_behaviours(self):
        """Testing parameter validation on class initialisation."""
        with pytest.raises(
            TypeError,
            match=r"`gtfs_pth` expected path-like, found <class 'int'>.",
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

    def test_is_valid(self, gtfs_fixture):
        """Assertions about validity_df table."""
        gtfs_fixture.is_valid()
        assert isinstance(gtfs_fixture.validity_df, pd.core.frame.DataFrame)
        assert gtfs_fixture.validity_df.shape == (7, 4)
        exp_cols = pd.Index(["type", "message", "table", "rows"])
        assert (gtfs_fixture.validity_df.columns == exp_cols).all()

    def test_print_alerts_defence(self, gtfs_fixture):
        """Check defensive behaviour of print_alerts()."""
        with pytest.raises(
            AttributeError,
            match=r"is None, did you forget to use `self.is_valid()`?",
        ):
            gtfs_fixture.print_alerts()

    @patch("builtins.print")  # testing print statements
    def test_print_alerts_single_case(self, mocked_print, gtfs_fixture):
        """Check alerts print as expected without truncation."""
        gtfs_fixture.is_valid()
        gtfs_fixture.print_alerts()
        # fixture contains single error
        assert mocked_print.mock_calls == [
            call("Invalid route_type; maybe has extra space characters")
        ]

    @patch("builtins.print")
    def test_print_alerts_multi_case(self, mocked_print, gtfs_fixture):
        """Check multiple alerts are printed as expected."""
        gtfs_fixture.is_valid()
        # fixture contains several warnings
        gtfs_fixture.print_alerts(alert_type="warning")
        assert mocked_print.mock_calls == [
            call("Unrecognized column agency_noc"),
            call("Repeated pair (route_short_name, route_long_name)"),
            call("Unrecognized column stop_direction_name"),
            call("Unrecognized column platform_code"),
            call("Unrecognized column trip_direction_name"),
            call("Unrecognized column vehicle_journey_code"),
        ]

    def test_viz_stops_defence(self, gtfs_fixture):
        """Check defensive behaviours of viz_stops()."""
        with pytest.raises(
            TypeError,
            match="`out_pth` expected path-like, found <class 'bool'>",
        ):
            gtfs_fixture.viz_stops(out_pth=True)
        with pytest.raises(
            TypeError, match="`geoms` expects a string. Found <class 'int'>"
        ):
            gtfs_fixture.viz_stops(out_pth="outputs/somefile.html", geoms=38)
        with pytest.raises(
            ValueError, match="`geoms` must be either 'point' or 'hull."
        ):
            gtfs_fixture.viz_stops(
                out_pth="outputs/somefile.html", geoms="foobar"
            )
        with pytest.raises(
            TypeError,
            match="`geom_crs`.*string or integer. Found <class 'float'>",
        ):
            gtfs_fixture.viz_stops(
                out_pth="outputs/somefile.html", geom_crs=1.1
            )

    @patch("builtins.print")
    def test_viz_stops_point(self, mock_print, tmpdir, gtfs_fixture):
        """Check behaviour of viz_stops when plotting point geom."""
        tmp = os.path.join(tmpdir, "points.html")
        gtfs_fixture.viz_stops(out_pth=tmp)
        assert os.path.exists(tmp)
        # check behaviour when parent directory doesn't exist
        no_parent_pth = os.path.join(tmpdir, "notfound", "points1.html")
        gtfs_fixture.viz_stops(out_pth=no_parent_pth)
        assert os.path.exists(no_parent_pth)
        # check behaviour when not implemented fileext used
        tmp1 = os.path.join(tmpdir, "points2.svg")
        gtfs_fixture.viz_stops(out_pth=tmp1)
        assert mock_print.mock_calls == [
            call(".svg format not implemented. Writing to .html")
        ]
        assert os.path.exists(os.path.join(tmpdir, "points2.html"))

    def test_viz_stops_hull(self, tmpdir, gtfs_fixture):
        """Check viz_stops behaviour when plotting hull geom."""
        tmp = os.path.join(tmpdir, "hull.html")
        gtfs_fixture.viz_stops(out_pth=tmp, geoms="hull")
        assert os.path.exists(tmp)

    def test__create_map_title_text(self):
        """Check helper can cope with non-metric cases."""
        gdf = GeoDataFrame()
        txt = _create_map_title_text(gdf=gdf, units="miles", geom_crs=27700)
        assert txt == (
            "GTFS Stops Convex Hull. Area Calculation for Metric Units Only. "
            "Units Found are in miles."
        )

    def test_get_route_modes(self, gtfs_fixture):
        """Assertions about the table returned by get_route_modes()."""
        gtfs_fixture.get_route_modes()
        assert isinstance(
            gtfs_fixture.route_mode_summary_df, pd.core.frame.DataFrame
        )
        exp_cols = pd.Index(["route_type", "desc", "n_routes", "prop_routes"])
        assert (gtfs_fixture.route_mode_summary_df.columns == exp_cols).all()
