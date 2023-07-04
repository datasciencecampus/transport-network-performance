"""Validating GTFS data."""
import gtfs_kit as gk
from pyprojroot import here
import pandas as pd
import geopandas as gpd
import folium
from datetime import datetime
import numpy as np
import os
import pathlib

from heimdall_transport.gtfs.routes import scrape_route_type_lookup


def _path_like_defence(pth, param_nm):
    """Handle path-like parameter values.

    Parameters
    ----------
    pth : (str, pathlib.PosixPath)
        The path to check.

    param_nm : str
        The name of the parameter being tested.

    Raises
    ------
    TypeError: `pth` is not either of string or pathlib.PosixPath.

    Returns
    -------
    None

    """
    if not isinstance(pth, (str, pathlib.PosixPath)):
        raise TypeError(f"`{param_nm}` expected path-like, found {type(pth)}.")


def _create_map_title_text(gdf, units, geom_crs):
    """Generate the map title text when plotting convex hull.

    Parameters
    ----------
    gdf :  gpd.GeoDataFrame
        GeoDataFrame containing the spatial features.
    units :  str
        Distance units of the GTFS feed from which `gdf` originated.
    geom_crs : (str, int):
        The geometric crs to use in reprojecting the data in order to
        calculate the area of the hull polygon.

    Returns
    -------
    str : The formatted text string for presentation in the map title.

    """
    if units in ["m", "km"]:
        hull_km2 = gdf.to_crs(geom_crs).area
        if units == "m":
            hull_km2 = hull_km2 / 1000000
        pre = "GTFS Stops Convex Hull Area: "
        post = " nearest km<sup>2</sup>."
        txt = f"{pre}{int(round(hull_km2[0], 0)):,}{post}"
    else:
        txt = (
            "GTFS Stops Convex Hull. Area Calculation for Metric "
            f"Units Only. Units Found are in {units}."
        )
    return txt


class Gtfs_Instance:
    """Create a feed instance for validation, cleaning & visualisation."""

    def __init__(
        self, gtfs_pth=here("tests/data/newport-20230613_gtfs.zip"), units="m"
    ):
        _path_like_defence(pth=gtfs_pth, param_nm="gtfs_pth")
        if not os.path.exists(gtfs_pth):
            raise FileExistsError(f"{gtfs_pth} not found on file.")

        ext = os.path.splitext(gtfs_pth)[-1]
        if ext != ".zip":
            raise ValueError(
                f"`gtfs_pth` expected a zip file extension. Found {ext}"
            )

        # validate units param
        if not isinstance(units, str):
            raise TypeError(f"`units` expected a string. Found {type(units)}")

        units = units.lower().strip()
        if units in ["metres", "meters"]:
            units = "m"
        elif units in ["kilometers", "kilometres"]:
            units = "km"
        accepted_units = ["m", "km"]

        if units not in accepted_units:
            raise ValueError(f"`units` accepts metric only. Found: {units}")

        self.feed = gk.read_feed(gtfs_pth, dist_units=units)

    def is_valid(self):
        """Check a feed is valid with `gtfs_kit`.

        Returns
        -------
            pd.core.frame.DataFrame: Table of errors, warnings & their
            descriptions.

        """
        self.validity_df = self.feed.validate()
        return self.validity_df

    def print_alerts(self, alert_type="error"):
        """Print validity errors & warnins messages in full.

        Parameters
        ----------
        alert_type : str, optional
                The alert type to print messages. Defaults to "error". Also
                accepts "warning".

        Returns
        -------
        None

        """
        if not hasattr(self, "validity_df"):
            raise AttributeError(
                "`self.validity_df` is None, did you forget to use "
                "`self.is_valid()`?"
            )

        msgs = (
            self.validity_df.set_index("type")
            .sort_index()
            .loc[alert_type]["message"]
        )

        # multiple errors
        if isinstance(msgs, pd.core.series.Series):
            for m in msgs:
                print(m)
        # case where single error
        elif isinstance(msgs, str):
            print(msgs)

    def clean_feed(self):
        """Attempt to clean feed using `gtfs_kit`."""
        self.feed = self.feed.clean()

    def viz_stops(self, out_pth, geoms="point", geom_crs=27700):
        """Visualise the stops on a map as points or convex hull. Writes file.

        Parameters
        ----------
        out_pth : str
            Path to write the map file html document to, including the file
            name. Must end with '.html' file extension.

        geoms : str
            Type of map to plot. If `geoms=point` (the default) uses `gtfs_kit`
            to map point locations of available stops. If `geoms=hull`,
            calculates the convex hull & its area. Defaults to "point".

        geom_crs : (str, int)
            Geometric CRS to use for the calculation of the convex hull area
            only. Defaults to "27700" (OSGB36, British National Grid).

        Returns
        -------
        None

        """
        # out_pth defence
        _path_like_defence(out_pth, param_nm="out_pth")

        pre, ext = os.path.splitext(out_pth)
        if ext != ".html":
            print(f"{ext} format not implemented. Writing to .html")
            out_pth = os.path.normpath(pre + ".html")

        parent_dir = os.path.dirname(out_pth)
        if not os.path.exists(parent_dir):
            # create parent directory
            os.mkdir(parent_dir)

        # geoms defence
        if not isinstance(geoms, str):
            raise TypeError(f"`geoms` expects a string. Found {type(geoms)}")
        geoms = geoms.lower().strip()
        accept_vals = ["point", "hull"]
        if geoms not in accept_vals:
            raise ValueError("`geoms` must be either 'point' or 'hull.'")

        # geom_crs defence
        if not isinstance(geom_crs, (str, int)):
            raise TypeError(
                f"`geom_crs` expects string or integer. Found {type(geom_crs)}"
            )

        if geoms == "point":
            # viz stop locations
            m = self.feed.map_stops(self.feed.stops["stop_id"])
        elif geoms == "hull":
            # visualise feed, output to file with area est, based on stop locs
            gtfs_hull = self.feed.compute_convex_hull()
            gdf = gpd.GeoDataFrame(
                {"geometry": gtfs_hull}, index=[0], crs="epsg:4326"
            )
            units = self.feed.dist_units
            # prepare the map title
            txt = _create_map_title_text(gdf, units, geom_crs)

            title_pre = "<h3 align='center' style='font-size:16px'><b>"
            title_html = f"{title_pre}{txt}</b></h3>"

            gtfs_centroid = self.feed.compute_centroid()
            m = folium.Map(
                location=[gtfs_centroid.y, gtfs_centroid.x], zoom_start=5
            )
            geo_j = gdf.to_json()
            geo_j = folium.GeoJson(
                data=geo_j, style_function=lambda x: {"fillColor": "red"}
            )
            geo_j.add_to(m)
            m.get_root().html.add_child(folium.Element(title_html))
        m.save(out_pth)

    def summarise_weekday(self, summ_ops=[np.min, np.max, np.mean, np.median]):
        """Produce a table of summary stats by weekday / weekend.

        This method is expensive & triggers a PerformanceWarning from pandas.

        Parameters
        ----------
        summ_ops :  list, optional
            The numpy summary operations to use. Defaults to
            [np.min, np.max, np.mean, np.median].

        Returns
        -------
        pd.core.frame.DataFrame: Summary table of the no. of routes, no. of
        trips, service distance & service duration.

        """
        available_dates = self.feed.get_dates()
        trip_stats = gk.trips.compute_trip_stats(self.feed)
        # next step is costly and comes with some pd warnings
        feed_stats = self.feed.compute_feed_stats(trip_stats, available_dates)
        # get datetime col
        feed_stats["date"] = pd.to_datetime(
            feed_stats["date"], format="%Y%m%d"
        )
        weekend = ["Saturday", "Sunday"]
        feed_stats["is_weekend"] = [
            datetime.strftime(x, "%A") in weekend for x in feed_stats["date"]
        ]
        # grouped is_weekend summary table, operation parameter
        keep_cols = [
            "num_routes",
            "num_trips",
            "service_distance",
            "service_duration",
        ]
        feed_stats.groupby("is_weekend")[keep_cols].agg(summ_ops)
        self.weekday_stats = feed_stats
        return self.weekday_stats

    def get_route_modes(self):
        """Summarise the available routes by their associated `route_type`.

        Returns
        -------
            pd.core.frame.DataFrame: Summary table of route counts by transport
            mode.

        """
        # Get the available modalities
        lookup = scrape_route_type_lookup()
        gtfs_route_types = [
            str(x) for x in self.feed.routes["route_type"].unique()
        ]
        # Get readable route_type descriptions
        out_tab = lookup[
            lookup["route_type"].isin(gtfs_route_types)
        ].reset_index(drop=True)
        out_tab["n_routes"] = (
            self.feed.routes["route_type"]
            .value_counts()
            .reset_index(drop=True)
        )
        out_tab["prop_routes"] = (
            self.feed.routes["route_type"]
            .value_counts(normalize=True)
            .reset_index(drop=True)
        )
        self.route_mode_summary_df = out_tab
        return self.route_mode_summary_df
