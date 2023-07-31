"""Validating GTFS data."""
import gtfs_kit as gk
from pyprojroot import here
import pandas as pd
import geopandas as gpd
import folium
import datetime
import numpy as np
import os
import inspect

from transport_performance.gtfs.routes import scrape_route_type_lookup
from transport_performance.utils.defence import (
    _is_gtfs_pth,
    _check_namespace_export,
    _check_parent_dir_exists,
)


def _get_intermediate_dates(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    """Return a list of daily timestamps between two dates.

    Parameters
    ----------
    start : pd.Timestamp
        The start date of the given time period in %Y%m%d format.
    end : pd.Timestamp
        The end date of the given time period in %Y%m%d format.

    Returns
    -------
    list[pd.Timestamp]
        A list of daily timestamps for each day in the time period

    """
    # checks for start and end
    if not isinstance(start, pd.Timestamp):
        raise TypeError(
            "'start' expected type pd.Timestamp."
            f" Recieved type {type(start)}"
        )
    if not isinstance(end, pd.Timestamp):
        raise TypeError(
            "'end' expected type pd.Timestamp." f" Recieved type {type(end)}"
        )
    result = []
    while start <= end:
        result.append(start)
        start = start + datetime.timedelta(days=1)
    return result


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


class GtfsInstance:
    """Create a feed instance for validation, cleaning & visualisation."""

    def __init__(
        self, gtfs_pth=here("tests/data/newport-20230613_gtfs.zip"), units="m"
    ):
        _is_gtfs_pth(pth=gtfs_pth, param_nm="gtfs_pth")

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

    def get_calendar_dates(self):
        """Return the available dates from the GTFS calendar.

        Returns
        -------
        list: Available datestrings within the GTFS calendar.

        """
        self.available_dates = self.feed.get_dates()
        return self.available_dates

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

        try:
            # In cases where no alerts of alert_type are found, KeyError raised
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
        except KeyError:
            print(f"No alerts of type {alert_type} were found.")

        return None

    def clean_feed(self):
        """Attempt to clean feed using `gtfs_kit`."""
        try:
            # In cases where shape_id is missing, keyerror is raised.
            # https://developers.google.com/transit/gtfs/reference#shapestxt
            # shows that shapes.txt is optional file.
            self.feed = self.feed.clean()
        except KeyError:
            print("KeyError. Feed was not cleaned.")

    def viz_stops(
        self, out_pth, geoms="point", geom_crs=27700, create_out_parent=False
    ):
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

        create_out_parent : bool
            Should the parent directory of `out_pth` be created if not found.

        Returns
        -------
        None

        """
        # out_pth defence
        _check_parent_dir_exists(
            pth=out_pth, param_nm="out_pth", create=create_out_parent
        )

        pre, ext = os.path.splitext(out_pth)
        if ext != ".html":
            print(f"{ext} format not implemented. Writing to .html")
            out_pth = os.path.normpath(pre + ".html")

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

        try:
            # map_stops will fail if stop_code not present. According to :
            # https://developers.google.com/transit/gtfs/reference#stopstxt
            # This should be an optional column
            if geoms == "point":
                # viz stop locations
                m = self.feed.map_stops(self.feed.stops["stop_id"])
            elif geoms == "hull":
                # visualise feed, output to file with area est, based on stops
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
        except KeyError:
            print("Key Error. Map was not written.")

    def summarise_days(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Produce a summarised table of route statistics by day of week.

        Parameters
        ----------
        summ_ops : list, optional
            A list of operators used to get a summary of a given day,
            by default [np.min, np.max, np.mean, np.median]
        return_summary : bool, optional
            When True, a summary is returned. When False, route data
            for each date is returned,
            by default True

        Returns
        -------
        pd.DataFrame: A dataframe containing either summarized
                      results or dated route data.

        """
        # return_summary checks
        if not isinstance(return_summary, bool):
            raise TypeError(
                "'return_summary' must be of type boolean."
                f" Found {type(return_summary)} : {return_summary}"
            )
        # summ_ops defence

        if isinstance(summ_ops, list):
            for i in summ_ops:
                if inspect.isfunction(i):
                    if not _check_namespace_export(pkg=np, func=i):
                        raise TypeError(
                            "Each item in `summ_ops` must be a numpy function."
                            f" Found {type(i)} : {i.__name__}"
                        )
                else:
                    raise TypeError(
                        (
                            "Each item in `summ_ops` must be a function."
                            f" Found {type(i)} : {i}"
                        )
                    )
        elif inspect.isfunction(summ_ops):
            if not _check_namespace_export(pkg=np, func=summ_ops):
                raise NotImplementedError(
                    "`summ_ops` expects numpy functions only."
                )
        else:
            raise TypeError(
                "`summ_ops` expects a numpy function or list of numpy"
                f" functions. Found {type(summ_ops)}"
            )

        # create a calendar lookup (one row = one date, rather than date range)
        calendar = self.feed.calendar.copy()
        # convert dates to dt and create a list of dates between them
        calendar["start_date"] = pd.to_datetime(
            calendar["start_date"], format="%Y%m%d"
        )
        calendar["end_date"] = pd.to_datetime(
            calendar["end_date"], format="%Y%m%d"
        )
        calendar["period"] = [
            [start, end]
            for start, end in zip(calendar["start_date"], calendar["end_date"])
        ]

        calendar["date"] = calendar["period"].apply(
            lambda x: _get_intermediate_dates(x[0], x[1])
        )
        # explode the dataframe into daily rows for each service
        #  (between the given time period)
        full_calendar = calendar.explode("date")
        calendar.drop(
            ["start_date", "end_date", "period"], axis=1, inplace=True
        )

        # obtain the day of a given date
        full_calendar["day"] = full_calendar["date"].dt.day_name()
        full_calendar["day"] = full_calendar["day"].apply(
            lambda day: day.lower()
        )

        # reformat the data into a long format and only keep dates
        # where the service is active.
        # this ensures that dates are only kept when the service
        # is running (e.g., Friday)
        melted_calendar = full_calendar.melt(
            id_vars=["date", "service_id", "day"], var_name="valid_day"
        )
        melted_calendar = melted_calendar[melted_calendar["value"] == 1]
        melted_calendar = melted_calendar[
            melted_calendar["day"] == melted_calendar["valid_day"]
        ][["service_id", "day", "date"]]

        # join the dates to the trip information to then join on the
        # route_type from the route table
        trips = self.feed.get_trips().copy()
        routes = self.feed.get_routes().copy()
        dated_trips = trips.merge(melted_calendar, on="service_id", how="left")
        dated_trips_routes = dated_trips.merge(
            routes, on="route_id", how="left"
        )[["route_id", "day", "date", "route_type"]].drop_duplicates()

        # group data into route counts per day
        service_count = (
            dated_trips_routes.groupby(["date", "route_type", "day"])
            .agg(
                {
                    "route_id": "count",
                }
            )
            .reset_index()
        )
        service_count.rename(
            mapper={"route_id": "route_count"}, axis=1, inplace=True
        )
        self.dated_route_counts = service_count.copy()

        # aggregate the to the average number of routes
        # on a given day (e.g., Monday)
        day_service_count = (
            service_count.groupby(["day", "route_type"])
            .agg({"route_count": summ_ops})
            .reset_index()
        )

        # order the days (for plotting future purposes)
        day_order = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        day_service_count["day_order"] = day_service_count["day"].apply(
            lambda x: day_order[x]
        )
        day_service_count.sort_values(
            "day_order", ascending=True, inplace=True
        )
        day_service_count.sort_index(axis=1, inplace=True)
        day_service_count.drop("day_order", inplace=True, axis=1)
        day_service_count = day_service_count.round()
        self.daily_summary = day_service_count.copy()

        if return_summary:
            return self.daily_summary
        return self.dated_route_counts

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
