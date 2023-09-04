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
    _is_expected_filetype,
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
        converted_gdf = gdf.to_crs(geom_crs)
        hull_area = converted_gdf.area
        crs_unit = converted_gdf.crs.axis_info[0].unit_name
        if crs_unit == "metre":
            hull_area = hull_area / 1000000
        if crs_unit not in ["kilometre", "metre"]:
            raise ValueError(
                "CRS units for convex hull not recognised. "
                "Recognised units include: ['metre', 'kilometre']. "
                f" Found '{crs_unit}.'"
            )
        pre = "GTFS Stops Convex Hull Area: "
        post = " nearest km<sup>2</sup>."
        txt = f"{pre}{int(round(hull_area[0], 0)):,}{post}"
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
        _is_expected_filetype(pth=gtfs_pth, param_nm="gtfs_pth")

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

    def _order_dataframe_by_day(
        self, df: pd.DataFrame, day_column_name: str = "day"
    ) -> pd.DataFrame:
        """Order a dataframe by days of the week in real-world order.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing a column with the name
            of the day of a record
        day_column_name : str, optional
            The name of the columns in the pandas dataframe
            that contains the name of the day of a record,
            by default "day"

        Returns
        -------
        pd.DataFrame
            The inputted dataframe ordered by the day column
            (by real world order).

        """
        # defences for parameters
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"'df' expected type pd.DataFrame, got {type(df)}")
        if not isinstance(day_column_name, str):
            raise TypeError(
                "'day_column_name' expected type str, "
                f"got {type(day_column_name)}"
            )

        # hard coded day order
        day_order = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        # apply the day order and sort the df
        df["day_order"] = (
            df[day_column_name].str.lower().apply(lambda x: day_order[x])
        )
        df.sort_values("day_order", ascending=True, inplace=True)
        df.sort_index(axis=1, inplace=True)
        df.drop("day_order", inplace=True, axis=1)
        return df

    def _preprocess_trips_and_routes(self) -> pd.DataFrame:
        """Create a trips table containing a record for each trip on each date.

        Returns
        -------
        pd.DataFrame
            A dataframe containing a record of every trip on every day
            from the gtfs feed.

        """
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
        )
        return dated_trips_routes

    def _get_pre_processed_trips(self):
        """Obtain pre-processed trip data."""
        try:
            return self.pre_processed_trips.copy()
        except AttributeError:
            self.pre_processed_trips = self._preprocess_trips_and_routes()
            return self.pre_processed_trips.copy()

    def _summary_defence(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> None:
        """Check for any invalid parameters in a summarising function.

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
        None

        """
        if not isinstance(return_summary, bool):
            raise TypeError(
                "'return_summary' must be of type boolean."
                f" Found {type(return_summary)} : {return_summary}"
            )
        # summ_ops defence

        if isinstance(summ_ops, list):
            for i in summ_ops:
                # updated for numpy >= 1.25.0, this check rules out cases
                # that are not functions
                if inspect.isfunction(i) or type(i).__module__ == "numpy":
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

    def summarise_trips(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Produce a summarised table of trip statistics by day of week.

        For trip count summaries, func counts distinct trip_id only. These
        are then summarised into average/median/min/max (default) number
        of trips per day. Raw data for each date can also be obtained by
        setting the 'return_summary' parameter to False (bool).

        Parameters
        ----------
        summ_ops : list, optional
            A list of operators used to get a summary of a given day,
            by default [np.min, np.max, np.mean, np.median]
        return_summary : bool, optional
            When True, a summary is returned. When False, trip data
            for each date is returned,
            by default True

        Returns
        -------
        pd.DataFrame: A dataframe containing either summarized
                      results or dated route data.

        """
        self._summary_defence(summ_ops=summ_ops, return_summary=return_summary)
        pre_processed_trips = self._get_pre_processed_trips()

        # clean the trips to ensure that there are no duplicates
        cleaned_trips = pre_processed_trips[
            ["date", "day", "trip_id", "route_type"]
        ].drop_duplicates()
        trip_counts = cleaned_trips.groupby(["date", "route_type"]).agg(
            {"trip_id": "count", "day": "first"}
        )
        trip_counts.reset_index(inplace=True)
        trip_counts.rename(
            mapper={"trip_id": "trip_count"}, axis=1, inplace=True
        )
        self.dated_trip_counts = trip_counts.copy()
        if not return_summary:
            return self.dated_trip_counts

        # aggregate to mean/median/min/max (default) trips on each day
        # of the week
        day_trip_counts = trip_counts.groupby(["day", "route_type"]).agg(
            {"trip_count": summ_ops}
        )
        day_trip_counts.reset_index(inplace=True)
        day_trip_counts = day_trip_counts.round(0)

        # order the days (for plotting future purposes)
        # order the days (for plotting future purposes)
        day_trip_counts = self._order_dataframe_by_day(df=day_trip_counts)
        day_trip_counts.reset_index(drop=True, inplace=True)
        self.daily_trip_summary = day_trip_counts.copy()
        return self.daily_trip_summary

    def summarise_routes(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Produce a summarised table of route statistics by day of week.

        For route count summaries, func counts route_id only, irrespective of
        which service_id the routes map to. If the services run on different
        calendar days, they will be counted separately. In cases where more
        than one service runs the same route on the same day, these will not be
        counted as distinct routes.

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
        self._summary_defence(summ_ops=summ_ops, return_summary=return_summary)
        pre_processed_trips = self._get_pre_processed_trips()
        cleaned_routes = pre_processed_trips[
            ["route_id", "day", "date", "route_type"]
        ].drop_duplicates()
        # group data into route counts per day
        route_count = (
            cleaned_routes.groupby(["date", "route_type", "day"])
            .agg(
                {
                    "route_id": "count",
                }
            )
            .reset_index()
        )
        route_count.rename(
            mapper={"route_id": "route_count"}, axis=1, inplace=True
        )
        self.dated_route_counts = route_count.copy()

        if not return_summary:
            return self.dated_route_counts

        # aggregate the to the average number of routes
        # on a given day (e.g., Monday)
        day_route_count = (
            route_count.groupby(["day", "route_type"])
            .agg({"route_count": summ_ops})
            .reset_index()
        )

        # order the days (for plotting future purposes)
        day_route_count = self._order_dataframe_by_day(df=day_route_count)
        day_route_count = day_route_count.round(0)
        day_route_count.reset_index(drop=True, inplace=True)
        self.daily_route_summary = day_route_count.copy()

        return self.daily_route_summary

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

    def validate_travel_between_consecutive_stops(self):
        """Validate the travel between consecutive stops in the GTFS data.

        Ensures that a trip is valid by examining the duration and distance
        of a trip. If a vehicle is travelling at an unusual speed, the trip can
        be deemed invalid.
        """
        pass

    def validate_travel_over_multiple_stops(self):
        """Validate travel over multiple stops in the GTFS data."""
        pass
