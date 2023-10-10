"""Validating GTFS data."""
import gtfs_kit as gk
import pandas as pd
import geopandas as gpd
import folium
import datetime
import numpy as np
import os
import inspect
import plotly.express as px
import plotly.io as plotly_io
from pretty_html_table import build_table
import zipfile
import warnings
import pathlib
from typing import Union, Callable
from plotly.graph_objects import Figure as PlotlyFigure

from transport_performance.gtfs.routes import (
    scrape_route_type_lookup,
    get_saved_route_type_lookup,
)
from transport_performance.utils.defence import (
    _is_expected_filetype,
    _check_namespace_export,
    _check_parent_dir_exists,
    _check_column_in_df,
    _type_defence,
    _check_item_in_list,
    _check_attribute,
    _enforce_file_extension,
)

from transport_performance.gtfs.report.report_utils import (
    TemplateHTML,
    _set_up_report_dir,
)
from transport_performance.utils.constants import PKG_PATH


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

    Raises
    ------
    TypeError
        If `start` or `end` are not of type pd.Timestamp.

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


def _create_map_title_text(
    gdf: gpd.GeoDataFrame, units: str, geom_crs: Union[str, int]
) -> str:
    """Generate the map title text when plotting convex hull.

    Parameters
    ----------
    gdf :  gpd.GeoDataFrame
        GeoDataFrame containing the spatial features.
    units :  str
        Distance units of the GTFS feed from which `gdf` originated.
    geom_crs : Union[str, int]:
        The geometric crs to use in reprojecting the data in order to
        calculate the area of the hull polygon.

    Returns
    -------
    str
        The formatted text string for presentation in the map title.

    Raises
    ------
    ValueError
        If `crs_unit` is not either "kilometre" or "metre".

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


def _convert_multi_index_to_single(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a dataframes index from MultiIndex to a singular index.

    This function also removes any differing names generated from numpy
    function

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to adjust index (columns) of.

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe with a modified index (columns)

    """
    df.columns = df.columns = [
        "_".join(value) if "" not in value else "".join(value)
        for value in df.columns.values
    ]
    df.columns = [
        column.replace("amin", "min").replace("amax", "max")
        for column in df.columns.values
    ]

    return df


class GtfsInstance:
    """Create a feed instance for validation, cleaning & visualisation.

    Parameters
    ----------
    gtfs_pth : Union[str, bytes, os.PathLike]
        File path to GTFS archive.
    units: str, optionl
        Spatial units of the GTFS file, defaults to "km".
    route_lookup_pth : Union[str, pathlib.Path], optional
        The path to the route type lookup. If left empty, the default path will
        be used. The default path points to a route lookup table that is held
        within this package, defaults to None.

    Attributes
    ----------
    feed : gtfs_kit.Feed
        A gtfs_kit feed produced using the files at `gtfs_pth` on init.
    gtfs_path : Union[str, pathlib.Path]
        The path to the GTFS archive.
    file_list: list
        Files in the GTFS archive.
    validity_df: pd.DataFrame
        Table of GTFS errors, warnings & their descriptions.
    dated_trip_counts: pd.DataFrame
        Dated trip counts by modality.
    daily_trip_summary: pd.DataFrame
        Summarized trip results by day of the week and modality.
    daily_route_summary: pd.DataFrame
        Dated route counts by modality.
    route_mode_summary_df: pd.DataFrame
        Summarized route counts by day of the week and modality.
    pre_processed_trips: pd.DataFrame
        A table of pre-processed trip data.

    Methods
    -------
    get_gtfs_files()
        Returns the `file_list` attribute.
    is_valid()
        Returns the `validity_df` attribute.
    print_alerts()
        Print validity errors & warning messages in full.
    clean_feed()
        Attempt to clean the `feed` attribute using `gtfs_kit`.
    viz_stops()
        Visualise the stops on a map as points or convex hull. Writes file.
    get_route_modes()
        Returns the `route_mode_summary_df` attribute.
    summarise_trips()
        Returns the `daily_trip_summary` attribute.
    summarise_routes()
        Returns the `daily_route_summary` attribute.
    html_report()
        Generate a HTML report describing the GTFS data.
    _produce_stops_map()
        Produces the stops map for use in `viz_stops()`.
    _order_dataframe_by_day()
        Orders tables by day. Used in `summarise_trips()` and
        `summarise_routes()`.
    _preprocess_trips_and_routes()
        Produces a table of dated trips for use in `_get_pre_processed_trips()`
        .
    _get_pre_processed_trips()
        Attempts to access the `pre_processed_trips` attribute and instantiates
        it with `_preprocess_trips_and_routes()` if not found.
    _summary_defence()
        Check the summary parameters for `summarise_trips()` and
        `summarise_routes()`
    _plot_summary()
        Save a plotly summary table, used in `html_report()`.
    _create_extended_repeated_pair_table()
        Return a table of repeated pair warnings. Used in
        `_extended_validation()`.
    _extended_validation()
        Generate HTML warning & error summary tables for use in `html_report()`
        .

    Raises
    ------
    TypeError
        `pth` is not either of string or pathlib.PosixPath.
    TypeError
        `units` is not of type str.
    FileExistsError
        `pth` does not exist on disk.
    ValueError
        `pth` does not have the expected file extension(s).
    ValueError
        `units` are not one of: "m", "km", "metres", "meters", "kilometres",
        "kilometers".

    """

    def __init__(
        self,
        gtfs_pth: Union[str, pathlib.Path],
        units: str = "km",
        route_lookup_pth: Union[str, pathlib.Path] = None,
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
        self.gtfs_path = gtfs_pth
        if route_lookup_pth is not None:
            _is_expected_filetype(
                pth=route_lookup_pth,
                exp_ext=".pkl",
                param_nm="route_lookup_pth",
            )
            self.ROUTE_LKP = get_saved_route_type_lookup(path=route_lookup_pth)
        else:
            self.ROUTE_LKP = get_saved_route_type_lookup()
        # Constant to remove non needed columns from repeated
        # pair error information.
        # This is a messy method however it is the only
        # way to ensure that the error report remains
        # dynamic and can adadpt to different tables
        # in the GTFS file.

        self.GTFS_UNNEEDED_COLUMNS = {
            "routes": [],
            "agency": ["agency_phone", "agency_lang"],
            "stop_times": [
                "stop_headsign",
                "pickup_type",
                "drop_off_type",
                "shape_dist_traveled",
                "timepoint",
            ],
            "stops": [
                "wheelchair_boarding",
                "location_type",
                "parent_station",
                "platform_code",
            ],
            "calendar_dates": [],
            "calendar": [],
            "trips": [
                "trip_headsign",
                "block_id",
                "shape_id",
                "wheelchair_accessible",
            ],
            "shapes": [],
        }

    def get_gtfs_files(self) -> list:
        """Return a list of files making up the GTFS file.

        Returns
        -------
        list
            A list of files that create the GTFS file

        """
        file_list = zipfile.ZipFile(self.gtfs_path).namelist()
        self.file_list = file_list
        return self.file_list

    def is_valid(self) -> pd.DataFrame:
        """Check a feed is valid with `gtfs_kit`.

        Returns
        -------
        pd.core.frame.DataFrame
            Table of errors, warnings & their descriptions.

        """
        self.validity_df = self.feed.validate()
        return self.validity_df

    def print_alerts(self, alert_type: str = "error") -> None:
        """Print validity errors & warning messages in full.

        Parameters
        ----------
        alert_type : str, optional
                The alert type to print messages. Also accepts "warning".
                Defaults to "error".

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            No `validity_df()` attrubute was found.
        UserWarning
            No alerts of the specified `alert_type` were found.

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
            warnings.warn(
                f"No alerts of type {alert_type} were found.", UserWarning
            )

        return None

    def clean_feed(self) -> None:
        """Attempt to clean feed using `gtfs_kit`."""
        try:
            # In cases where shape_id is missing, keyerror is raised.
            # https://developers.google.com/transit/gtfs/reference#shapestxt
            # shows that shapes.txt is optional file.
            self.feed = self.feed.clean()
        except KeyError:
            # TODO: Issue 74 - Improve this to clean feed when KeyError raised
            print("KeyError. Feed was not cleaned.")

    def _produce_stops_map(
        self, what_geoms: str, is_filtered: bool, crs: Union[int, str]
    ) -> folium.folium.Map:
        """Avoiding complexity hook. Returns the required map.

        Parameters
        ----------
        what_geoms : str
            Has the user asked to visualise 'hull' or 'points'?
        is_filtered : bool
            Has the user specified to plot IDs in stops or stop_times only?
        crs : Union[int, str]
            The crs to use for hull calculation.

        Returns
        -------
        folium.folium.Map
            Folium map object, either points or convex hull.

        """
        if what_geoms == "point":
            if is_filtered:
                plot_ids = self.feed.stop_times["stop_id"]
            else:
                plot_ids = self.feed.stops["stop_id"]
            # viz stop locations
            m = self.feed.map_stops(plot_ids)

        elif what_geoms == "hull":
            if is_filtered:
                # filter the stops table to only those stop_ids present
                # in stop_times, this ensures hull viz agrees with point viz
                stop_time_ids = set(self.feed.stop_times["stop_id"])
                gtfs_hull = self.feed.compute_convex_hull(
                    stop_ids=stop_time_ids
                )
            else:
                # if not filtering, use gtfs_kit method
                gtfs_hull = self.feed.compute_convex_hull()
            # visualise feed, output to file with area est, based on stops
            gdf = gpd.GeoDataFrame(
                {"geometry": gtfs_hull}, index=[0], crs="epsg:4326"
            )
            units = self.feed.dist_units
            # prepare the map title
            txt = _create_map_title_text(gdf, units, crs)
            title_pre = "<h3 align='center' style='font-size:16px'><b>"
            title_html = f"{title_pre}{txt}</b></h3>"
            geo_j = gdf.to_json()
            geo_j = folium.GeoJson(
                data=geo_j, style_function=lambda x: {"fillColor": "red"}
            )
            m = folium.Map()
            geo_j.add_to(m)
            m.get_root().html.add_child(folium.Element(title_html))
            # format map zoom and center
            m.fit_bounds(m.get_bounds())

        return m

    def viz_stops(
        self,
        out_pth: Union[str, pathlib.Path],
        geoms: str = "point",
        geom_crs: Union[int, str] = 27700,
        create_out_parent: bool = False,
        filtered_only: bool = True,
    ) -> None:
        """Visualise the stops on a map as points or convex hull. Writes file.

        Parameters
        ----------
        out_pth : Union[str, pathlib.Path]
            Path to write the map file html document to, including the file
            name. Must end with '.html' file extension.
        geoms : str, optional
            Type of map to plot. If `geoms=point` (the default) uses `gtfs_kit`
            to map point locations of available stops. If `geoms=hull`,
            calculates the convex hull & its area, defaults to "point".
        geom_crs : Union[str, int], optional
            Geometric CRS to use for the calculation of the convex hull area
            only, defaults to "27700" (OSGB36, British National Grid).
        create_out_parent : bool, optional
            Should the parent directory of `out_pth` be created if not found,
            defaults to False.
        filtered_only: bool, optional
            When True, only stops referenced within stop_times.txt will be
            plotted. When False, stops referenced in stops.txt will be plotted.
            Note that gtfs_kit filtering behaviour removes stops from
            stop_times.txt but not stops.txt, defaults to True.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            `out_pth` is not either of string or pathlib.PosixPath.
            `geoms` is not of type str
            `geom_crs` is not of type str or int
            `create_out_parent` or `filtered_only` are not of type bool
        FileNotFoundError
            Raised if the parent directory of `out_pth` could not be found on
            disk and `create_out_parent` is False.
        KeyError
            The stops table has no 'stops_code' column.
        UserWarning
            If the file extension of `out_pth` is not .html, the extension will
            be changed to .html.

        """
        typing_dict = {
            "out_pth": [out_pth, (str, pathlib.Path)],
            "geoms": [geoms, str],
            "geoms_crs": [geom_crs, (str, int)],
            "create_out_parent": [create_out_parent, bool],
            "filtered_only": [filtered_only, bool],
        }
        for k, v in typing_dict.items():
            _type_defence(v[0], param_nm=k, types=v[-1])
        # out_pth defence
        _check_parent_dir_exists(
            pth=out_pth, param_nm="out_pth", create=create_out_parent
        )
        out_pth = _enforce_file_extension(out_pth, ".html", ".html", "out_pth")

        # geoms defence
        geoms = geoms.lower().strip()
        ACCEPT_VALS = ["point", "hull"]
        _check_item_in_list(geoms, ACCEPT_VALS, "geoms")

        try:
            m = self._produce_stops_map(
                what_geoms=geoms, is_filtered=filtered_only, crs=geom_crs
            )
            # map_stops will fail if stop_code not present. According to :
            # https://developers.google.com/transit/gtfs/reference#stopstxt
            # This should be an optional column
            m.save(out_pth)
        except KeyError:
            # KeyError inside of an except KeyError here. This is to provide
            # a more detailed error message on why a KeyError is being raised.
            raise KeyError(
                "The stops table has no 'stop_code' column. While "
                "this is an optional field in a GTFS file, it "
                "raises an error through the gtfs-kit package."
            )

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

        Raises
        ------
        TypeError
            `df` is not of type pd.DataFrame.
            `day_column_name` is not of type str

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

    def _get_pre_processed_trips(self) -> pd.DataFrame:
        """Obtain pre-processed trip data.

        Returns
        -------
        pd.DataFrame
            `pre_processed_trips` attribute.

        """
        try:
            return self.pre_processed_trips.copy()
        except AttributeError:
            self.pre_processed_trips = self._preprocess_trips_and_routes()
            return self.pre_processed_trips.copy()

    def _summary_defence(
        self,
        summ_ops: list[Callable] = [np.min, np.max, np.mean, np.median],
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
            for each date is returned, by default True.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            `return_summary` is not of type pd.df.
            `summ_ops` must be a numpy function or a list.
            Each item in a `summ_ops` list must be a function.
            Each item in a `summ_ops` list must be a numpy namespace export.
        NotImplementedError
            `summ_ops` is a function not exported from numpy.

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
            by default [np.min, np.max, np.mean, np.median].
        return_summary : bool, optional
            When True, a summary is returned. When False, trip data
            for each date is returned, by default True.

        Returns
        -------
        pd.DataFrame
            A dataframe containing either summarized results or dated trip
            data.

        Raises
        ------
        TypeError
            return_summary is not of type pd.df.
            summ_ops must be a numpy function or a list.
            Each item in a summ_ops list must be a function.
            Each item in a summ_ops list must be a numpy namespace export.
        NotImplementedError
            summ_ops is a function not exported from numpy.

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
        day_trip_counts = (
            trip_counts.groupby(["day", "route_type"])
            .agg({"trip_count": summ_ops})
            .reset_index()
        )

        # order the days (for plotting future purposes)
        day_trip_counts = self._order_dataframe_by_day(df=day_trip_counts)
        day_trip_counts = day_trip_counts.round(0)
        day_trip_counts.reset_index(drop=True, inplace=True)

        # reformat columns
        # including normalsing min and max between different
        # numpy versions (amin/min, amax/max)
        day_trip_counts = _convert_multi_index_to_single(df=day_trip_counts)

        self.daily_trip_summary = day_trip_counts.copy()
        return self.daily_trip_summary

    def summarise_routes(
        self,
        summ_ops: list[Callable] = [np.min, np.max, np.mean, np.median],
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
            by default [np.min, np.max, np.mean, np.median].
        return_summary : bool, optional
            When True, a summary is returned. When False, route data
            for each date is returned, by default True.

        Returns
        -------
        pd.DataFrame
            A dataframe containing either summarized results or dated route
            data.

        Raises
        ------
        TypeError
            return_summary is not of type pd.df.
            summ_ops must be a numpy function or a list.
            Each item in a summ_ops list must be a function.
            Each item in a summ_ops list must be a numpy namespace export.
        NotImplementedError
            summ_ops is a function not exported from numpy.

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

        # reformat columns
        # including normalsing min and max between different
        # numpy versions (amin/min, amax/max)
        day_route_count = _convert_multi_index_to_single(day_route_count)

        self.daily_route_summary = day_route_count.copy()
        return self.daily_route_summary

    def get_route_modes(self) -> pd.DataFrame:
        """Summarise the available routes by their associated `route_type`.

        Returns
        -------
        pd.core.frame.DataFrame
            Summary table of route counts by transport mode.

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

    def _plot_summary(
        self,
        target_column: str,
        which: str = "trip",
        orientation: str = "v",
        day_column: str = "day",
        width: int = 2000,
        height: int = 800,
        xlabel: str = None,
        ylabel: str = None,
        plotly_kwargs: dict = {},
        return_html: bool = False,
        save_html: bool = False,
        save_image: bool = False,
        out_dir: Union[pathlib.Path, str] = pathlib.Path(
            os.path.join("outputs", "gtfs")
        ),
        img_type: str = "png",
    ) -> Union[PlotlyFigure, str]:
        """Plot (and save) a summary table using plotly.

        Parameters
        ----------
        target_column : str
            The name of the column contianing the
            target data (counts)
        which : str, optional
            Which summary to plot. Options include 'trip' and 'route',
            by default "trip"
        orientation : str, optional
            The orientation of the bar plot ("v" or "h"),
            by default "v"
        day_column : str, optional
            The name of the column containing the day,
            by default "day"
        width : int, optional
            The width of the plot (in pixels), by default 2000
        height : int, optional
            The height of the plot (in pixels), by default 800
        xlabel : str, optional
            The label for the x axis. If left empty, the column name will be
            used, by default None
        ylabel : str, optional
            The label for the y axis. If left empty, the column name will be
            used, by default None
        plotly_kwargs : dict, optional
            Kwargs to pass to fig.update_layout() for additional plot
            customisation, by default {}
        return_html : bool, optional
            Whether or not to return a html string, by default False
        save_html : bool, optional
            Whether or not to save the plot as a html file, by default False
        save_image : bool, optional
            Whether or not to save the plot as a PNG, by default False
        out_dir : Union[pathlib.Path, str], optional
            The directory to save the plot into. If a file extension is added
            to this directory, it won't be cleaned. Whatever is passed as the
            out dir will be used as the parent directory of the save, leaving
            the responsibility on the user to specify the correct path., by
            default os.path.join("outputs", "gtfs")
        img_type : str, optional
            The type of the image to be saved. E.g, .svg or .jpeg., by default
            "png"

        Returns
        -------
        Union[PlotlyFigure, str]
            Returns either a HTML string or the plotly figure

        Raises
        ------
        ValueError
            An error is raised if orientation is not 'v' or 'h'.
        ValueError
            An error is raised if an invalid iamge type is passed.

        """
        # parameter type defences
        _type_defence(which, "which", str)
        _type_defence(day_column, "day_column", str)
        _type_defence(target_column, "target_column", str)
        _type_defence(plotly_kwargs, "plotly_kwargs", dict)
        _type_defence(return_html, "return_html", bool)
        _type_defence(width, "width", int)
        _type_defence(height, "height", int)
        _type_defence(xlabel, "xlabel", (str, type(None)))
        _type_defence(ylabel, "ylabel", (str, type(None)))
        _type_defence(save_html, "save_html", bool)
        _type_defence(save_image, "save_iamge", bool)
        _type_defence(img_type, "img_type", str)

        # lower params
        orientation = orientation.lower()
        which = which.lower()

        # ensure 'which' is valid
        _check_item_in_list(
            item=which, _list=["trip", "route"], param_nm="which"
        )

        raw_pth = os.path.join(
            out_dir,
            "summary_" + datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),
        )
        _check_parent_dir_exists(raw_pth, "save_pth", create=True)

        # orientation input defences
        _check_item_in_list(
            item=orientation, _list=["v", "h"], param_nm="orientation"
        )

        # assign the correct values depending on which breakdown has been
        # chosen
        if which == "trip":
            _check_attribute(
                obj=self,
                attr="daily_trip_summary",
                message=(
                    "The daily_trip_summary table could not be found."
                    " Did you forget to call '.summarise_trips()' first?"
                ),
            )
            summary_df = self.daily_trip_summary
            target_column = (
                f"trip_count_{target_column}"
                if "trip_count" not in target_column
                else target_column
            )

        if which == "route":
            _check_attribute(
                obj=self,
                attr="daily_route_summary",
                message=(
                    "The daily_route_summary table could not be found."
                    " Did you forget to call '.summarise_routes()' first?"
                ),
            )
            summary_df = self.daily_route_summary
            target_column = (
                f"route_count_{target_column}"
                if "route_count" not in target_column
                else target_column
            )

        # dataframe column defences
        _check_column_in_df(df=summary_df, column_name=target_column)
        _check_column_in_df(df=summary_df, column_name=day_column)

        # convert column type for better graph plotting, use desc
        summary_df["route_type"] = summary_df["route_type"].astype("str")
        summary_df = summary_df.merge(
            self.ROUTE_LKP, how="left", on="route_type"
        )
        summary_df["desc"] = summary_df["desc"].fillna(
            summary_df["route_type"]
        )
        summary_df["desc"] = summary_df["desc"].apply(
            lambda x: x.split(".")[0]
        )

        xlabel = (
            xlabel
            if xlabel
            else (target_column if orientation == "h" else day_column)
        )
        ylabel = (
            ylabel
            if ylabel
            else (target_column if orientation == "v" else day_column)
        )

        # plot summary using plotly express
        fig = px.bar(
            summary_df,
            x=day_column if orientation == "v" else target_column,
            y=target_column if orientation == "v" else day_column,
            color="desc",
            barmode="group",
            text_auto=True,
            height=height,
            width=width,
            orientation=orientation,
        )

        # format plotly figure
        fig.update_layout(
            plot_bgcolor="white",
            yaxis=dict(
                tickfont=dict(size=18),
                gridcolor="black",
                showline=True,
                showgrid=False if orientation == "h" else True,
                linecolor="black",
                linewidth=2,
                title=ylabel,
            ),
            xaxis=dict(
                tickfont=dict(size=18),
                gridcolor="black",
                showline=True,
                showgrid=False if orientation == "v" else True,
                linecolor="black",
                linewidth=2,
                title=xlabel,
            ),
            font=dict(size=18),
            legend=dict(
                xanchor="right",
                x=0.99,
                yanchor="top",
                y=0.99,
                title="Route Type",
                traceorder="normal",
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        # apply custom arguments if passed
        if plotly_kwargs:
            fig.update_layout(**plotly_kwargs)

        # save the plot if specified (with correct file type)
        if save_html:
            plotly_io.write_html(
                fig=fig,
                file=os.path.normpath(raw_pth + ".html"),
                full_html=False,
            )

        if save_image:
            valid_img_formats = [
                "png",
                "pdf",
                "jpg",
                "jpeg",
                "webp",
                "svg",
            ]
            path = _enforce_file_extension(
                path=os.path.normpath(
                    raw_pth + f".{img_type.replace('.', '')}"
                ),
                exp_ext=valid_img_formats,
                default_ext="png",
                param_nm="img_type",
            )

            plotly_io.write_image(
                fig=fig,
                file=path,
            )
        if return_html:
            return plotly_io.to_html(fig, full_html=False)
        return fig

    def _create_extended_repeated_pair_table(
        self,
        table: pd.DataFrame,
        join_vars: Union[str, list],
        original_rows: list[int],
    ) -> pd.DataFrame:
        """Generate an extended table for repeated pair warnings.

        Parameters
        ----------
        table : pd.DataFrame
            The dataframe with the repeated pair warnings
        join_vars : Union[str, list]
            The variables that have repeated pairs
        original_rows : list[int]
            The original duplicate rows, contained in the GTFS validation table
            (rows column)

        Returns
        -------
        pd.DataFrame
            An extended dataframe containing repeated pairs

        """
        error_table = table.copy().iloc[original_rows]
        remaining = table.copy().loc[~table.index.isin(original_rows)]
        joined_rows = error_table.merge(
            remaining,
            how="left",
            on=join_vars,
            suffixes=["_original", "_duplicate"],
        )
        return joined_rows

    def _extended_validation(
        self, output_path: Union[str, pathlib.Path], scheme: str = "green_dark"
    ) -> None:
        """Generate HTML outputs of impacted rows from GTFS errors/warnings.

        Parameters
        ----------
        output_path : Union[str, pathlib.Path]
            The path to save the HTML output to
        scheme : str, optional
            Colour scheme from pretty_html_table, by default "green_dark".
            Colour schemes can be found here:
            https://pypi.org/project/pretty-html-table/

        Returns
        -------
        None

        """
        table_map = {
            "agency": self.feed.agency,
            "routes": self.feed.routes,
            "stop_times": self.feed.stop_times,
            "stops": self.feed.stops,
            "trips": self.feed.trips,
            "calendar": self.feed.calendar,
        }

        # determine which errors/warnings have rows that can be located
        validation_table = self.is_valid()
        validation_table["valid_row"] = validation_table["rows"].apply(
            lambda x: 1 if len(x) > 0 else 0
        )
        ext_validation_table = validation_table.copy()[
            validation_table["valid_row"] == 1
        ]
        # locate the impacted rows for each error
        for table, rows, message, msg_type in zip(
            ext_validation_table["table"],
            ext_validation_table["rows"],
            ext_validation_table["message"],
            ext_validation_table["type"],
        ):
            # create a more informative table for repeated pairs
            if "Repeated pair" in message:
                join_vars = (
                    message.split("(")[1]
                    .replace(")", "")
                    .replace(" ", "")
                    .split(",")
                )
                drop_cols = [
                    col
                    for col in self.GTFS_UNNEEDED_COLUMNS[table]
                    if col not in join_vars
                ]
                filtered_tbl = table_map[table].copy().drop(drop_cols, axis=1)
                impacted_rows = self._create_extended_repeated_pair_table(
                    table=filtered_tbl,
                    join_vars=join_vars,
                    original_rows=rows,
                )
                base_columns = [
                    item
                    for item in list(filtered_tbl.columns)
                    if item not in join_vars
                ]
                duplicate_counts = {}
                for col in base_columns:
                    duplicate_counts[col] = impacted_rows[
                        impacted_rows[f"{col}_original"]
                        == impacted_rows[f"{col}_duplicate"]
                    ].shape[0]
            else:
                impacted_rows = table_map[table].copy().iloc[rows]

            # create the html to display the impacted rows (clean possibly)
            table_html = f"""
            <head>
                <link rel="stylesheet" href="styles.css">
            </head>
            <body>
            <h1 style="font-family: 'Poppins', sans-serif;margin: 10px;">
                <a href="index.html" style="color:grey;weight:bold;">
                Back to Index</a><hr>
                Table: {table}<br>Message: {message}<br>
                Type: <span style="color:{'red' if msg_type == 'error' else
                        'orange'};font-family: 'Poppins', sans-serif;">
                        {msg_type}</span>
            </h1>"""

            # Add additional information for repeated pairs
            # to the HTML report
            try:
                for counter, var in enumerate(duplicate_counts):
                    if counter == 0:
                        table_html = (
                            table_html
                            + """<br><div
                        style="margin:10px;">
                        <span style="font-weight:bold;font-size:large">
                        Duplicate Counts</span>"""
                        )
                    table_html = table_html + (
                        f"""
                    <div style="display: block;">
                            <dd style="font-weight: bold;
                                margin-inline-start: 0;
                                display: inline-block;
                                min-width: 160px;">{var}: </dd>
                            <dt style="display: inline-block;">
                            {duplicate_counts[var]}</dt>
                    </div>"""
                    )
                table_html = table_html + "</div>"
            except NameError:
                pass

            # add a more detailed route_type decription
            if "route_type" in impacted_rows.columns:
                impacted_rows["route_type"] = impacted_rows[
                    "route_type"
                ].astype("str")
                impacted_rows = impacted_rows.merge(
                    self.ROUTE_LKP, how="left", on="route_type"
                )
                impacted_rows["desc"] = impacted_rows["desc"].fillna(
                    impacted_rows["route_type"]
                )

            table_html = table_html + build_table(
                impacted_rows, scheme, padding="10px", escape=False
            )

            table_html = table_html + "</body>"

            # save the output
            save_name = f"{'_'.join(message.split(' '))}_{table}"
            with open(f"{output_path}/gtfs_report/{save_name}.html", "w") as f:
                f.write(table_html)

        return None

    def html_report(
        self,
        report_dir: Union[str, pathlib.Path] = "outputs",
        overwrite: bool = False,
        summary_type: str = "mean",
        extended_validation: bool = True,
    ) -> None:
        """Generate a HTML report describing the GTFS data.

        Parameters
        ----------
        report_dir : Union[str, pathlib.Path], optional
            The directory to save the report to, by default "outputs"
        overwrite : bool, optional
            Whether or not to overwrite the existing report if it already
            exists in the report_dir, by default False
        summary_type : str, optional
            The type of summary to show on the summaries on the gtfs report, by
            default "mean"
        extended_validation : bool, optional
            Whether or not to create extended reports for gtfs validation
            errors/warnings.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            An error raised if the type of summary passed is invalid

        """
        _type_defence(overwrite, "overwrite", bool)
        _type_defence(summary_type, "summary_type", str)
        _set_up_report_dir(path=report_dir, overwrite=overwrite)
        summary_type = summary_type.lower()
        if summary_type not in ["mean", "min", "max", "median"]:
            raise ValueError("'summary type' must be mean, median, min or max")

        # store todays date
        date = datetime.datetime.strftime(datetime.datetime.now(), "%d-%m-%Y")

        # feed evaluation
        self.clean_feed()
        validation_dataframe = self.is_valid()

        # create extended reports if requested
        if extended_validation:
            self._extended_validation(output_path=report_dir)
            info_href = (
                validation_dataframe["message"].apply(
                    lambda x: "_".join(x.split(" "))
                )
                + "_"
                + validation_dataframe["table"]
                + ".html"
            )
            validation_dataframe["info"] = [
                f"""<a href="{href}"> Further Info</a>"""
                if len(rows) > 1
                else "Unavailable"
                for href, rows in zip(info_href, validation_dataframe["rows"])
            ]

        eval_temp = TemplateHTML(
            path=(
                os.path.join(
                    PKG_PATH,
                    "data",
                    "gtfs",
                    "report",
                    "html_templates",
                    "evaluation_template.html",
                )
            )
        )
        eval_temp._insert(
            "eval_placeholder_1",
            build_table(
                validation_dataframe,
                "green_dark",
                padding="10px",
                escape=False,
            ),
        )
        eval_temp._insert("eval_title_1", "GTFS Feed Warnings and Errors")

        eval_temp._insert(
            "eval_placeholder_2",
            build_table(self.feed.agency, "green_dark", padding="10px"),
        )
        eval_temp._insert("eval_title_2", "GTFS Agency Information")

        eval_temp._insert(
            "name_placeholder", self.feed.feed_info["feed_publisher_name"][0]
        )
        eval_temp._insert(
            "url_placeholder",
            self.feed.feed_info["feed_publisher_url"][0],
            replace_multiple=True,
        )
        eval_temp._insert(
            "lang_placeholder", self.feed.feed_info["feed_lang"][0]
        )
        eval_temp._insert(
            "start_placeholder", self.feed.feed_info["feed_start_date"][0]
        )
        eval_temp._insert(
            "end_placeholder", self.feed.feed_info["feed_end_date"][0]
        )
        eval_temp._insert(
            "version_placeholder", self.feed.feed_info["feed_version"][0]
        )

        count_lookup = dict(self.feed.describe().to_numpy())
        eval_temp._insert(
            "agency_placeholder", str(len(count_lookup["agencies"]))
        )
        eval_temp._insert(
            "routes_placeholder", str(count_lookup["num_routes"])
        )
        eval_temp._insert("trips_placeholder", str(count_lookup["num_trips"]))
        eval_temp._insert("stops_placeholder", str(count_lookup["num_stops"]))
        eval_temp._insert(
            "shapes_placeholder", str(count_lookup["num_shapes"])
        )

        self.get_gtfs_files()
        file_list_html = ""
        for num, file in enumerate(self.file_list, start=1):
            file_list_html = (
                file_list_html
                + f"""
                    <div>
                        <dd>{num}. </dd>
                        <dt>{file}</dt>
                    </div>"""
            )

        eval_temp._insert("eval_placeholder_3", file_list_html)
        eval_temp._insert("eval_title_3", "GTFS Files Included")

        eval_temp._insert("date", date)

        with open(
            f"{report_dir}/gtfs_report/index.html", "w", encoding="utf8"
        ) as eval_f:
            eval_f.writelines(eval_temp._get_template())

        # stops
        self.viz_stops(
            out_pth=(
                pathlib.Path(f"{report_dir}/gtfs_report/stop_locations.html")
            )
        )
        self.viz_stops(
            out_pth=pathlib.Path(f"{report_dir}/gtfs_report/convex_hull.html"),
            geoms="hull",
            geom_crs=27700,
        )
        stops_temp = TemplateHTML(
            (
                os.path.join(
                    PKG_PATH,
                    "data",
                    "gtfs",
                    "report",
                    "html_templates",
                    "stops_template.html",
                )
            )
        )
        stops_temp._insert("stops_placeholder_1", "stop_locations.html")
        stops_temp._insert("stops_placeholder_2", "convex_hull.html")
        stops_temp._insert("stops_title_1", "Stops from GTFS data")
        stops_temp._insert(
            "stops_title_2", "Convex Hull Generated from GTFS Data"
        )
        stops_temp._insert("date", date)
        with open(
            f"{report_dir}/gtfs_report/stops.html", "w", encoding="utf8"
        ) as stops_f:
            stops_f.writelines(stops_temp._get_template())

        # summaries
        self.summarise_routes()
        self.summarise_trips()
        route_html = self._plot_summary(
            which="route",
            target_column=summary_type,
            return_html=True,
            width=1200,
            height=800,
            ylabel="Route Count",
            xlabel="Day",
        )
        trip_html = self._plot_summary(
            which="trip",
            target_column=summary_type,
            return_html=True,
            width=1200,
            height=800,
            ylabel="Trip Count",
            xlabel="Day",
        )

        summ_temp = TemplateHTML(
            path=(
                os.path.join(
                    PKG_PATH,
                    "data",
                    "gtfs",
                    "report",
                    "html_templates",
                    "summary_template.html",
                )
            )
        )
        summ_temp._insert("plotly_placeholder_1", route_html)
        summ_temp._insert(
            "plotly_title_1",
            f"Route Summary by Day and Route Type ({summary_type})",
        )
        summ_temp._insert("plotly_placeholder_2", trip_html)
        summ_temp._insert(
            "plotly_title_2",
            f"Trip Summary by Day and Route Type ({summary_type})",
        )
        summ_temp._insert("date", date)
        with open(
            f"{report_dir}/gtfs_report/summaries.html", "w", encoding="utf8"
        ) as summ_f:
            summ_f.writelines(summ_temp._get_template())

        print(
            f"GTFS Report Created at {report_dir}\n"
            f"View your report here: {report_dir}/gtfs_report"
        )

        return None
