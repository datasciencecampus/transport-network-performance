"""Validating multiple GTFS at once."""
from typing import Union
from tqdm import tqdm
import pathlib
import glob
import os
import warnings

from geopandas import GeoDataFrame
import numpy as np
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.utils.defence import (
    _type_defence,
    _is_expected_filetype,
    _check_parent_dir_exists,
    _enforce_file_extension,
)


class MultiGtfsInstance:
    """Create a feed instance for multiple GTFS files.

    This allows for multiple GTFS files to be cleaned, validated, summarised,
    filtered and saved at the same time.

    Parameters
    ----------
    path : Union[str, list, pathlib.Path]
        A list of paths, a singular paath object, or a glob string.
        See more informtion on glob strings here:
        https://docs.python.org/3/library/glob.html

    Attributes
    ----------
    paths : list
        A list of the GTFS paths used to create the MultiGtfsInstance object.
    instances : list
        A list of GtfsInstance objects created from self.paths.
    daily_trip_summary : pd.DataFrame
        A combined summary of statistics for trips from all GTFS files in the
        MultiGtfsInstance.
    daily_route_summary : pd.DataFrame
        A combined summary of statistics for routes from all GTFS files in the
        MultiGtfsInstance.

    Methods
    -------
    save_feeds()
        Saves each GtfsInstance to a directory.
    clean_feeds()
        Cleans all of the GTFS files.
    is_valid()
        Validates all of the GTFS files.
    filter_to_date()
        Filter all of the GTFS files to a specific date(s).
    filter_to_bbox()
        Filter all of the GTFS files to a specific bbox.
    summarise_trips()
        Create a summary of all of the routes throughout all GTFS files.
    summarise_routes()
        Create a summary of all of the trips throughout all GTFS files.
    viz_stops()
        Plot each of the stops from all GtfsInstance's on a folium Map object.
    validate_empty_feeds()
        Check if there are empty feeds within the MultiGtfsInstance.

    Raises
    ------
    TypeError
        'path' is not of type string or list.
    FileNotFoundError
        One (or more) of the paths passed to 'path' does not exist.
    FileNotFoundError
        There are no GTFS files found in the passed list of paths, or from the
        glob string.
    ValueError
        Path as no file extension.
    ValueError
        One (or more) of the paths passed are not of the filetype '.zip'.

    Returns
    -------
    None

    """

    def __init__(self, path: Union[str, list, pathlib.Path]) -> None:
        # defences
        _type_defence(path, "path", (str, list, pathlib.Path))
        # check if a pathlib.Path object has been passed (single gtfs)
        if isinstance(path, pathlib.Path):
            #  if a directory is passed, convert to string for glob string
            if os.path.splitext(path)[-1] == "":
                path = str(path) + "/*.zip"
            else:
                path = [path]
        # defend a glob string
        if isinstance(path, str):
            gtfs_paths = glob.glob(path)
            path = gtfs_paths
        if len(path) < 1:
            raise FileNotFoundError("No GTFS files found.")
        # check all paths are zip files
        for i, pth in enumerate(path):
            _is_expected_filetype(pth, f"path[{i}]", True, ".zip")

        self.paths = path
        # instantiate the GtfsInstance's
        self.instances = [GtfsInstance(fpath) for fpath in path]

    def save_feeds(self, dir: Union[pathlib.Path, str]) -> None:
        """Save the GtfsInstances to a directory.

        Parameters
        ----------
        dir : Union[pathlib.Path, str]
            The directory to export the GTFS files into.

        Returns
        -------
        None

        """
        defence_path = os.path.join(dir, "test.test")
        _check_parent_dir_exists(defence_path, "dir", create=True)
        save_paths = [
            os.path.join(
                dir, os.path.splitext(os.path.basename(p))[0] + "_new.zip"
            )
            for p in self.paths
        ]
        progress = tqdm(zip(save_paths, self.instances), total=len(self.paths))
        for path, inst in progress:
            progress.set_description(f"Saving at {path}")
            inst.save(path)
        return None

    def clean_feeds(self, clean_kwargs: Union[dict, None] = None) -> None:
        """Clean each of the feeds in the MultiGtfsInstance.

        Parameters
        ----------
        clean_kwargs : Union[dict, None], optional
            The kwargs to pass to GtfsInstance.clean_feed() for each Gtfs in
            the MultiGtfsInstance, by default None

        Returns
        -------
        None

        """
        # defences
        _type_defence(clean_kwargs, "clean_kwargs", (dict, type(None)))
        if isinstance(clean_kwargs, type(None)):
            clean_kwargs = {}
        # clean GTFS instances
        progress = tqdm(
            zip(self.paths, self.instances), total=len(self.instances)
        )
        for path, inst in progress:
            progress.set_description(f"Cleaning GTFS from path {path}")
            inst.clean_feed(**clean_kwargs)
        return None

    def is_valid(
        self, validation_kwargs: Union[dict, None] = None
    ) -> pd.DataFrame:
        """Validate each of the feeds in the MultiGtfsInstance.

        Parameters
        ----------
        validation_kwargs : Union[dict, None], optional
            The kwargs to pass to GtfsInstance.is_valid() for each Gtfs in
            the MultiGtfsInstance, by default None

        Returns
        -------
        self.validity_df : pd.DataFrame
            A dataframe containing the validation messages from all of the
            GtfsInstance's.

        """
        # defences
        _type_defence(
            validation_kwargs, "validation_kwargs", (dict, type(None))
        )
        if isinstance(validation_kwargs, type(None)):
            validation_kwargs = {}
        # clean GTFS instances
        progress = tqdm(
            zip(self.paths, self.instances), total=len(self.instances)
        )
        for path, inst in progress:
            progress.set_description(f"Validating GTFS from path {path}")
            inst.is_valid(**validation_kwargs)

        # concat all validation tables into one
        tables = []
        for inst in self.instances:
            valid_df = inst.validity_df.copy()
            valid_df["GTFS"] = inst.gtfs_path
            tables.append(valid_df)
        combined_validation = pd.concat(tables)
        self.validity_df = combined_validation
        return self.validity_df.copy().reset_index(drop=True)

    def validate_empty_feeds(self, delete: bool = False) -> list:
        """Ensure the feeds in MultiGtfsInstance are not empty.

        Parameters
        ----------
        delete : bool, optional
            Whether or not to delete the empty feeds, by default False

        Returns
        -------
        list
            A list of feeds that are empty and their index in GtfsInstance.

        """
        empty_feeds = [
            (ind, inst)
            for ind, inst in enumerate(self.instances)
            if len(inst.feed.stop_times) < 1
        ]
        if delete and empty_feeds:
            self.instances = [
                gtfs_inst
                for index, gtfs_inst in enumerate(self.instances, start=0)
                if index not in [i[0] for i in empty_feeds]
            ]
            # aligning filenames with feed contents, reversing to preserve
            # index order
            for f, _ in reversed(empty_feeds):
                self.paths.pop(f)

        if not self.instances:
            warnings.warn("MultiGtfsInstance has no feeds.", UserWarning)
        return empty_feeds

    def filter_to_date(
        self, dates: Union[str, list], delete_empty_feeds: bool = False
    ) -> None:
        """Filter each GTFS to date(s).

        Parameters
        ----------
        dates : Union[str, list]
            The date(s) to filter the GTFS to
        delete_empty_feeds : bool, optional
            Whether or not to remove empty feeds, by default False

        Returns
        -------
        None

        """
        # defences
        _type_defence(dates, "dates", (str, list))
        # convert to normalsed format
        if isinstance(dates, str):
            dates = [dates]
        # filter gtfs
        progress = tqdm(zip(self.paths, self.instances), total=len(self.paths))
        for path, inst in progress:
            progress.set_description(f"Filtering GTFS from path {path}")
            try:
                inst.filter_to_date(dates=dates)
            except ValueError as e:
                raise ValueError(f"In GTFS {path}.\n{e}")
            # implement delete empty feeds for filter_to_date also
            if delete_empty_feeds:
                warnings.filterwarnings("error", category=UserWarning)
                try:
                    self.validate_empty_feeds(True)
                except UserWarning:
                    self._raise_empty_feed_error(dates)
                warnings.resetwarnings()
        return None

    def _raise_empty_feed_error(self, bbox: list):
        """Error indicating that a feed has been emptied by filtering."""
        raise ValueError(
            f"BBOX '{bbox}' has filtered the MultiGtfs to contain no "
            "data. Please re-instantiate your MultiGtfsInstance"
        )

    def filter_to_bbox(
        self,
        bbox: Union[list, GeoDataFrame],
        crs: Union[str, int] = "epsg:4326",
        delete_empty_feeds: bool = False,
    ) -> None:
        """Filter GTFS to a bbox.

        Parameters
        ----------
        bbox : Union[list, GeoDataFrame]
            The bbox to filter the GTFS to. Leave as none if the GTFS does not
            need to be cropped. Format - [xmin, ymin, xmax, ymax]
        crs : Union[str, int], optional
            The CRS of the given bbox, by default "epsg:4326"
        delete_empty_feeds : bool, optional
            Whether or not to remove empty feeds, by default False

        Returns
        -------
        None

        """
        # defences
        _type_defence(bbox, "bbox", (list, GeoDataFrame))
        _type_defence(crs, "crs", (str, int))
        _type_defence(delete_empty_feeds, "delete_empty_feeds", bool)
        # filter gtfs
        progress = tqdm(zip(self.paths, self.instances), total=len(self.paths))
        for path, inst in progress:
            progress.set_description(f"Filtering GTFS from path {path}")
            inst.filter_to_bbox(bbox=bbox, crs=crs)
        if delete_empty_feeds:
            warnings.filterwarnings("error", category=UserWarning)
            try:
                self.validate_empty_feeds(True)
            except UserWarning:
                self._raise_empty_feed_error(bbox)
            warnings.resetwarnings()
        # validate feeds in case MultiGtfs is empty
        empty_count = len(self.validate_empty_feeds(False))
        if empty_count == len(self.instances):
            self._raise_empty_feed_error(bbox)
        return None

    def _summarise_core(
        self,
        which: str = "trips",
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Summarise the MultiGtfsInstance by either trip_id or route_id.

        Parameters
        ----------
        which : str, optional
            Which summary to create. Options include ['trips', 'routes'],
            by default "trips"
        summ_ops : list, optional
            A list of numpy operators to gather a summary on. Accepts operators
              (e.g., np.min) or strings ("min"),
            by default [np.min, np.max, np.mean, np.median]
        return_summary: bool, optional
            When set to False, full data for each trip on each date will be
            returned.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the summary.

        Raises
        ------
        ValueError
            Raises when 'which' is not either 'trips' or 'routes'

        """
        # defences
        _type_defence(summ_ops, "summ_ops", list)
        _type_defence(which, "which", str)
        _type_defence(return_summary, "return_summary", bool)
        which = which.lower().strip()
        if which not in ["trips", "routes"]:
            raise ValueError(
                f"'which' must be one of ['trips', 'routes'].  Got {which}"
            )

        # choose summary
        if which == "trips":
            group_col = "trip_id"
            count_col = "trip_count"
        else:
            group_col = "route_id"
            count_col = "route_count"

        # concat pre-processed trips/routes
        daily_schedule = [
            g._preprocess_trips_and_routes() for g in self.instances
        ]
        daily_schedule = pd.concat(daily_schedule)
        if not return_summary:
            return daily_schedule
        schedule = daily_schedule[
            ["date", "day", group_col, "route_type"]
        ].drop_duplicates()
        # group to each date and take counts
        trip_counts = (
            schedule.groupby(["route_type", "date", "day"])
            .agg({group_col: "count"})
            .reset_index()
        )
        trip_counts.rename(mapper={group_col: count_col}, axis=1, inplace=True)
        trip_counts = (
            trip_counts.groupby(["day", "route_type"])
            .agg({count_col: summ_ops})
            .reset_index()
            .round(0)
        )
        # reformat index
        trip_counts.columns = trip_counts.columns = [
            "_".join(value) if "" not in value else "".join(value)
            for value in trip_counts.columns.values
        ]
        trip_counts.columns = [
            column.replace("amin", "min").replace("amax", "max")
            for column in trip_counts.columns.values
        ]
        trip_counts = self.instances[0]._order_dataframe_by_day(trip_counts)
        return trip_counts

    def summarise_trips(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Summarise the combined GTFS data by trip_id.

        Parameters
        ----------
        summ_ops : list, optional
            A list of numpy operators to gather a summary on. Accepts
            operators
            (e.g., np.min) or strings ("min")
            ,by default [np.min, np.max, np.mean, np.median]
        return_summary: bool, optional
            When set to False, full data for each trip on each date will be
            returned.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the summary

        """
        self.daily_trip_summary = self._summarise_core(
            which="trips", summ_ops=summ_ops, return_summary=return_summary
        )
        return self.daily_trip_summary.copy()

    def summarise_routes(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Summarise the combined GTFS data by route_id.

        Parameters
        ----------
        summ_ops : list, optional
            A list of numpy operators to gather a summary on. Accepts
            operators
            (e.g., np.min) or strings ("min"),
            by default [np.min, np.max, np.mean, np.median].
        return_summary: bool, optional
            When set to False, full data for each trip on each date will be
            returned.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the summary

        """
        self.daily_route_summary = self._summarise_core(
            which="routes", summ_ops=summ_ops, return_summary=return_summary
        )
        return self.daily_route_summary.copy()

    def viz_stops(
        self,
        path: Union[str, pathlib.Path] = None,
        return_viz: bool = True,
        filtered_only: bool = True,
    ) -> Union[folium.Map, None]:
        """Visualise all stops from all of the GTFS files.

        Parameters
        ----------
        path : Union[str, pathlib.Path], optional
            The path to save the folium map to, by default None.
        return_viz : bool, optional
            Whether or not to return the folium map object, by default True.
        filtered_only : bool, optional
            Whether to filter the stops that are plotted to only stop_id's that
            are present in the stop_times table.

        Returns
        -------
        folium.Map
            A folium map with all stops plotted on it.
        None
            Returns none if 'return_viz' is False.

        Raises
        ------
        ValueError
            An error is raised if bot parameters are None as the map won't be
            saved or returned.

        """
        # defences
        _type_defence(path, "path", (str, pathlib.Path, type(None)))
        _type_defence(return_viz, "return_viz", (bool, type(None)))
        _type_defence(filtered_only, "filtered_only", bool)
        if path:
            _check_parent_dir_exists(path, "path", True)
            _enforce_file_extension(path, ".html", ".html", "path")
        if not path and not return_viz:
            raise ValueError(
                "Both 'path' and 'return_viz' parameters are of NoneType."
            )
        # combine stop tables
        parts = []
        for inst in self.instances:
            subset = inst.feed.stops[
                ["stop_lat", "stop_lon", "stop_name", "stop_id", "stop_code"]
            ].copy()
            if filtered_only:
                valid_ids = inst.feed.stop_times.stop_id.unique()
                subset = subset[subset.stop_id.isin(valid_ids)]
            subset["gtfs_path"] = os.path.basename(inst.gtfs_path)
            parts.append(subset)

        all_stops = pd.concat(parts)

        # plot all stops to a folium map
        map = folium.Map(control_scale=True)
        STOP_STYLE = {
            "radius": 8,
            "fill": "true",
            "color": "blue",
            "weight": 1,
            "fillOpacity": 0.8,
        }
        CLICKED_STYLE = {
            "radius": 9,
            "fill": "true",
            "color": "green",
            "weight": 1,
            "fillOpacity": 0.5,
        }
        callback = f"""\
        function (row) {{
            var imarker;
            marker = L.circleMarker(new L.LatLng(row[0], row[1]),
                {STOP_STYLE}
            );
            // add a popup with stop info
            marker.bindPopup(
                    '<b><u>Stop Information</u></b><br>' +
                    '<b>Stop Name:</b>' + row[2] + '<br>' +
                    '<b>Stop ID:</b>' + row[3] + '<br>' +
                    '<b>Stop Code:</b>' + row[4] + '<br>'
            );
            // function for changing marker properties on hover
            function marker_hover(mark) {{
                mark.target.setStyle(
                    {CLICKED_STYLE}
                );
            }};
            function marker_return(mark) {{
                mark.target.setStyle(
                    {STOP_STYLE}
                );
            }};
            // add listeners
            marker.on('mouseover', marker_hover);
            marker.on('mouseout', marker_return);
            // return marker
            return marker;

        }};
        """
        FastMarkerCluster(
            data=all_stops, callback=callback, disableClusteringAtZoom=15
        ).add_to(map)
        # fit map to bounds
        map_bounds = (
            (all_stops.stop_lat.min(), all_stops.stop_lon.min()),
            (all_stops.stop_lat.max(), all_stops.stop_lon.max()),
        )
        map.fit_bounds(map_bounds)

        # save and return
        if path:
            map.save(path)
        if return_viz:
            return map
        return None
