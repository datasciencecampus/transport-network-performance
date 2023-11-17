"""Validating multiple GTFS at once."""
from typing import Union
from tqdm import tqdm
import pathlib
import glob
import os

from geopandas import GeoDataFrame
import numpy as np
import pandas as pd

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.utils.defence import (
    _type_defence,
    _is_expected_filetype,
    _check_parent_dir_exists,
)


class FileCountError(Exception):
    """Raised when the number of files found is less than expected."""

    pass


class MultiGtfsInstance:
    """Create a feed instance for multiple GTFS files.

    Tihs allows for multiple GTFS files to be cleaned, validated, summarised,
    filtered and saved at the same time.

    Parameters
    ----------
    path : Union[str, list]
        list of paths, or a glob string. See more informtion on glovb strings
        here: https://docs.python.org/3/library/glob.html

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
    save()
        Saves each GtfsInstance to a directory.
    clean_feed()
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

    Raises
    ------
    TypeError
        'path' is not of type string or list.
    FileCountError
        The glob string used for glob.glob does not find at least 2 files.
    FileNotFoundError
        One (or more) of the paths passed to 'path' does not exist.
    ValueError
        Path as no file extension.
    ValueError
        One (or more) of the paths passed are not of the filetype '.zip'.

    Returns
    -------
    None

    """

    def __init__(self, path: Union[str, list]) -> None:
        # defences
        _type_defence(path, "path", (str, list))
        # defend a glob string
        if isinstance(path, str):
            gtfs_paths = glob.glob(path)
            if len(gtfs_paths) < 2:
                raise FileCountError(
                    f"At least 2 files expected at {path}. Found "
                    f"{len(gtfs_paths)}"
                )
            path = gtfs_paths
        # check all paths are zip files
        for i, pth in enumerate(path):
            _is_expected_filetype(pth, f"path[{i}]", True, ".zip")

        self.paths = path
        # instantiate the GtfsInstance's
        self.instances = [GtfsInstance(fpath) for fpath in path]

    def save(self, dir: Union[pathlib.Path, str]) -> None:
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

    def clean_feed(self, clean_kwargs: Union[dict, None] = None) -> None:
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

    def is_valid(self, validation_kwargs: Union[dict, None] = None) -> None:
        """Validate each of the feeds in the MultiGtfsInstance.

        Parameters
        ----------
        validation_kwargs : Union[dict, None], optional
            The kwargs to pass to GtfsInstance.is_valid() for each Gtfs in
            the MultiGtfsInstance, by default None

        Returns
        -------
        None

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
            progress.set_description(f"Cleaning GTFS from path {path}")
            inst.is_valid(**validation_kwargs)
        return None

    def filter_to_date(self, dates: Union[str, list]) -> None:
        """Filter each GTFS to date(s).

        Parameters
        ----------
        dates : Union[str, list]
            The date(s) to filter the GTFS to

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
            inst.filter_to_date(dates=dates)
        return None

    def filter_to_bbox(
        self, bbox: Union[list, GeoDataFrame], crs: str = "epsg:4326"
    ) -> None:
        """Filter GTFS to a bbox.

        Parameters
        ----------
        bbox : Union[list, GeoDataFrame]
            The bbox to filter the GTFS to. Leave as none if the GTFS does not
            need to be cropped. Format - [xmin, ymin, xmax, ymax]
        crs : str, optional
            The CRS of the given bbox, by default "epsg:4326"

        Returns
        -------
        None

        """
        # defences
        _type_defence(bbox, "bbox", (list, GeoDataFrame))
        _type_defence(crs, "crs", str)
        # filter gtfs
        progress = tqdm(zip(self.paths, self.instances), total=len(self.paths))
        for path, inst in progress:
            progress.set_description(f"Filtering GTFS from path {path}")
            inst.filter_to_bbox(bbox=bbox, crs=crs)
        return None

    def _summarise_core(
        self,
        which: str = "route",
        summ_ops: list = [np.min, np.max, np.mean, np.median],
    ) -> pd.DataFrame:
        """Concat and summarise the summaries for each GtfsInstance.

        Parameters
        ----------
        which : str, optional
            Which feature to summarise. Options include ['trip', 'route'],
            by default "route"
        summ_ops : list, optional
            A list of operators used for summaries,
            by default [np.min, np.max, np.mean, np.median]

        Returns
        -------
        pd.DataFrame
            A dataframe containing the concated summaries

        Raises
        ------
        ValueError
            An error raised when the 'which' parameter is invalid.

        """
        # only small defences here as defences are included in summarise_routes
        # for each GtfsInstance.
        _type_defence(summ_ops, "summ_ops", list)
        _type_defence(which, "which", str)
        which = which.lower().strip()
        if which not in ["route", "trip"]:
            raise ValueError(
                f"'which' must be on of ['route', 'trip']. Got {which}"
            )
        # concat summaries
        if which == "route":
            summaries = [
                inst.summarise_routes(summ_ops, True)
                for inst in self.instances
            ]
        else:
            summaries = [
                inst.summarise_trips(summ_ops, True) for inst in self.instances
            ]
        combined_sums = pd.concat(summaries)
        group_cols = ["day", "route_type"]
        op_map = {
            col: col.split("_")[-1]
            for col in combined_sums.columns.values
            if col not in group_cols
        }
        total_sums = (
            combined_sums.groupby(group_cols)
            .agg(op_map)
            .reset_index()
            .round(0)
        )
        return total_sums

    def summarise_trips(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Prodice a summarised table of trip statistics for each day.

        Parameters
        ----------
        summ_ops : list, optional
            A list of operators used for summaries,
            by default [np.min, np.max, np.mean, np.median]
        return_summary : bool, optional
            Wether or not to return the summary,
            by default True

        Returns
        -------
        pd.DataFrame
            A dataframe containing the summarised trips

        """
        _type_defence(return_summary, "return_summary", bool)
        self.daily_trip_summary = self._summarise_core(
            which="trip", summ_ops=summ_ops
        ).copy()
        if return_summary:
            return self.daily_trip_summary

    def summarise_routes(
        self,
        summ_ops: list = [np.min, np.max, np.mean, np.median],
        return_summary: bool = True,
    ) -> pd.DataFrame:
        """Prodice a summarised table of route statistics for each day.

        Parameters
        ----------
        summ_ops : list, optional
            A list of operators used for summaries,
            by default [np.min, np.max, np.mean, np.median]
        return_summary : bool, optional
            Wether or not to return the summary,
            by default True

        Returns
        -------
        pd.DataFrame
            A dataframe containing the summarised routes

        """
        _type_defence(return_summary, "return_summary", bool)
        self.daily_route_summary = self._summarise_core(
            which="route", summ_ops=summ_ops
        ).copy()
        if return_summary:
            return self.daily_route_summary
