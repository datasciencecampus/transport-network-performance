"""Validating multiple GTFS at once."""
from typing import Union
from tqdm import tqdm
import pathlib
import glob
import os

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
    """Filler Docstring."""

    def __init__(self, path: Union[str, list]) -> None:
        # defences
        _type_defence(path, "path", (str, list))
        # defend a regex string
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
