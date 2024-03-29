"""Utility functions for OSM files."""
import subprocess
from pyprojroot import here
from typing import Union
import pathlib

from transport_performance.utils.defence import (
    _type_defence,
    _check_iterable,
    _check_parent_dir_exists,
    _is_expected_filetype,
    _enforce_file_extension,
)


def filter_osm(
    pbf_pth: Union[str, pathlib.PosixPath] = here(
        "tests/data/newport-2023-06-13.osm.pbf"
    ),
    out_pth: Union[str, pathlib.PosixPath] = here(
        "outputs/osm/filtered.osm.pbf"
    ),
    bbox: list = [-3.01, 51.58, -2.99, 51.59],
    tag_filter: bool = True,
    install_osmosis: bool = False,
) -> None:
    """Filter an osm.pbf file to a bbox. Relies on homebrew with osmosis.

    Parameters
    ----------
    pbf_pth: ((str, pathlib.PosixPath), optional)
        Path to the open street map pbf to be filtered. Defaults to
        here("tests/data/newport-2023-06-13.osm.pbf").
    out_pth: ((str, pathlib.PosixPath), optional)
        Path to write to. Defaults to here("outputs/osm/filtered.osm.pbf").
    bbox:  (list, optional)
        Bounding box used to perform the filter, in left, bottom, right top
        order. Defaults to [-3.01, 51.58, -2.99, 51.59].
    tag_filter: (bool, optional)
        Should non-highway ways be filtered? Excludes buildings, waterway,
        landuse & natural. Defaults to True.
    install_osmosis: (bool, optional)
        Should brew be used to install osmosis if not found. Defaults to False.

    Raises
    ------
        Exception: Subprocess error.
        Exception: Osmosis not found. Will not raise if install_osmosis=True.

    Returns
    -------
        None

    """
    # defence
    _type_defence(pbf_pth, "pbf_pth", (pathlib.Path, str))
    # if pbf_pth is str, convert to Path in order to access as_posix method
    pbf_pth = pathlib.Path(pbf_pth)
    _is_expected_filetype(pbf_pth, param_nm="pbf_pth", exp_ext=".pbf")
    _enforce_file_extension(out_pth, ".pbf", ".pbf", "out_pth")
    for nm, val in {
        "tag_filter": tag_filter,
        "install_osmosis": install_osmosis,
    }.items():
        _type_defence(val, nm, bool)
    # check bbox values makes sense, else osmosis will error
    if not bbox[0] < bbox[2]:
        raise ValueError(
            (
                f"Bounding box longitude West {bbox[0]}"
                f" is not smaller than East {bbox[2]}"
            )
        )
    elif not bbox[1] < bbox[3]:
        raise ValueError(
            (
                f"Bounding box latitude South {bbox[1]}"
                f" is not smaller than North {bbox[3]}"
            )
        )

    _check_iterable(
        bbox,
        param_nm="bbox",
        iterable_type=list,
        check_elements=True,
        exp_type=float,
    )
    _check_parent_dir_exists(out_pth, param_nm="out_pth", create=True)
    # Compile the osmosis command
    cmd = [
        "osmosis",
        "--read-pbf",
        pbf_pth.as_posix(),
        "--bounding-box",
        f"left={bbox[0]}",
        f"bottom={bbox[1]}",
        f"right={bbox[2]}",
        f"top={bbox[3]}",
        # https://github.com/conveyal/r5/issues/276#issuecomment-306638448
        "completeWays=yes",
        "completeRelations=yes",
    ]
    if tag_filter:  # optionaly filter ways
        print("Rejecting ways: buildings, waterway, landuse & natural.")
        cmd.extend(
            [
                "--tf",
                "reject-ways",
                "building=*",
                "waterway=*",
                "landuse=*",
                "natural=*",
            ]
        )

    cmd.extend(["--used-node", "--write-pbf", out_pth])

    try:
        subprocess.run(cmd, check=True)
        print(f"Filter completed. Written to {out_pth}")
    except subprocess.CalledProcessError as e1:
        raise Exception(f"Error executing osmosis command: {e1}")
    except FileNotFoundError as e2:
        if install_osmosis:
            print(f"osmosis command was not recognised: {e2}. Trying install.")
            subprocess.run(["brew", "install", "osmosis"])
            print("Installation of `osmosis successful.`")
            subprocess.run(cmd, check=True)
            print(f"Retry filter pbf completed. Written to {out_pth}")
        else:
            raise Exception("`osmosis` is not found. Please install.")

    return None
