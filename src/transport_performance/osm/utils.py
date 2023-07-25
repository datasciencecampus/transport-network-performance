"""Utility functions for OSM files."""
import subprocess
from pyprojroot import here

from transport_performance.utils.defence import (
    _bool_defence,
    _check_list,
    _check_parent_dir_exists,
    _is_expected_filetype,
)


def filter_osm(
    pbf_pth=here("tests/data/newport-2023-06-13.osm.pbf"),
    out_pth="filtered.osm.pbf",
    bbox=[-3.01, 51.58, -2.99, 51.59],
    tag_filter=True,
    install_osmosis=False,
):
    """Filter an osm.pbf file to a bbox. Relies on homebrew with osmosis.

    Parameters
    ----------
    pbf_pth: ((str, pathlib.PosixPath), optional)
        Path to the open street map pbf to be filtered. Defaults to
        here("tests/data/newport-2023-06-13.osm.pbf").
    out_pth: ((str, pathlib.PosixPath), optional)
        Path to write to. Defaults to "filtered.osm.pbf".
    bbox:  (list, optional)
        Bounding box used to perform the filter, in left, bottom, right top
        order. Defaults to [-3.01, 51.58, -2.99, 51.59].
    tag_filter: (bool, optional)
        Should non-highway ways be filtered? Excludes waterway, landuse &
        natural. Defaults to True.
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
    _is_expected_filetype(pbf_pth, param_nm="pbf_pth", exp_ext=".pbf")
    _is_expected_filetype(
        out_pth, param_nm="out_pth", exp_ext=".pbf", check_existing=False
    )
    _bool_defence(tag_filter, param_nm="tag_filter")
    _check_list(bbox, param_nm="bbox", check_elements=True, exp_type=float)
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
    ]
    if tag_filter:  # optionaly filter ways
        print("Rejecting ways:  waterway, landuse & natural.")
        cmd.extend(["--tf", "reject-ways", "waterway=* landuse=* natural=*"])

    cmd.extend(["--used-node", "--write-pbf", out_pth])

    try:
        subprocess.run(cmd, check=True)
        print(f"Filter completed. Written to {out_pth}")
    except subprocess.CalledProcessError as e2:
        raise Exception(f"Error executing osmosis command: {e2}")
    except FileNotFoundError as e3:
        if install_osmosis:
            print(f"osmosis command was not recognised: {e3}. Trying install.")
            subprocess.run(["brew", "install", "osmosis"])
            print("Installation of `osmosis successful.`")
            subprocess.run(cmd, check=True)
            print(f"Retry filter pbf completed. Written to {out_pth}")
        else:
            raise Exception("`osmosis` is not found. Please install.")

    return None


filter_osm(install_osmosis=True)
