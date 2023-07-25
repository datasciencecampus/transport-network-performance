"""Utility functions for OSM files."""
import subprocess
from pyprojroot import here


def filter_osm(
    pbf_pth=here("tests/data/newport-2023-06-13.osm.pbf"),
    out_pth="filtered.osm.pbf",
    bbox=[-3.01, 51.58, -2.99, 51.59],
    tag_filter=True,
):
    """Filter an osm.pbf file to a bounding box. Relies on homebrew.

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

    Raises
    ------
        NameError: Brew has no formula 'osmosis'.
        Exception: Subprocess error.
        Exception: FileNotFoundError - osmosis not compatible with executed
        command.

    Returns
    -------
        None

    """
    # check osmosis is installed
    check_cmd = ["brew", "deps", "--tree", "--installed", "osmosis"]
    try:
        subprocess.run(
            check_cmd,
            check=True,
        )
    except subprocess.CalledProcessError as e1:
        raise NameError(f"No formula with name 'osmosis': {e1}")

    print(type(pbf_pth))
    inpth = pbf_pth.as_posix()

    # Compile the osmosis command
    cmd = [
        "osmosis",
        "--read-pbf",
        inpth,
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
        raise Exception(f"The osmosis command was not recognised: {e3}")

    return None
