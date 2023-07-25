"""Utility functions for GTFS archives."""
import gtfs_kit as gk
import geopandas as gpd
from shapely.geometry import box
from pyprojroot import here

from transport_performance.utils.defence import (
    _is_expected_filetype,
    _check_list,
)


def bbox_filter_gtfs(
    in_pth=here("tests/data/newport-20230613_gtfs.zip"),
    out_pth=here("data/external/filtered_gtfs.zip"),
    bbox_list=[-3.077081, 51.52222, -2.925075, 51.593596],
    units="m",
    crs="epsg:4326",
):
    """Filter a GTFS feed to any routes intersecting with a bounding box.

    Parameters
    ----------
    in_pth : (str, pathlib.PosixPath)
        Path to the unfiltered GTFS feed. Defaults to
        here("tests/data/newport-20230613_gtfs.zip").
    out_pth : (str, pathlib.PosixPath)
        Path to write the filtered feed to. Defaults to
        here("data/external/filtered_gtfs.zip").
    bbox_list : list(float)
        A list of x and y values in the order of minx, miny, maxx, maxy.
        Defaults to [-3.077081, 51.52222, -2.925075, 51.593596].
    units : str
        Distance units of the original GTFS. Defaults to "m".
    crs : str
        What projection should the `bbox_list` be interpreted as. Defaults to
        "epsg:4326" for lat long.

    Returns
    -------
    None

    """
    _is_expected_filetype(pth=in_pth, param_nm="in_pth")
    _is_expected_filetype(
        pth=out_pth, param_nm="out_pth", check_existing=False
    )
    _check_list(ls=bbox_list, param_nm="bbox_list", exp_type=float)
    for param in [units, crs]:
        if not isinstance(param, str):
            raise TypeError(f"Expected string. Found {type(param)} : {param}")

    # create box polygon around provided coords, need to splat
    box_poly = box(*bbox_list)
    # gtfs_kit expects gdf
    gdf = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[box_poly])
    feed = gk.read_feed(in_pth, dist_units=units)
    newport_feed = gk.miscellany.restrict_to_area(feed=feed, area=gdf)
    newport_feed.write(out_pth)
    print(f"Filtered feed written to {out_pth}.")

    return None
