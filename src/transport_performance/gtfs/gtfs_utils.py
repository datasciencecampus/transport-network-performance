"""Utility functions for GTFS archives."""
import gtfs_kit as gk
import geopandas as gpd
from shapely.geometry import box
from pyprojroot import here
import pandas as pd
import os
import plotly.graph_objects as go
from typing import Union, TYPE_CHECKING
import pathlib
from geopandas import GeoDataFrame
import warnings

if TYPE_CHECKING:
    from transport_performance.gtfs.validation import GtfsInstance

from transport_performance.utils.defence import (
    _is_expected_filetype,
    _check_iterable,
    _type_defence,
    _check_attribute,
    _gtfs_defence,
    _validate_datestring,
    _enforce_file_extension,
    _check_parent_dir_exists,
    _check_item_in_iter,
)
from transport_performance.utils.constants import PKG_PATH


def _validate_bbox_list(bbox: list) -> None:
    """Small function to ensure bbox coords are in the correct order."""
    _type_defence(bbox, "bbox", list)
    # check len
    if len(bbox) != 4:
        raise ValueError(
            f"bbox should have a length of 4, found {len(bbox)} items in list."
        )
    # check data type
    _check_iterable(
        iterable=bbox,
        param_nm="bbox",
        iterable_type=list,
        exp_type=float,
    )
    # check coords
    if bbox[0] > bbox[2]:
        raise ValueError(
            f"BBOX xmin ({bbox[0]}) is greater than xmax ({bbox[2]})"
        )
    if bbox[1] > bbox[3]:
        raise ValueError(
            f"BBOX ymin ({bbox[1]}) is greater than ymax ({bbox[3]})"
        )
    return None


def filter_gtfs(
    gtfs: "GtfsInstance",
    bbox: Union[GeoDataFrame, list, None] = None,
    crs: Union[str, int] = "epsg:4326",
    filter_dates: list = [],
) -> None:
    """Filter the GTFS to either a bbox or a date.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GTFS to filter
    bbox : Union[GeoDataFrame, list, None], optional
        The bbox to filter the GTFS to. Leave as none if the GTFS does not need
         to be cropped. Format - [xmin, ymin, xmax, ymax], by default None
    crs : Union[str, int], optional
        The CRS of the given bbox, by default "epsg:4326"
    filter_dates : list, optional
        The dates to filter the GTFS to. Leave as an empty list if you do not
        require the GTFS to be filtered to a date, by default []

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised if any of the dates passed to 'filter_dates' isn't present in
        the gtfs

    """
    # defences
    _gtfs_defence(gtfs, "gtfs")
    _type_defence(bbox, "bbox", (GeoDataFrame, list, type(None)))
    _type_defence(crs, "crs", (str, int))
    _type_defence(filter_dates, "filter_dates", list)

    # check that filtering has been requested
    if bbox is None and len(filter_dates) == 0:
        warnings.warn(
            UserWarning(
                "No filtering requested. Please pass either a bbox "
                "or a list of dates"
            ),
        )
        return None

    # handle bbox
    if isinstance(bbox, list):
        _validate_bbox_list(bbox)
        # create box polygon around provided coords, need to unpack
        bbox = box(*bbox)
        # gtfs_kit expects gdf
        bbox = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[bbox])

    # filter to bbox
    if bbox is not None:
        gtfs.feed = gtfs.feed.restrict_to_area(bbox)

    # filter to date
    if len(filter_dates) > 0:
        _check_iterable(
            filter_dates, "filter_dates", exp_type=str, iterable_type=list
        )
        for date in filter_dates:
            _validate_datestring(date)
        feed_dates = gtfs.feed.get_dates()
        diff = set(filter_dates).difference(feed_dates)
        if diff:
            raise ValueError(
                f"{diff} passed to 'filter_dates' not present in feed dates "
                f"Feed dates include: {feed_dates}"
            )
        gtfs.feed = gtfs.feed.restrict_to_dates(filter_dates)

    # remove attr (for future runs)
    if hasattr(gtfs, "pre_processed_trips"):
        delattr(gtfs, "pre_processed_trips")

    # post-filtering checks on GTFS
    if len(gtfs.feed.stop_times) < 1:
        warnings.warn(
            f"GTFS from path {gtfs.gtfs_path} is empty after filtering",
            UserWarning,
        )
    return None


def bbox_filter_gtfs(
    in_pth: Union[pathlib.Path, str] = (
        os.path.join(PKG_PATH, "data", "gtfs", "newport-20230613_gtfs.zip"),
    ),
    out_pth: Union[pathlib.Path, str] = pathlib.Path(
        here("data/external/filtered_gtfs.zip")
    ),
    bbox: Union[GeoDataFrame, list] = [
        -3.077081,
        51.52222,
        -2.925075,
        51.593596,
    ],
    units: str = "km",
    crs: str = "epsg:4326",
    filter_dates: list = [],
) -> None:
    """Filter a GTFS feed to any routes intersecting with a bounding box.

    Optionally filter to a list of given dates.

    WARNING: THIS FUNCTION IS DEPRECATED AND WILL INEVITABLY BE REMOVED.

    Parameters
    ----------
    in_pth : Union[pathlib.Path, str], optional
        Path to the unfiltered GTFS feed. Defaults to
        os.path.join(PKG_PATH, "data", "gtfs", "newport-20230613_gtfs.zip").
    out_pth : Union[pathlib.Path, str], optional
        Path to write the filtered feed to. Defaults to
        here("data/external/filtered_gtfs.zip").
    bbox : Union[gpd.GeoDataFrame, list(float)], optional
        A list of x and y values in the order of minx, miny, maxx, maxy.
        Defaults to [-3.077081, 51.52222, -2.925075, 51.593596].
    units : str, optional
        Distance units of the original GTFS. Defaults to "km".
    crs : str, optional
        What projection should the `bbox_list` be interpreted as. Defaults to
        "epsg:4326" for lat long.
    filter_dates: list, optional
        A list of dates to restrict the feed to. Not providing filter_dates
        means that date filtering will not be applied. Defaults to [].

    Returns
    -------
    None

    Raises
    ------
    TypeError
        `bbox` is not of type list or gpd.GeoDataFrame.
        `units` or `crs` are not of type str.
        `out_pth` or `in_pth` are not of types str or pathlib.Path.
        Elements of a `bbox` list are not of type float.
    FileExistsError
        `in_pth` does not exist on disk.
    ValueError
        `in_pth` or `out_pth` does not have the expected .zip extension.


    """
    warnings.filterwarnings("once", category=DeprecationWarning)
    warnings.warn(
        "This function has been deprecated and it's functionality has been "
        "refined in .filter_gtfs().",
        DeprecationWarning,
    )
    typing_dict = {
        "bbox": [bbox, (list, GeoDataFrame)],
        "units": [units, str],
        "crs": [crs, str],
        "out_pth": [out_pth, (str, pathlib.Path)],
        "in_pth": [in_pth, (str, pathlib.Path)],
        "filter_dates": [filter_dates, list],
    }
    for k, v in typing_dict.items():
        _type_defence(v[0], k, v[-1])

    # check paths have valid zip extensions
    _is_expected_filetype(pth=in_pth, param_nm="in_pth")
    _enforce_file_extension(out_pth, ".zip", ".zip", "out_pth")

    if isinstance(bbox, list):
        _check_iterable(
            iterable=bbox,
            param_nm="bbox_list",
            iterable_type=list,
            exp_type=float,
        )
        # create box polygon around provided coords, need to splat
        bbox = box(*bbox)
        # gtfs_kit expects gdf
        bbox = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[bbox])

    feed = gk.read_feed(in_pth, dist_units=units)
    restricted_feed = gk.miscellany.restrict_to_area(feed=feed, area=bbox)
    # optionally retrict to a date
    if len(filter_dates) > 0:
        _check_iterable(filter_dates, "filter_dates", list, exp_type=str)
        # check date format is acceptable
        [_validate_datestring(x) for x in filter_dates]
        feed_dates = restricted_feed.get_dates()
        diff = set(filter_dates).difference(feed_dates)
        if diff:
            raise ValueError(f"{diff} not present in feed dates.")
        restricted_feed = gk.miscellany.restrict_to_dates(
            restricted_feed, filter_dates
        )
    restricted_feed.write(out_pth)
    print(f"Filtered feed written to {out_pth}.")

    return None


def _add_validation_row(
    gtfs, _type: str, message: str, table: str, rows: list = []
) -> None:
    """Add a row to the validity_df dataframe.

    Parameters
    ----------
    gtfs : GtfsInstance
        The gtfs instance containing the validity_df df
    _type : str
        The type of error/warning
    message : str
        The error/warning message
    table : str
        The impacted table
    rows : list, optional
        The impacted rows, by default []

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        An error is raised if the validity df does not exist

    """
    _gtfs_defence(gtfs, "gtfs")
    _type_defence(_type, "_type", str)
    _type_defence(message, "message", str)
    _type_defence(rows, "rows", list)
    _check_attribute(
        gtfs,
        "validity_df",
        message=(
            "The validity_df does not exist as an "
            "attribute of your GtfsInstance object, \n"
            "Did you forget to run the .is_valid() method?"
        ),
    )
    _check_item_in_iter(_type, ["warning", "error"], "_type")
    temp_df = pd.DataFrame(
        {
            "type": [_type],
            "message": [message],
            "table": [table],
            "rows": [rows],
        }
    )

    gtfs.validity_df = pd.concat([gtfs.validity_df, temp_df]).reset_index(
        drop=True
    )
    return None


def filter_gtfs_around_trip(
    gtfs,
    trip_id: str,
    buffer_dist: int = 10000,
    units: str = "m",
    crs: str = "27700",
    out_pth=os.path.join("data", "external", "trip_gtfs.zip"),
) -> None:
    """Filter a GTFS file to an area around a given trip in the GTFS.

    Parameters
    ----------
    gtfs : GtfsInstance
        The GtfsInstance object to crop
    trip_id : str
        The trip ID
    buffer_dist : int, optional
        The distance to create a buffer around the trip, by default 10000
    units : str, optional
        Distance units of the original GTFS, by default "m"
    crs : str, optional
        The CRS to use for adding a buffer, by default "27700"
    out_pth : _type_, optional
        Where to save the new GTFS file,
          by default os.path.join("data", "external", "trip_gtfs.zip")

    Returns
    -------
    None

    Raises
    ------
    ValueError
        An error is raised if a shapeID is not available

    """
    # NOTE: No defence for units as its deleted later on
    _gtfs_defence(gtfs, "gtfs")
    _type_defence(trip_id, "trip_id", str)
    _type_defence(buffer_dist, "buffer_dist", int)
    _type_defence(crs, "crs", str)
    _check_parent_dir_exists(out_pth, "out_pth", create=True)
    trips = gtfs.feed.trips
    shapes = gtfs.feed.shapes

    shape_id = list(trips[trips["trip_id"] == trip_id]["shape_id"])[0]

    # defence
    if pd.isna(shape_id):
        raise ValueError(
            "'shape_id' not available for trip with trip_id: " f"{trip_id}"
        )

    # create a buffer around the trip
    trip_shape = shapes[shapes["shape_id"] == shape_id]
    gdf = gpd.GeoDataFrame(
        trip_shape,
        geometry=gpd.points_from_xy(
            trip_shape.shape_pt_lon, trip_shape.shape_pt_lat
        ),
        crs="EPSG:4326",
    )
    buffered_trip = gdf.to_crs(crs).buffer(distance=buffer_dist)
    bbox = buffered_trip.total_bounds

    # filter the gtfs to the new bbox
    bbox_filter_gtfs(
        in_pth=gtfs.gtfs_path,
        bbox=list(bbox),
        crs=crs,
        units=units,
        out_pth=out_pth,
    )

    return None


# NOTE: Possibly move to a more generalised utils file
def convert_pandas_to_plotly(
    df: pd.DataFrame, return_html: bool = False
) -> go.Figure:
    """Convert a pandas dataframe to a visual plotly figure.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe to convert to plotly
        (single index only)
    return_html : bool, optional
        Whether or not to return the html element,
        by default False

    Returns
    -------
    go.Figure
        A plotly figure containing the drawn dataframe

    Raises
    ------
    LookupError
        An error raised if an invalid colour scheme is passed
    TypeError
        An error raised if the given pandas dataframe is MultiIndex

    """
    # pre-defined colour schemes
    schemes = {
        "dsc": {
            "header_fill": "#12436D",
            "header_font_colour": "white",
            "cell_fill": "#A285D1",
            "cell_font_colour": "black",
            "font_family": "sans-serif",
            "line_colour": "black",
        }
    }
    # defences
    _type_defence(df, "df", pd.DataFrame)
    _type_defence(return_html, "return_html", bool)
    # no use of _type_defence() here as IMO a more descriptive error message is
    # required. in this case.
    if isinstance(df.columns, pd.MultiIndex) or isinstance(
        df.index, pd.MultiIndex
    ):
        raise TypeError(
            "Pandas dataframe must have a singular index, not MultiIndex. "
            "This means that 'df.columns' or 'df.index' does not return a "
            "MultiIndex."
        )
    # harcoding scheme for now. Could be changed to param if more are added
    scheme = "dsc"
    # create plotly df
    fig = go.Figure(
        data=go.Table(
            header=dict(
                values=df.columns.values,
                fill_color=schemes[scheme]["header_fill"],
                font=dict(
                    color=schemes[scheme]["header_font_colour"],
                    family=schemes[scheme]["font_family"],
                ),
                line_color=schemes[scheme]["line_colour"],
            ),
            cells=dict(
                values=[df[col_name] for col_name in df.columns],
                fill_color="#A285D1",
                font=dict(
                    color=schemes[scheme]["cell_font_colour"],
                    family=schemes[scheme]["font_family"],
                ),
                align="left",
                line_color=schemes[scheme]["line_colour"],
            ),
        )
    )

    if return_html:
        return fig.to_html(full_html=False)
    return fig
