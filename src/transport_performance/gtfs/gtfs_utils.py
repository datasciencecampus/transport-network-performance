"""Utility functions for GTFS archives."""
import gtfs_kit as gk
import geopandas as gpd
from shapely.geometry import box
from pyprojroot import here
import plotly.graph_objects as go
import pandas as pd
from typing import Union
import pathlib
from geopandas import GeoDataFrame
import os

from transport_performance.utils.defence import (
    _is_expected_filetype,
    _check_iterable,
    _type_defence,
)
from transport_performance.utils.constants import PKG_PATH


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
) -> None:
    """Filter a GTFS feed to any routes intersecting with a bounding box.

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
    typing_dict = {
        "bbox": [bbox, (list, GeoDataFrame)],
        "units": [units, str],
        "crs": [crs, str],
        "out_pth": [out_pth, (str, pathlib.Path)],
        "in_pth": [in_pth, (str, pathlib.Path)],
    }
    for k, v in typing_dict.items():
        _type_defence(v[0], k, v[-1])

    # check paths have valid zip extensions
    _is_expected_filetype(pth=in_pth, param_nm="in_pth")
    _is_expected_filetype(
        pth=out_pth, param_nm="out_pth", check_existing=False
    )

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
    restricted_feed.write(out_pth)
    print(f"Filtered feed written to {out_pth}.")

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
