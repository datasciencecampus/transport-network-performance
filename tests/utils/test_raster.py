"""Unit tests for transport_performance/utils/raster.py.

Fixtures used in this file geographically represent a small area over ONS site
in Newport. The data vales do not represent any real or meaningfull measurments
, they are purely for use as a unit test and mimic input data used in this repo
. By design these have been constructed such that the 4 grids touch without
overlap. This allows effective unit testing of the `merge_raster_files`
function by making assertions on the bounds of the inputs and merged output.
"""

import os
import pytest
import numpy as np
import rasterio as rio
import xarray as xr
import rioxarray  # noqa: F401 - import required for xarray but not needed here

from transport_performance.utils.raster import merge_raster_files


def np_to_rioxarry(
    arr: np.ndarray,
    aff: rio.Affine,
    as_type: str = "float32",
    no_data: int = -200,
    crs: str = "ESRI:54009",
) -> xr.DataArray:
    """Convert numpy array to rioxarry.

    This function is only used within pytest, as a convinent way to build
    fixtures without duplicating code.

    Parameters
    ----------
    arr : np.ndarray
        Input numpy array
    aff : rio.Affine
        Affine transform, for input data
    as_type : str, optional
        Data type, by default "int16"
    no_data : int, optional
        Value to use for no data, by default -200
    crs : _type_, optional
        Coordinate Reference system for input data, by default "ESRI:54009"

    Returns
    -------
    xr.DataArray
        Input numpy array as an xarray.DataArray with the correct

    """
    # get geometry of input arr and generate col, row indcies
    height = arr.shape[0]
    width = arr.shape[1]
    cols = np.arange(width)
    rows = np.arange(height)

    # transform indcies to x, y coordinates
    xs, ys = rio.transform.xy(aff, rows, cols)

    # build x_array object
    x_array = (
        xr.DataArray(arr.T, dims=["x", "y"], coords=dict(x=xs, y=ys))
        .transpose("y", "x")
        .astype(as_type)
        .rio.write_nodata(no_data)
        .rio.write_transform(aff)
        .rio.set_crs(crs)
    )

    return x_array


@pytest.fixture
def merge_xarr_1():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the top left 4x4 raster.
    """
    array_1 = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    transform_1 = rio.Affine(100, 0, -225800, 0, -100, 6036800)
    xarray_1 = np_to_rioxarry(array_1, transform_1)

    return xarray_1


@pytest.fixture
def merge_xarr_2():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the top right 4x4 raster.
    """
    array_2 = np.random.randn(4, 4)
    transform_2 = rio.Affine(100, 0, -225400, 0, -100, 6036800)
    xarray_2 = np_to_rioxarry(array_2, transform_2, as_type="float32")

    return xarray_2


@pytest.fixture
def merge_xarr_3():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the bottom left 4x4 raster.
    """
    array_3 = np.random.randn(4, 4)
    transform_3 = rio.Affine(100, 0, -225800, 0, -100, 6036400)
    xarray_3 = np_to_rioxarry(array_3, transform_3, as_type="float32")

    return xarray_3


@pytest.fixture
def merge_xarr_4():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the bottom right 4x4 raster.
    """
    array_4 = np.random.randn(4, 4)
    transform_4 = rio.Affine(100, 0, -225400, 0, -100, 6036400)
    xarray_4 = np_to_rioxarry(array_4, transform_4, as_type="float32")

    return xarray_4


@pytest.fixture
def merge_xarrs_fpath(
    merge_xarr_1: xr.DataArray,
    merge_xarr_2: xr.DataArray,
    merge_xarr_3: xr.DataArray,
    merge_xarr_4: xr.DataArray,
    tmp_path: str,
) -> str:
    """Build temp directory containing dummy GeoTIFF files.

    Used to replicate the behaviour of the `merge_raster_files` raster utils
    function. Writes all the dummy GeoTIFFs to a folder within a temp
    directory.

    Parameters
    ----------
    merge_xarr_1 : xr.DataArray
        Dummy input data from `merge_xarr_1` pytest fixture
    merge_xarr_2 : xr.DataArray
        Dummy input data from `merge_xarr_2` pytest fixture
    merge_xarr_3 : xr.DataArray
        Dummy input data from `merge_xarr_3` pytest fixture
    merge_xarr_4 : xr.DataArray
        Dummy input data from `merge_xarr_4` pytest fixture
    tmp_path : str
        Temporary directory to use for pytest run (a pytest object)

    Returns
    -------
    str
        Directory within the pytest temp dir in which the dummy GeoTIFF data
        have been written

    """
    # create dir to write dummy data within the temp dir
    write_dir = os.path.join(tmp_path, "merge_test_files")
    os.mkdir(write_dir)

    # write all dummy data to the dir within the temp dir
    tiff_files = [merge_xarr_1, merge_xarr_2, merge_xarr_3, merge_xarr_4]
    file_prefix = "test"
    for i, xarr in enumerate(tiff_files):
        out_filepath = os.path.join(write_dir, f"{file_prefix}_{i}.tif")
        xarr.rio.to_raster(out_filepath)

    return write_dir


class TestUtilsRaster:
    """A class to test utils/raster functions."""

    def test_merge_raster_files(self, merge_xarrs_fpath: str) -> None:
        """Test `merge_raster_files`.

        Parameters
        ----------
        merge_xarrs_fpath : str
            Temporary dir containing the dummy GeoTIFF data. Output from the
            merge_xarrs_fpath pytest fixture

        """
        # print fpath where input data resides in tmp folder
        # useful when using -rP flag in pytest to see directory
        print(f"Temp file path for merge tif inputs: {merge_xarrs_fpath}")

        # create a regex to select tif - not needed, but tests the func arg
        subset_regex = r".{0,}test_[0-9].{0,9}"

        # build the output directory and file name
        merge_out_dir = os.path.join(
            os.path.dirname(merge_xarrs_fpath), "merge_test_output"
        )
        merge_out_filename = "merged.tif"

        # run function to merge dummy inputs and get bounds
        bounds = merge_raster_files(
            merge_xarrs_fpath,
            merge_out_dir,
            merge_out_filename,
            subset_regex=subset_regex,
        )

        # ensure no gaps between merged inputs
        assert bounds["inputs"][0][2] == bounds["inputs"][1][0]
        assert bounds["inputs"][0][1] == bounds["inputs"][2][3]
        assert bounds["inputs"][2][2] == bounds["inputs"][3][0]
        assert bounds["inputs"][3][3] == bounds["inputs"][1][1]

        # ensure the bounds of the output is the extrema of all the inputs
        minx = min([coords[0] for coords in bounds["inputs"]])
        miny = min([coords[1] for coords in bounds["inputs"]])
        maxx = max([coords[2] for coords in bounds["inputs"]])
        maxy = max([coords[3] for coords in bounds["inputs"]])
        assert bounds["output"][0] == (minx, miny, maxx, maxy)
