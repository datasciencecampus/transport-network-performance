"""Unit tests for transport_performance/population/rasterpop.py.

Fixtures used in this file geographically represent a small area over ONS site
in Newport. The data vales do not represent any real or meaningfull measurments
, they are purely for use as a unit test and mimic input data used in this repo
.
"""

import os
import pytest
import numpy as np
import rasterio as rio
import xarray as xr
import rioxarray  # noqa: F401 - import required for xarray but not needed here


def np_to_rioxarray(
    arr: np.ndarray,
    aff: rio.Affine,
    as_type: str = "float32",
    no_data: int = -200,
    crs: str = "ESRI:54009",
) -> xr.DataArray:
    """Convert numpy array to rioxarry.

    This function is only used within pytest, as a convinent way to build
    fixtures without duplicating code. TODO: this is a duplicate of the
    original function in tests/utils/test_raster.py. See #41 for more details.

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
def xarr_1() -> xr.DataArray:
    """Create a dummay xarray.DataFrame for RasterPop methods testing."""
    array_1 = np.array(
        [
            [1.1, 2.2, 3.3, 4.4],
            [5.5, 6.6, 7.7, 8.8],
            [9.9, 10.1, 11.2, 12.3],
            [13.4, 14.5, 15.6, 16.7],
        ]
    )
    transform_1 = rio.Affine(100, 0, -225800, 0, -100, 6036800)
    xarray_1 = np_to_rioxarray(array_1, transform_1)

    return xarray_1


@pytest.fixture
def xarr_1_fpath(xarr_1: xr.DataArray, tmp_path: str) -> str:
    """Build temp directory to store dummy data to test read methods.

    Parameters
    ----------
    xarr_1 : xr.DataArray
        Dummy input data from `xarr_1` pytest fixture.
    tmp_path : str
        Temporary directory to use for pytest run.

    Returns
    -------
    out_filepath : str
        Filepath to dummy GeoTIFF data.

    """
    # create dir to write dummy data within the temp dir
    write_dir = os.path.join(tmp_path, "test_file")
    os.mkdir(write_dir)

    # write dummy data to the dir within the temp dir
    out_filepath = os.path.join(write_dir, "input.tif")
    xarr_1.rio.to_raster(out_filepath)

    return out_filepath


class TestRasterPop:
    """A class to test population.RasterPop methods."""

    def test__read_and_clip(self, xarr_1_fpath: str) -> None:
        """Test _read and clip method.

        Parameters
        ----------
        xarr_1_fpath : str
            Filepath to dummy data GeoTIFF file. Output from `xarr_1_fpath`
            fixture.

        """
        # print fpath where input data resides in tmp folder
        # useful when using -rP flag in pytest to see directory
        print(f"Temp file path for tif input: {xarr_1_fpath}")
