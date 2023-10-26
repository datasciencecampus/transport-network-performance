"""Unit tests for transport_performance/utils/raster.py.

Fixtures used in this file geographically represent a small area over ONS site
in Newport. The data vales do not represent any real or meaningfull measurments
, they are purely for use as a unit test and mimic input data used in this repo
. By design these have been constructed such that the 4 grids touch without
overlap. This allows effective unit testing of the `merge_raster_files`
function by making assertions on the bounds of the inputs and merged output.
"""

import os
import re
import pytest
import pathlib
import numpy as np
import rasterio as rio
import xarray as xr
import rioxarray  # noqa: F401 - import required for xarray but not needed here

from typing import Type
from pytest_lazyfixture import lazy_fixture
from _pytest.python_api import RaisesContext
from transport_performance.utils.raster import (
    merge_raster_files,
    sum_resample_file,
)
from transport_performance.utils.test_utils import _np_to_rioxarray


@pytest.fixture
def merge_xarr_1():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the top left 4x4 raster.
    """
    array_1 = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    transform_1 = rio.Affine(100, 0, -225800, 0, -100, 6036800)
    xarray_1 = _np_to_rioxarray(array_1, transform_1)

    return xarray_1


@pytest.fixture
def merge_xarr_2():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the top right 4x4 raster.
    """
    array_2 = np.random.randn(4, 4)
    transform_2 = rio.Affine(100, 0, -225400, 0, -100, 6036800)
    xarray_2 = _np_to_rioxarray(array_2, transform_2, as_type="float32")

    return xarray_2


@pytest.fixture
def merge_xarr_3():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the bottom left 4x4 raster.
    """
    array_3 = np.random.randn(4, 4)
    transform_3 = rio.Affine(100, 0, -225800, 0, -100, 6036400)
    xarray_3 = _np_to_rioxarray(array_3, transform_3, as_type="float32")

    return xarray_3


@pytest.fixture
def merge_xarr_4():
    """Create a dummay xarray.DataFrame for merge function testing.

    Creates a xarray.DataFrame that forms the bottom right 4x4 raster.
    """
    array_4 = np.random.randn(4, 4)
    transform_4 = rio.Affine(100, 0, -225400, 0, -100, 6036400)
    xarray_4 = _np_to_rioxarray(array_4, transform_4, as_type="float32")

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


@pytest.fixture
def resample_xarr_fpath(
    merge_xarr_1: xr.DataArray,
    tmp_path: str,
) -> str:
    """Build temp directory containing dummy GeoTIFF files to test resampling.

    Used to replicate the behaviour of the `merge_raster_files` raster utils
    function. Writes all the dummy GeoTIFFs to a folder within a temp
    directory.

    Parameters
    ----------
    merge_xarr_1 : xr.DataArray
        Dummy input data from `merge_xarr_1` pytest fixture
    tmp_path : str
        Temporary directory to use for pytest run (a pytest object)

    Returns
    -------
    str
        Filepath to dummy GeoTIFF data

    """
    # create dir to write dummy data within the temp dir
    write_dir = os.path.join(tmp_path, "resample_test_file")
    os.mkdir(write_dir)

    # write dummy data to the dir within the temp dir
    out_filepath = os.path.join(write_dir, "input.tif")
    merge_xarr_1.rio.to_raster(out_filepath)

    return out_filepath


@pytest.fixture
def save_empty_text_file(resample_xarr_fpath: str) -> str:
    """Save an empty text file.

    Parameters
    ----------
    resample_xarr_fpath : str
        File path to dummy raster data, used to make sure file is in the same
        directory.

    Returns
    -------
    str
        Dummy text file name.

    """
    # save an empty text file to the same directory
    working_dir = os.path.dirname(resample_xarr_fpath)
    test_file_name = "text.txt"
    with open(os.path.join(working_dir, test_file_name), "w") as f:
        f.write("")

    return test_file_name


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

    @pytest.mark.parametrize(
        "input_dir, output_dir, output_filename, subset_regex, expected",
        [
            # test input dir that does not exist
            (
                "test",
                None,
                None,
                None,
                pytest.raises(FileNotFoundError, match="not found on disk."),
            ),
            # test input dir not correct type
            (
                1.0,
                None,
                None,
                None,
                pytest.raises(TypeError, match="expected path-like"),
            ),
            # test input dir with no tiff files
            (
                "tests/data/gtfs",
                None,
                None,
                None,
                pytest.raises(
                    FileNotFoundError, match=r"No `\*.tif` files found in .*"
                ),
            ),
            # test output dir not correct type
            (
                lazy_fixture("merge_xarrs_fpath"),
                1.0,
                None,
                None,
                pytest.raises(TypeError, match="expected path-like"),
            ),
            # test output filename not correct type
            (
                lazy_fixture("merge_xarrs_fpath"),
                "",
                1.0,
                None,
                pytest.raises(
                    TypeError,
                    match=(
                        "^`output_filename` expected .*str.*. Got .*float.*"
                    ),
                ),
            ),
            # test output filename not correct file extension
            (
                lazy_fixture("merge_xarrs_fpath"),
                "",
                "test.txt",
                None,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Format .txt provided. Expected ['tif'] for path given"
                        " to 'merged_dir'. Path defaulted to .tif"
                    ),
                ),
            ),
            # test subset_regex not correct type
            (
                lazy_fixture("merge_xarrs_fpath"),
                None,
                None,
                1.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "^`subset_regex` expected (.*str.*NoneType.*)."
                        " Got .*float.*"
                    ),
                ),
            ),
            # test subset_regex that is too specific for find tif inputs
            (
                lazy_fixture("merge_xarrs_fpath"),
                None,
                None,
                "test_regex",
                pytest.raises(
                    FileNotFoundError,
                    match=(
                        r"No `\*.tif` files found in .* after applying regex "
                        r"'test_regex'."
                    ),
                ),
            ),
        ],
    )
    def test_merge_raster_files_on_fail(
        self,
        input_dir,
        output_dir,
        output_filename,
        subset_regex,
        expected: Type[RaisesContext],
        tmp_path: pathlib.Path,
    ) -> None:
        """Test `merge_raster_files` when raises occur.

        Parameters
        ----------
        input_dir
            Path to input directory
        output_dir
            Merged output directory
        output_filename
            Merged output filename
        subset_regex
            Regex str to select subset of input files within `input_dir`
        expected : Type[RaisesContext]
            exception to test with
        tmp_path : pathlib.Path
            path to temporary pytest directory.

        Notes
        -----
        1. Arguments are type hinted here to make unit tests more maintable.
        See `merge_raster_files()` docstring for more details.

        """
        with expected:
            if isinstance(output_dir, (str, pathlib.Path)):
                output_dir = os.path.join(tmp_path, output_dir)
            merge_raster_files(
                input_dir=input_dir,
                output_dir=output_dir,
                output_filename=output_filename,
                subset_regex=subset_regex,
            )

    def test_sum_resample_file(self, resample_xarr_fpath: str) -> None:
        """Test `sum_resample_file`.

        Use `merge_xarr_1` to test resampling, which will have a known and
        expected result.

        Parameters
        ----------
        resample_xarr_fpath : str
            Filepath to dummy data.

        """
        # print fpath where input data resides in tmp folder
        # useful when using -rP flag in pytest to see directory
        print(f"Temp filepath for resampling test: {resample_xarr_fpath}")

        # set the output location to sub dir in a different folder
        # adding different sub dir to test resolution of issue 121
        output_fpath = os.path.join(
            os.path.dirname(os.path.dirname(resample_xarr_fpath)),
            "resample_outputs",
            "outputs",
            "output.tif",
        )
        sum_resample_file(resample_xarr_fpath, output_fpath)

        # re-read in resampled output to test
        xds_out = rioxarray.open_rasterio(output_fpath)

        # assert expected resolution, shape, no data value and crs
        assert xds_out.rio.transform().a == 200
        assert xds_out.rio.transform().e == -200
        assert xds_out.to_numpy().shape == (1, 2, 2)
        assert xds_out.rio.nodata == -200.0
        assert xds_out.rio.crs.to_string() == "ESRI:54009"

        # assert correct resampling values (summing consitiuent grids)
        expected_result = np.array([[[14, 22], [46, 54]]])
        assert np.array_equal(expected_result, xds_out.to_numpy())

    @pytest.mark.parametrize(
        "input_filepath, output_filepath, resample_factor, expected",
        [
            # test input_filepath that has an incorrect type
            (
                1.0,
                None,
                None,
                pytest.raises(
                    TypeError,
                    match="expected (.*str.*pathlib.Path.*). Got .*float.*",
                ),
            ),
            # test input_filepath that does not exist
            (
                "test/test/test/test.tif",
                "tester.tif",
                5,
                pytest.raises(FileNotFoundError, match="not found on file."),
            ),
            # test input_filepath that does not have correct file extension
            (
                os.path.join("tests", "data", "newport-2023-06-13.osm.pbf"),
                None,
                None,
                pytest.raises(
                    ValueError,
                    match=(
                        "`input_filepath` expected file extension .tif. Found"
                        " .pbf*"
                    ),
                ),
            ),
            # test output_filepath correct type
            (
                lazy_fixture("resample_xarr_fpath"),
                1.0,
                2,
                pytest.raises(
                    TypeError, match="expected path-like, found .*float.*"
                ),
            ),
            # test output_filepath correct type
            (
                lazy_fixture("resample_xarr_fpath"),
                "",
                "",
                pytest.raises(
                    TypeError,
                    match=("^`resample_factor` expected .*int.*. Got .*str.*"),
                ),
            ),
        ],
    )
    def test_sum_resample_on_fail(
        self,
        input_filepath,
        output_filepath,
        resample_factor,
        expected: Type[RaisesContext],
    ) -> None:
        """Test sum_resample_file in failing cases.

        Parameters
        ----------
        input_filepath
            input filepath for test
        output_filepath
            output filepath for test
        resample_factor
            resample factor to pass to `sum_resample_file`
        expected : Type[RaisesContext]
            exception to test with

        Notes
        -----
        1. Arguments are type hinted here to make unit tests more maintable.
        See `merge_raster_files()` docstring for more details.

        """
        with expected:
            sum_resample_file(
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                resample_factor=resample_factor,
            )
