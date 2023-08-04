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

from typing import Type, Tuple
from shapely.geometry import Polygon
from numpy.dtypes import Float64DType
from pytest_lazyfixture import lazy_fixture
from _pytest.python_api import RaisesContext

from transport_performance.population.rasterpop import RasterPop

# value to test thresholding (removal of populations below this value)
THRESHOLD_TEST = 5


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
            [1.25, 2.0, 3.75, 4.0],
            [5.5, 6.25, 7.5, 8.75],
            [9.25, 10.5, 11.0, 12.0],
            [13.75, 14.75, 15.25, 16.5],
        ]
    )
    transform_1 = rio.Affine(100, 0, -225800, 0, -100, 6036800)
    xarray_1 = np_to_rioxarray(array_1, transform_1)

    return xarray_1


@pytest.fixture
def xarr_1_aoi(xarr_1: xr.DataArray) -> Tuple[Type[Polygon], dict]:
    """Create aoi polygon for xarr1.

    This is a cross shape pattern, that excludes the 4 corners of the xarr1.

    Parameters
    ----------
    xarr_1 : xr.DataArray
        Input dummy array, output from `xarr_1` fixture.

    Returns
    -------
    Type[Polygon]
        A polygon representing a dummy area of interest for xarr_1
    dict
        A dictionary of expected results when applying the area of interest
        polygon. Keys include:
        - "post_clip": state of values after reading and clipping xarr_1.
        - "geopandas": expected values of population column in geopandas df.

    """
    coords = (
        (-225650, 6036750),
        (-225650, 6036650),
        (-225750, 6036650),
        (-225750, 6036550),
        (-225650, 6036550),
        (-225650, 6036450),
        (-225550, 6036450),
        (-225550, 6036500),
        (-225500, 6036500),
        (-225500, 6036550),
        (-225450, 6036550),
        (-225450, 6036650),
        (-225500, 6036650),
        (-225500, 6036700),
        (-225550, 6036700),
        (-225550, 6036750),
        (-225650, 6036750),
    )

    # build a dictionary to store expected results (to use during asserts)
    expected = {}

    # expected result after reading and clipping the array
    # set nan values in corners to match post aoi clipping expectations
    # add extra dimension to match reading of a band in rioxarray
    exp_post_clip = np.copy(xarr_1.to_numpy())
    exp_post_clip[0, 0] = np.nan
    exp_post_clip[0, -1] = np.nan
    exp_post_clip[-1, 0] = np.nan
    exp_post_clip[-1, -1] = np.nan
    exp_post_clip = np.expand_dims(exp_post_clip, axis=0)

    # expected results after converting to geopandas
    # flatten to get 1-d array, and remove nans as per _to_geopandas()
    exp_gpd = exp_post_clip.flatten()
    exp_gpd = exp_gpd[~np.isnan(exp_gpd)]

    # extended results following rounding. Note: xarray/numpy round functions
    # use round half to even method, so 10.5 gets rounded down to 10. Also
    # 6 and 15 aren't repeated since geocube combines neighbouring cells with
    # identical values into a larger polygon
    exp_round = np.array([2, 4, 6, 8, 9, 9, 10, 11, 12, 15])

    # expected result are setting a threshold of THRESHOLD_TEST
    exp_threshold = exp_gpd[exp_gpd >= THRESHOLD_TEST]

    # updated the dictionary for returning
    expected["post_clip"] = exp_post_clip
    expected["geopandas"] = exp_gpd
    expected["round"] = exp_round
    expected["threshold"] = exp_threshold

    return Polygon(coords), expected


@pytest.fixture
def xarr_1_uc() -> Tuple[Type[Polygon], dict]:
    """Create urban centre polygon for xarr1.

    This is a rectangular pattern encompassing the 2 right hand central grids
    of xarr1.

    Returns
    -------
    Type[Polygon]
        A polygon representing a dummy urban centre for xarr_1
    dict
        A dictionary of expected results when applying the area of interest
        polygon. Keys include:
        - "within_uc": expected statuses (booleans) showing which grids are
        within the dummy urban centre.

    """
    coords = (
        (-225600, 6036700),
        (-225600, 6036500),
        (-225500, 6036500),
        (-225500, 6036700),
        (-225600, 6036700),
    )

    # build a dictionary to store expected results (to use during asserts)
    expected = {}

    # build an array of booleans indicating which cells are within the UC
    exp_within_uc = np.array(
        [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ]
    )

    # update expected dictionary for returning
    expected["within_uc"] = exp_within_uc

    return Polygon(coords), expected


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

    def test__raster_pop_internal_methods(
        self,
        xarr_1_fpath: str,
        xarr_1_aoi: tuple,
        xarr_1_uc: tuple,
    ) -> None:
        """Test all the internal methods of the RasterPop call.

        Test the performance of the base internal methods. This test is written
        in this way since the internal methods are dependent on one another,
        and this permits unit testing of each internal stage.

        Parameters
        ----------
        xarr_1_fpath : str
            Filepath to dummy data GeoTIFF file. Output from `xarr_1_fpath`
            fixture.
        xarr_1_aoi : tuple
            A tuple containing the area of interest polygon for xarr_1 (index
            0) and a dictionary of expected method outputs after applying the
            area of interest polygon (index 1). For more information on the
            valid keys of this dictionary see the docstring of the `xarr_1_aoi`
            fixture.
        xarr_1_uc : tuple
            A tuple containing the urban centre polygon for xarr_1 (index
            0) and a dictionary of expected method outputs after applying the
            urban centre polygon (index 1). For more information on the valid
            keys of this dictionary see the docstring of the `xarr_1_uc`
            fixture.

        """
        # print fpath where input data resides in tmp folder
        # useful when using -rP flag in pytest to see directory
        print(f"Temp file path for tif input: {xarr_1_fpath}")

        # instantiate RasterPop and read + clip to AOI
        rp = RasterPop(xarr_1_fpath)

        # call and test read and clip method asserting post clip expectation
        rp._read_and_clip(aoi_bounds=xarr_1_aoi[0])
        assert np.array_equal(
            rp._xds.to_numpy(), xarr_1_aoi[1]["post_clip"], equal_nan=True
        )

        # call and test _to_geopandas and assert to geopandas expectation
        rp._to_geopandas()
        assert np.array_equal(
            rp.pop_gdf.population, xarr_1_aoi[1]["geopandas"]
        )
        assert isinstance(rp.pop_gdf.population.dtype, Float64DType)
        assert rp.centroid_gdf.crs == "EPSG:4326"

        # call and test _within_urban_centre
        rp._within_urban_centre(xarr_1_uc[0])
        assert "within_urban_centre" in rp.pop_gdf.columns
        assert "within_urban_centre" in rp.centroid_gdf.columns
        assert np.array_equal(
            rp.pop_gdf.within_urban_centre, xarr_1_uc[1]["within_uc"]
        )
        assert np.array_equal(
            rp.centroid_gdf.within_urban_centre, xarr_1_uc[1]["within_uc"]
        )

    @pytest.mark.parametrize(
        "round, threshold, expected, key",
        [
            (True, None, lazy_fixture("xarr_1_aoi"), "round"),
            (False, THRESHOLD_TEST, lazy_fixture("xarr_1_aoi"), "threshold"),
        ],
    )
    def test_rasterpop(
        self,
        round: bool,
        threshold: int,
        expected: tuple,
        key: str,
        xarr_1_fpath: str,
        xarr_1_aoi: tuple,
        xarr_1_uc: tuple,
        var_name: str = "test_var_name",
    ) -> None:
        """Test RasterPop expected functionalities.

        Parameters
        ----------
        round : bool
            Flag to set rounding pre-processing stage. Round population when
            then flag is true.
        threshold : int
            Threshold to set for removing population grids below a minimum
            level. None implies no thresholding will occur.
        expected : tuple
            Expected result to assert against. Defined and build inside the
            `xarr_1_aoi` fixture.
        key : str
            Key to use in `xarr_1_aoi` dict output, to assert against.
        xarr_1_fpath : str
            Filepath to dummy data
        xarr_1_aoi : tuple
            Dummy area of interest. Output from `xarr_1_aoi` fixture.
        xarr_1_uc : tuple
            Dummy urban centre fixture. Output from `xarr_1_uc` fixture.
        var_name : str, optional
            Name of variable in raster file, by default "test_var_name" to
            test this functionality of renaming the variable of interest.

        """
        # instanstiate RasterPop
        rp = RasterPop(xarr_1_fpath)

        # get the population data for the dummy aoi and uc
        pop_gdf, _ = rp.get_pop(
            aoi_bounds=xarr_1_aoi[0],
            round=round,
            threshold=threshold,
            var_name=var_name,
            urban_centre_bounds=xarr_1_uc[0],
        )

        # assert the population values are as expected
        assert np.array_equal(pop_gdf[var_name], expected[1][key])

    @pytest.mark.parametrize(
        "fpath, aoi_bounds, urban_centre_bounds, expected",
        [
            ("test.tif", None, None, pytest.raises(FileNotFoundError)),
            (
                lazy_fixture("xarr_1_fpath"),
                ("test"),
                None,
                pytest.raises(TypeError),
            ),
            (
                lazy_fixture("xarr_1_fpath"),
                lazy_fixture("xarr_1_aoi"),
                "test",
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_rasterpop_raises(
        self,
        fpath: str,
        aoi_bounds: Tuple[str],
        urban_centre_bounds: str,
        expected: Type[RaisesContext],
    ) -> None:
        """Test raises statements in RasterPop.

        Parameters
        ----------
        fpath : str
            Filepath to dummy data.
        aoi_bounds : Tuple[str]
            Area of interest bounds, as a tuple where the 0th index is used (
            for consistency with the `xarr1_aoi` fixture).
        urban_centre_bounds : str
            Urban centre bounds test string.
        expected : Type[RaisesContext]
            Expected raise result.

        Note
        ----
        The use of strings to compare with expected Polygon types is not
        completely thourough, but is a basic unit test to ensure the raises
        is thrown when there is a type mismatch.

        """
        # trigger the raise and ensure it matches the expected type.
        with expected:
            rp = RasterPop(fpath)
            rp.get_pop(
                aoi_bounds=aoi_bounds[0],
                urban_centre_bounds=urban_centre_bounds,
            )
