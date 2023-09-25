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
import geopandas as gpd

from typing import Type, Tuple
from shapely.geometry import Polygon, Point
from numpy.dtypes import Float64DType
from pytest_lazyfixture import lazy_fixture
from _pytest.python_api import RaisesContext
from contextlib import nullcontext as does_not_raise

from transport_performance.population.rasterpop import RasterPop
from transport_performance.utils.test_utils import _np_to_rioxarray


# value to test thresholding (removal of populations below this value)
THRESHOLD_TEST = 5


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
    xarray_1 = _np_to_rioxarray(array_1, transform_1)

    return xarray_1


@pytest.fixture
def xarr_1_aoi(xarr_1: xr.DataArray) -> Tuple[Type[Polygon], dict]:
    """Create dummy aoi polygon for xarr1.

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
        - "round": expected values following rounding
        - "threshold": expected values following thresholding
        - "grid": expected grid geometry for test cell
        - "centroid": expected centroid coordinate for test cell.

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

    # expected results after converting to geopandas dataframe
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

    # test coords of one cell. light unit test since `geocube` unit tests the
    # `vectorize` function here: https://github.com/corteva/geocube/blob/master
    # /test/integration/test_vector.py#L10
    test_grid_idx = 1
    test_grid_coords = (
        (-225600, 6036800),
        (-225600, 6036700),
        (-225500, 6036700),
        (-225500, 6036800),
        (-225600, 6036800),
    )
    test_grid_polygon = Polygon(test_grid_coords)
    exp_test_grid = {"idx": test_grid_idx, "polygon": test_grid_polygon}

    # test a single centroid. light unit test since `geopandas` unit tests the
    # `centroid` function here: https://github.com/geopandas/geopandas/blob/b4b
    # 10313ab57bf2c55592a28fb99687c9a538fc2/geopandas/tests/test_geom_methods.p
    # y#L778
    test_centoid_idx = 7
    test_centoid_coord = (-3.0300622961483463, 51.566809602591896)
    test_centroid_point = Point(test_centoid_coord)
    exp_test_centroid = {"idx": test_centoid_idx, "point": test_centroid_point}

    # updated the dictionary for returning
    expected["post_clip"] = exp_post_clip
    expected["geopandas"] = exp_gpd
    expected["round"] = exp_round
    expected["threshold"] = exp_threshold
    expected["grid"] = exp_test_grid
    expected["centroid"] = exp_test_centroid

    return Polygon(coords), expected


@pytest.fixture
def xarr_1_uc() -> Tuple[Type[Polygon], dict]:
    """Create dummy urban centre polygon for xarr1.

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


@pytest.fixture
def xarr_1_4326(
    xarr_1_aoi: tuple, xarr_1_uc: tuple
) -> Tuple[Polygon, Polygon]:
    """Build AOI and urban centre polygons in EPSG:4326.

    Parameters
    ----------
    xarr_1_aoi : tuple
        Output from `xarr_1_aoi`.
    xarr_1_uc : tuple
        Output from `xarr_1_uc`.

    Returns
    -------
    Tuple[Polygon, Polygon]
        A tuple of polygons in EPSG:4326. At the 0th index is the AOI and at
        the 1st index is the urban centre.

    Note
    ----
    The dummy urban centre boundary used in these unit tests is set to the
    extremites of 2 cells (representive of real example). Conversion to
    EPSG:4325 introduces floating point/rounding errors. This results in
    the reconversion back to mollweide (the step being unit tested) is no
    longer the precise extremeties. This means the grids are no longer with the
    bounds of the re-converted urban centre geometry. Therefore, only for this
    unit test, a buffer is applied to the urban centre before conversion (just
    1m). This is enough to negate the floating point/rounding errors and bring
    the grids back within the converted urban centre boundary. This step is
    not needed for the area of interest since floating point/round errors can
    not lead to a conflicting scenario since there is always sufficent overlap
    into the cells.

    """
    # convert aoi to EPSG:4326
    aoi_4326_gdf = gpd.GeoDataFrame(
        geometry=[xarr_1_aoi[0]], crs="ESRI:54009"
    ).to_crs("EPSG:4326")

    # convert uc to EPSG:4326 - see note for more details.
    uc_4326_gdf = gpd.GeoDataFrame(geometry=[xarr_1_uc[0]], crs="ESRI:54009")
    uc_4326_gdf["geometry"] = uc_4326_gdf.geometry.buffer(1, join_style=2)
    uc_4326_gdf = uc_4326_gdf.to_crs("EPSG:4326")

    return aoi_4326_gdf.loc[0, "geometry"], uc_4326_gdf.loc[0, "geometry"]


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
            A tuple containing the area of interest polygon for xarr_1 and a
            dictionary of expected method outputs after applying the area of
            interest polygon. For more information on the valid keys of this
            dictionary see the docstring of the `xarr_1_aoi` fixture.
        xarr_1_uc : tuple
            A tuple containing the urban centre polygon for xarr_1 and a
            dictionary of expected method outputs after applying the urban
            centre polygon. For more information on the valid keys of this
            dictionary see the docstring of the `xarr_1_uc` fixture.

        """
        # print fpath where input data resides in tmp folder
        # useful when using -rP flag in pytest to see directory
        print(f"Temp file path for tif input: {xarr_1_fpath}")

        rp = RasterPop(xarr_1_fpath)

        # call and test read and clip method asserting post clip expectation
        rp._read_and_clip(aoi_bounds=xarr_1_aoi[0])
        assert np.array_equal(
            rp._xds.to_numpy(), xarr_1_aoi[1]["post_clip"], equal_nan=True
        )

        # call and test _to_geopandas and assert to geopandas expectations
        rp._to_geopandas()
        assert np.array_equal(
            rp.pop_gdf.population, xarr_1_aoi[1]["geopandas"]
        )
        assert isinstance(rp.pop_gdf.population.dtype, Float64DType)
        assert (
            rp.pop_gdf.geometry.iloc[xarr_1_aoi[1]["grid"]["idx"]]
            == xarr_1_aoi[1]["grid"]["polygon"]
        )
        assert rp.centroid_gdf.crs == "EPSG:4326"
        assert (
            rp.centroid_gdf.geometry.iloc[xarr_1_aoi[1]["centroid"]["idx"]]
            == xarr_1_aoi[1]["centroid"]["point"]
        )

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
        """Test RasterPop core functionalities.

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
        "fpath, expected",
        [
            # non-existant file
            (
                "test.tif",
                pytest.raises(
                    FileNotFoundError, match="test.tif not found on file."
                ),
            ),
            # existing file but an incorrect file type
            (
                "tests/data/newport-2023-06-13.osm.pbf",
                pytest.raises(
                    ValueError,
                    match=(
                        "`filepath` expected file extension .tif. Found .pbf"
                    ),
                ),
            ),
        ],
    )
    def test_rasterpop_init_raises(
        self,
        fpath: str,
        expected: Type[RaisesContext],
    ) -> None:
        """Test raises statements in RasterPop.

        Parameters
        ----------
        fpath : str
            Filepath to non-existant/incorrect file type data.
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
            RasterPop(fpath)

    @pytest.mark.parametrize(
        "aoi_bounds, aoi_crs, round, threshold, var_name, urban_centre_bounds,"
        " urban_centre_crs, expected",
        [
            # test aoi_bounds incorrect type
            (
                "test",
                None,
                False,
                None,
                "population",
                lazy_fixture("xarr_1_uc"),
                None,
                pytest.raises(
                    TypeError,
                    match="^`aoi_bounds` expected .*Polygon.*. Got .*str.*",
                ),
            ),
            # test aoi_crs incorrect type
            (
                lazy_fixture("xarr_1_aoi"),
                1.0,
                False,
                None,
                "population",
                lazy_fixture("xarr_1_uc"),
                None,
                pytest.raises(
                    TypeError,
                    match="^`aoi_crs` expected .*str.*. Got .*float.*",
                ),
            ),
            # test bounds incorrect type
            (
                lazy_fixture("xarr_1_aoi"),
                None,
                "test",
                None,
                "population",
                lazy_fixture("xarr_1_uc"),
                None,
                pytest.raises(
                    TypeError,
                    match="^`round` expected .*bool.*. Got .*str.*",
                ),
            ),
            # test threshold incorrect type
            (
                lazy_fixture("xarr_1_aoi"),
                None,
                False,
                "test",
                "population",
                lazy_fixture("xarr_1_uc"),
                None,
                pytest.raises(
                    TypeError,
                    match=(
                        "^`threshold` expected (.*int.*float.*)."
                        " Got .*str.*"
                    ),
                ),
            ),
            # test var_name incorrect type
            (
                lazy_fixture("xarr_1_aoi"),
                None,
                False,
                None,
                1.0,
                lazy_fixture("xarr_1_uc"),
                None,
                pytest.raises(
                    TypeError,
                    match="^`var_name` expected .*str.*. Got .*float.*",
                ),
            ),
            # test urban_centre_bounds incorrect type
            (
                lazy_fixture("xarr_1_aoi"),
                None,
                False,
                None,
                "population",
                "test",
                None,
                pytest.raises(
                    TypeError,
                    match=(
                        "^`urban_centre_bounds` expected .*Polygon.*."
                        " Got .*str.*"
                    ),
                ),
            ),
            # test urban_centre_crs incorrect type
            (
                lazy_fixture("xarr_1_aoi"),
                None,
                False,
                None,
                "population",
                lazy_fixture("xarr_1_uc"),
                1.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "^`urban_centre_crs` expected .*str.*. Got .*float.*"
                    ),
                ),
            ),
        ],
    )
    def test_rasterpop_get_pop_raises(
        self,
        xarr_1_fpath: str,
        aoi_bounds: tuple,
        aoi_crs,
        round,
        threshold,
        var_name,
        urban_centre_bounds: tuple,
        urban_centre_crs,
        expected: Type[RaisesContext],
    ) -> None:
        """Test raises statements in RasterPop `get_population()` method.

        Parameters
        ----------
        xarr_1_fpath : str
            Filepath to dummy data.
        aoi_bounds : tuple
            Area of interest bounds, as a tuple where the 0th index is used (
            for consistency with the `xarr_1_aoi` fixture).
        aoi_crs
            see `get_data()` docstring
        round
            see `get_data()` docstring
        threshold
            see `get_data()` docstring
        var_name
            see `get_data()` docstring
        urban_centre_bounds : tuple
            Urban centre bounds test, as a tuple where the 0th index is used (
            for consistency with the `xarr_1_uc` fixture).
        urban_centre_crs
            see `get_data()` docstring
        expected : Type[RaisesContext]
            Expected raise result.

        Note
        ----
        1. For more information on the other parameters used in this test see
        the `get_pop` docstring. Avoiding duplication by not describing or
        type hinting them here.
        2. When testing `aoi_bounds` and `urban_centre_bounds` input types,
        the capabilty to index strings is being somewhat 'abused' here such
        that "t" out of "test" is being used as the test argument. This
        presents no technical issues because passing "t" is equally valid as
        "test" in this case.

        """
        # trigger the raise and ensure it matches the expected type.
        with expected:
            rp = RasterPop(xarr_1_fpath)
            rp.get_pop(
                aoi_bounds=aoi_bounds[0],
                aoi_crs=aoi_crs,
                round=round,
                threshold=threshold,
                var_name=var_name,
                urban_centre_bounds=urban_centre_bounds[0],
                urban_centre_crs=urban_centre_crs,
            )

    def test_rasterpop_crs_conversion(
        self,
        xarr_1_fpath: str,
        xarr_1_4326: tuple,
        xarr_1_aoi: tuple,
        xarr_1_uc: tuple,
    ):
        """Test the AOI and Urban Centre CRS conversion steps.

        Check population values and within urban centre categorisation
        remains consistent dispite providing inputs in EPSG:4326.

        Parameters
        ----------
        xarr_1_fpath : str
            Filepath to dummy data
        xarr_1_4326 : tuple
            Area of interest and urban centre polygons in "EPSG:4326". Output
            from the `xarr_1_4326` fixture.
        xarr_1_aoi : tuple
            Output from the `xarr_1_aoi` fixture. Contains expected results
            to use for assertion (checking pop values remain consistent).
        xarr_1_uc : tuple
            _Output from the `xarr_1_uc` fixture. Contains expected results to
            use for assertion (checking within urban centre bools are
            consistent).

        """
        # instantiate RasterPop with AOI and UC in EPSG:4326
        rp = RasterPop(xarr_1_fpath)
        pop_gdf, _ = rp.get_pop(
            aoi_bounds=xarr_1_4326[0],
            aoi_crs="EPSG:4326",
            urban_centre_bounds=xarr_1_4326[1],
            urban_centre_crs="EPSG:4326",
        )

        # check results remain consistent
        assert np.array_equal(pop_gdf.population, xarr_1_aoi[1]["geopandas"])
        assert np.array_equal(
            pop_gdf.within_urban_centre, xarr_1_uc[1]["within_uc"]
        )

    @pytest.mark.parametrize(
        "which, save_folder, save_filename",
        [
            ("folium", "outputs", "folium.html"),
            ("cartopy", "outputs", "cartopy.png"),
            ("matplotlib", "outputs", "matplotlib.png"),
        ],
    )
    def test_plot_on_pass(
        self,
        xarr_1_fpath: str,
        xarr_1_aoi: tuple,
        xarr_1_uc: tuple,
        tmp_path: str,
        which: str,
        save_folder: str,
        save_filename: str,
    ) -> None:
        """Test plotting methods.

        Parameters
        ----------
        xarr_1_fpath : str
            filepath to dummy raster data
        xarr_1_aoi : tuple
            aoi polygon for dummy input
        xarr_1_uc : tuple
            urban centre polugon for dummy input
        tmp_path : str
            temporary path to save output within
        which : str
            plotting backend to use
        save_folder: str
            folder to save output within
        save_filename : str
            filename to use when saving the file within the temp directory

        """
        # create the full output path
        output_path = os.path.join(tmp_path, save_folder, save_filename)

        # run raster pop and assert that the file is generated
        rp = RasterPop(xarr_1_fpath)
        rp.get_pop(xarr_1_aoi[0], urban_centre_bounds=xarr_1_uc[0])
        rp.plot(which=which, save=output_path)
        assert os.path.exists(output_path)

    def test_plot_before_get_data(self, xarr_1_fpath: str) -> None:
        """Test case where plot is called before getting data.

        Parameters
        ----------
        xarr_1_fpath : str
            file path to dummy data.

        """
        rp = RasterPop(xarr_1_fpath)
        with pytest.raises(
            NotImplementedError,
            match="Unable to call `plot` without calling `get_pop`.",
        ):
            rp.plot()

    @pytest.mark.parametrize(
        "which, save, expected",
        [
            # test an unknown which argument incorrect type
            (
                "unknown",
                None,
                pytest.raises(
                    ValueError,
                    match=(
                        "Unrecognised value for `which` unknown. "
                        "Must be one of "
                    ),
                ),
            ),
            # test which argument incorrect type
            (
                1.0,
                None,
                pytest.raises(
                    TypeError,
                    match="^`which` expected .*str.*. Got .*float.*",
                ),
            ),
            # test save argument incorrect type
            (
                "folium",
                1.0,
                pytest.raises(
                    TypeError,
                    match="^`save` expected .*path-like.*, found .*float.*",
                ),
            ),
        ],
    )
    def test_plot_unknown_which(
        self,
        xarr_1_fpath: str,
        xarr_1_aoi: tuple,
        xarr_1_uc: tuple,
        which,
        save,
        expected: Type[RaisesContext],
    ) -> None:
        """Test a plot call with an unknown backend plot type.

        Parameters
        ----------
        xarr_1_fpath : str
            file path to dummy data.
        xarr_1_aoi : tuple
            area of interest polygon for dummy data.
        xarr_1_uc : tuple
            urban centre polygon for dummy data.
        which
            see `RasterPop.plot()` docstring
        save
            see `RasterPop.plot()` docstring
        expected : Type[RaisesContext]
            Expected raise result.

        Note
        ----
        1. For more information on the other parameters used in this test see
        the `plot()` docstring. Avoiding duplication by not describing or type
        hinting them here.

        """
        rp = RasterPop(xarr_1_fpath)
        rp.get_pop(xarr_1_aoi[0], urban_centre_bounds=xarr_1_uc[0])
        with expected:
            rp.plot(which=which, save=save)

    def test_plot_foliumn_no_uc(
        self,
        xarr_1_fpath: str,
        xarr_1_aoi: tuple,
    ) -> None:
        """Test folium plotting backend with no urban centre.

        Unit test written to capture fix for issue 52.

        Parameters
        ----------
        xarr_1_fpath : str
            file path to dummy data.
        xarr_1_aoi : tuple
            area of interest polygon for dummy data.

        """
        rp = RasterPop(xarr_1_fpath)
        rp.get_pop(xarr_1_aoi[0])
        with does_not_raise():
            rp.plot(which="folium")

    def test_plot_cartopy_unknown_attr_location(
        self,
        xarr_1_fpath: str,
        xarr_1_aoi: tuple,
    ) -> None:
        """Test cartopy plotting backend with an unknown attribute location.

        Parameters
        ----------
        xarr_1_fpath : str
            Filepath to dummy data.
        xarr_1_aoi : tuple
            Area of interest for dummy data.

        """
        rp = RasterPop(xarr_1_fpath)
        rp.get_pop(xarr_1_aoi[0])
        test_attr_location = "unknown"
        with pytest.raises(
            ValueError,
            match=(
                f"Unrecognised value for `attr_location` {test_attr_location}."
                "Expecting one of "
            ),
        ):
            rp.plot(
                which="cartopy",
                attr_location=test_attr_location,
            )
