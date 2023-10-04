"""Unit tests for transport_performance/urban_centres/urban_centres_class.

Fixtures used in this file are made up. An affine transform matrix has been
created to match the crs, and the bounding box fixture and centre coordinates
were created from it.

Note: in the class parameterised tests below there are some arguments that are
not used across all tests within them. This is a deliberate design choice,
since pytest expects all parameterised arguments to be passed - removing or
excluding from a signle test triggers errors. The alternative would be to
separate the tests and reparameterise each separetly, but this would lead to a
larger codebase that is more difficult to maintain.
"""

import os
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import pathlib
import pytest
import rasterio as rio

from pytest_lazyfixture import lazy_fixture
from shapely.geometry import Polygon
from typing import Union, Type
from _pytest.python_api import RaisesContext

import transport_performance.urban_centres.raster_uc as ucc


# fixtures
@pytest.fixture
def dummy_pop_array(tmp_path: str):
    """Create dummy population array.

    Parameters
    ----------
    tmp_path : str
        Temporary directory to use for pytest run.

    Returns
    -------
    write_dir : str
        Filepath to dummy raster data.

    """
    aff = rio.Affine(1000.0, 0.0, -243000.0, 0.0, -1000.0, 6056000.0)
    crs = rio.CRS.from_string("ESRI: 54009")
    dummy = np.array(
        [
            [5000, 5000, 5000, 1500, 1500, 0, 0, 0, 5000, 5000],
            [5000, 5000, 5000, 0, 0, 1500, 0, 0, 0, 0],
            [5000, 5000, 5000, 1500, 1500, 0, 0, 0, 0, 0],
            [5000, 1500, 1500, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 500, 500, 100, 0, 0, 0],
            [1000, 0, 0, 0, 100, 40, 5000, 0, 0, 0],
        ]
    )

    metadata = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -200,
        "width": dummy.shape[1],
        "height": dummy.shape[0],
        "count": 1,
        "crs": crs,
        "transform": aff,
        "compress": "lzw",
    }

    write_dir = os.path.join(tmp_path, "input.tif")
    with rio.open(write_dir, "w", **metadata) as d:
        d.write(dummy, 1)

    return write_dir


@pytest.fixture()
def bbox():
    """Create dummy bbox.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        Boundaries of the bounding box.

    """
    minx = -243000
    miny = 6056000
    maxx = minx + (1000 * 10)
    maxy = miny + (-1000 * 7)

    polygon_geom = Polygon(
        [
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
        ]
    )

    gdf = gpd.GeoDataFrame(
        index=[0], crs="ESRI: 54009", geometry=[polygon_geom]
    )

    return gdf


@pytest.fixture()
def non_overlapping_bbox():
    """Create dummy bbox that does not overlap with raster.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        Boundaries of the bounding box.

    """
    minx = -23000
    miny = 656000
    maxx = minx + (1000 * 10)
    maxy = miny + (-1000 * 7)

    polygon_geom = Polygon(
        [
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
        ]
    )

    gdf = gpd.GeoDataFrame(
        index=[0], crs="ESRI: 54009", geometry=[polygon_geom]
    )

    return gdf


@pytest.fixture()
def wrong_crs_bbox():
    """Create dummy bbox with wrong crs.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        Boundaries of the bounding box.

    """
    minx = -23000
    miny = 656000
    maxx = minx + (1000 * 10)
    maxy = miny + (-1000 * 7)

    polygon_geom = Polygon(
        [
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
        ]
    )

    gdf = gpd.GeoDataFrame(
        index=[0], crs="EPSG: 4326", geometry=[polygon_geom]
    )

    return gdf


@pytest.fixture
def cluster_centre():
    """Create dummy cluster centre.

    Returns
    -------
    tuple
        Coordinates of the cluster centre.

    """
    return (-242000.0, 6055000.0)


# tests
# test exceptions for file path
@pytest.mark.parametrize(
    "filepath, func, expected",
    [
        (lazy_fixture("dummy_pop_array"), "str", does_not_raise()),
        (lazy_fixture("dummy_pop_array"), "path", does_not_raise()),
        # wrong path
        (
            "wrongpath.tif",
            "str",
            pytest.raises(
                FileNotFoundError,
                match=(r".*wrongpath.tif not found on file."),
            ),
        ),
        # wrong extension
        (
            "wrongpath",
            "str",
            pytest.raises(ValueError, match=(r"No file extension was found")),
        ),
        # wrong type
        (
            1234,
            "num",
            pytest.raises(
                TypeError,
                match=(r"`pth` expected .*'str'.*Path'.* Got .*'int'.*"),
            ),
        ),
    ],
)
def test_file(
    filepath: Union[str, pathlib.Path],
    func: str,
    bbox: gpd.GeoDataFrame,
    cluster_centre: tuple,
    expected: Type[RaisesContext],
):
    """Test filepath.

    Parameters
    ----------
    filepath : Union[str, pathlib.Path]
        Filepath to dummy raster data.
    func : str
        Type of the filepath argument provided.
    bbox : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    cluster_centre : tuple
        Coordinates for the centre of the cluster.
    expected : Type[RaisesContext]
        Expected raise result.

    """
    if func == "str":
        filepath = str(filepath)
    elif func == "num":
        filepath = filepath
    else:
        filepath = Path(filepath)
    with expected:
        assert (
            ucc.UrbanCentre(filepath).get_urban_centre(bbox, cluster_centre)
            is not None
        )


# test exceptions for bounding box
@pytest.mark.parametrize(
    "window, expected",
    [
        (lazy_fixture("bbox"), does_not_raise()),
        (
            "string",
            pytest.raises(
                TypeError, match=(r"`bbox` expected GeoDataFrame, got str")
            ),
        ),
        # wrong format bbox
        (
            pd.DataFrame(),
            pytest.raises(
                TypeError,
                match=(r"`bbox` expected GeoDataFrame, got DataFrame"),
            ),
        ),
        # badly defined bbox
        (
            gpd.GeoDataFrame(),
            pytest.raises(
                AttributeError,
                match=(
                    r"The CRS attribute of a GeoDataFrame without an "
                    r"active geometry column is not defined"
                ),
            ),
        ),
        # bbox not overlapping
        (
            lazy_fixture("non_overlapping_bbox"),
            pytest.raises(
                ValueError, match=(r"Input shapes do not overlap raster")
            ),
        ),
        # wrong crs bbox
        (
            lazy_fixture("wrong_crs_bbox"),
            pytest.raises(
                ValueError, match=(r"Raster and bounding box crs do not match")
            ),
        ),
    ],
)
def test_bbox(
    dummy_pop_array: str,
    window: gpd.GeoDataFrame,
    cluster_centre: tuple,
    expected: Type[RaisesContext],
):
    """Test bounding box.

    Parameters
    ----------
    dummy_pop_array : str
        Filepath to dummy raster data.
    window : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    cluster_centre : tuple
        Coordinates for the centre of the cluster.
    expected : Type[RaisesContext]
        Expected raise result.

    Note
    ----
    The bounding box is defined as the whole of the raster. The filtering of
    the raster using a window is from a third party package and should be
    tested elsewhere.

    """
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                window, cluster_centre
            )
            is not None
        )


# test exceptions for area centre
@pytest.mark.parametrize(
    "centre_coords, centre_crs, expected",
    [
        (lazy_fixture("cluster_centre"), None, does_not_raise()),
        # different crs
        ((51.74, -3.25), "EPSG: 4326", does_not_raise()),
        # outside cluster
        (
            (-235000.0, 6055000.0),
            None,
            pytest.raises(
                ValueError,
                match=(
                    r"Coordinates provided are not included within any "
                    r"cluster"
                ),
            ),
        ),
        # outside bbox
        (
            (-200000.0, 6055000.0),
            None,
            pytest.raises(
                IndexError,
                match=(r"Coordinates fall outside of raster window"),
            ),
        ),
        # check tuple constrains
        (
            (50, 3),
            None,
            pytest.raises(
                TypeError, match=(r"Elements of `coords` need to be float")
            ),
        ),
        (
            (50, 3, 3),
            None,
            pytest.raises(
                ValueError, match=(r"`coords` expected a tuple of lenght 2")
            ),
        ),
        (
            50,
            None,
            pytest.raises(
                TypeError, match=(r"`centre` expected tuple, got int")
            ),
        ),
        (
            "(50, 3)",
            None,
            pytest.raises(
                TypeError, match=(r"`centre` expected tuple, got str")
            ),
        ),
    ],
)
def test_centre(
    dummy_pop_array: str,
    bbox: gpd.GeoDataFrame,
    centre_coords: tuple,
    centre_crs: str,
    expected: Type[RaisesContext],
):
    """Test centre.

    Parameters
    ----------
    dummy_pop_array : str
        Filepath to dummy raster data.
    bbox : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    centre_coords : tuple
        Coordinates for the centre of the cluster.
    centre_crs: str
        CRS string for the centre coordinates.
    expected : Type[RaisesContext]
        Expected raise result.

    """
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, centre=centre_coords, centre_crs=centre_crs
            )
            is not None
        )


# test exceptions for band
@pytest.mark.parametrize(
    "band, expected",
    [
        (1, does_not_raise()),
        (
            1.5,
            pytest.raises(
                TypeError, match=(r"`band_n` expected integer, got float")
            ),
        ),
        (2, pytest.raises(IndexError, match=(r"band index 2 out of range"))),
        (
            "2",
            pytest.raises(
                TypeError, match=(r"`band_n` expected integer, got str")
            ),
        ),
    ],
)
def test_band_n(
    dummy_pop_array: str,
    bbox: gpd.GeoDataFrame,
    cluster_centre: tuple,
    band: int,
    expected: Type[RaisesContext],
):
    """Test raster band parameter.

    Parameters
    ----------
    dummy_pop_array : str
        Filepath to dummy raster data.
    bbox : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    cluster_centre : tuple
        Coordinates for the centre of the cluster.
    band : int
        Band number to load from the raster file.
    expected : Type[RaisesContext]
        Expected raise result.

    """
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, cluster_centre, band_n=band
            )
            is not None
        )


# test cell population threshold
@pytest.mark.parametrize(
    "cell_pop_t, expected, flags",
    [
        (1500, does_not_raise(), [True, True, False]),
        (5000, does_not_raise(), [True, False, False]),
        (
            1500.5,
            pytest.raises(
                TypeError,
                match=(r"`cell_pop_threshold` expected integer, got float"),
            ),
            [],
        ),
        (
            "1500",
            pytest.raises(
                TypeError,
                match=(r"`cell_pop_threshold` expected integer, got str"),
            ),
            [],
        ),
        # tests value that would not create any cluster
        (
            150000,
            pytest.raises(
                ValueError,
                match=(
                    r"`cell_pop_threshold` value too high, no cells over "
                    r"threshold"
                ),
            ),
            [],
        ),
    ],
)
class TestCellPop:
    """Class to test effect of cell pop threshold on output."""

    def test_cell_pop_t(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        cell_pop_t: int,
        expected: Type[RaisesContext],
        flags: list,
    ):
        """Test cell_pop_threshold parameter.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        cell_pop_t : int
            Threshold to define what cells are included.
        expected : Type[RaisesContext]
            Expected raise result.
        flags : list
            List to check including results of algorithm for specific cells.

        Note
        ----
        `flags` is not used in this function, but it has to be included in the
        signature as it is in the parameters.

        """
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, cell_pop_threshold=cell_pop_t
                )
                is not None
            )

    def test_cell_pop_t_output(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        cell_pop_t: int,
        expected: Type[RaisesContext],
        flags: list,
    ):
        """Test cell_pop_threshold output.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        cell_pop_t : int
            Threshold to define what cells are included.
        expected : Type[RaisesContext]
            Expected raise result.
        flags : list
            List to check including results of algorithm for specific cells.

        Note
        ----
        `expected` is not used in this function, but it has to be included in
        the signature as it is in the parameters.

        """
        if flags != []:
            uc = ucc.UrbanCentre(dummy_pop_array)
            uc.get_urban_centre(
                bbox, cluster_centre, cell_pop_threshold=cell_pop_t
            )
            # fills with 5 and 7
            assert uc._UrbanCentre__pop_filt_array[0, 2] == flags[0]
            # fills with 5 but not 7
            assert uc._UrbanCentre__pop_filt_array[0, 3] == flags[1]
            # doesn't fill (checks if outside bounds are 0)
            assert uc._UrbanCentre__pop_filt_array[6, 0] == flags[2]


# test diagonal boolean
@pytest.mark.parametrize(
    "diagonal, expected, cluster, num_clusters",
    [
        (True, does_not_raise(), 1, 3),
        (False, does_not_raise(), 3, 4),
        (
            1,
            pytest.raises(TypeError, match=(r"`diag` must be a boolean")),
            0,
            0,
        ),
        (
            "True",
            pytest.raises(TypeError, match=(r"`diag` must be a boolean")),
            0,
            0,
        ),
    ],
)
class TestDiag:
    """Class to test effect of diagonal boolean on output."""

    def test_diag(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        diagonal: bool,
        expected: Type[RaisesContext],
        cluster: int,
        num_clusters: int,
    ):
        """Test diag parameter.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        diagonal : bool
            Flag to indicate if diagonals are included in cluster.
        expected : Type[RaisesContext]
            Expected raise result.
        cluster : int
            Cluster number returned in specific cell provided. This will be
            different depending on inclusion of diagonals.
        num_clusters : int
            Total number of clusters created.

        Note
        ----
        `cluster` and `num_clusters` are not used in this function, but they
        have to be included in the signature as they are in the parameters.

        """
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, diag=diagonal
                )
                is not None
            )

    def test_diag_output(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        diagonal: bool,
        expected: Type[RaisesContext],
        cluster: int,
        num_clusters: int,
    ):
        """Test diag parameter output.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        diagonal : bool
            Flag to indicate if diagonals are included in cluster.
        expected : Type[RaisesContext]
            Expected raise result.
        cluster : int
            Cluster number returned in specific cell provided. This will be
            different depending on inclusion of diagonals.
        num_clusters : int
            Total number of clusters created.

        Note
        ----
        `expected` is not used in this function, but it has to be included in
        the signature as they are in the parameters.

        """
        if cluster != 0:
            uc = ucc.UrbanCentre(dummy_pop_array)
            uc.get_urban_centre(bbox, cluster_centre, diag=diagonal)
            # checks if diagonal cell is clustered with main blob or separate
            assert uc._UrbanCentre__cluster_array[1, 5] == cluster
            assert uc._UrbanCentre__num_clusters == num_clusters


# test cluster population threshold
@pytest.mark.parametrize(
    "cluster_pop_t, expected, clusters",
    [
        (50000, does_not_raise(), [1, 0, 0]),
        (10000, does_not_raise(), [1, 2, 0]),
        (
            50000.5,
            pytest.raises(
                TypeError,
                match=(r"`cluster_pop_threshold` expected integer, got float"),
            ),
            [],
        ),
        (
            "50000",
            pytest.raises(
                TypeError,
                match=(r"`cluster_pop_threshold` expected integer, got str"),
            ),
            [],
        ),
        # test value that would filter out all clusters
        (
            1000000,
            pytest.raises(
                ValueError,
                match=(
                    r"`cluster_pop_threshold` value too high, no clusters "
                    r"over threshold"
                ),
            ),
            [],
        ),
    ],
)
class TestClusterPop:
    """Class to test effect of clustering pop threshold on output."""

    def test_cluster_pop_t(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        cluster_pop_t: int,
        expected: Type[RaisesContext],
        clusters: list,
    ):
        """Test pop_threshold parameter.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        cluster_pop_t : int
            Threshold to define what clusters are kept.
        expected : Type[RaisesContext]
            Expected raise result.
        clusters : list
            List with cluster numbers for specific cells. Clusters expected
            are based on `cluster_pop_t`

        Note
        ----
        `clusters` is not used in this function, but it has to be included in
        the signature as they are in the parameters.

        """
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, cluster_pop_threshold=cluster_pop_t
                )
                is not None
            )

    def test_cluster_pop_t_output(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        cluster_pop_t: int,
        expected: Type[RaisesContext],
        clusters: list,
    ):
        """Test pop_threshold outputs.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        cluster_pop_t : int
            Threshold to define what clusters are kept.
        expected : Type[RaisesContext]
            Expected raise result.
        clusters : list
            List with cluster numbers for specific cells. Clusters expected
            are based on `cluster_pop_t`

        Note
        ----
        `expected` is not used in this function, but it has to be included in
        the signature as they are in the parameters.

        """
        if clusters != []:
            uc = ucc.UrbanCentre(dummy_pop_array)
            uc.get_urban_centre(
                bbox, cluster_centre, cluster_pop_threshold=cluster_pop_t
            )
            # checks if diagonal cell is clustered with main blob or separate
            assert uc._UrbanCentre__urban_centres_array[0, 0] == clusters[0]
            assert uc._UrbanCentre__urban_centres_array[0, 9] == clusters[1]
            assert uc._UrbanCentre__urban_centres_array[6, 6] == clusters[2]


# test adjacent cells threshold to fill
@pytest.mark.parametrize(
    "cell_fill_t, expected, fills",
    [
        (5, does_not_raise(), [1, 1, 0]),
        (7, does_not_raise(), [1, 0, 0]),
        (
            5.5,
            pytest.raises(
                TypeError,
                match=(r"`cell_fill_threshold` expected integer, got float"),
            ),
            [],
        ),
        (
            "5",
            pytest.raises(
                TypeError,
                match=(r"`cell_fill_threshold` expected integer, got str"),
            ),
            [],
        ),
        # test values outside boundaries
        (
            11,
            pytest.raises(
                ValueError,
                match=(
                    r"Wrong value for `cell_fill_threshold`, please enter "
                    r"value between 5 and 8"
                ),
            ),
            [],
        ),
        (
            0,
            pytest.raises(
                ValueError,
                match=(
                    r"Wrong value for `cell_fill_threshold`, please enter "
                    r"value between 5 and 8"
                ),
            ),
            [],
        ),
    ],
)
class TestFill:
    """Class to test effect of fill threshold on output."""

    def test_cell_fill_t(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        cell_fill_t: int,
        expected: Type[RaisesContext],
        fills: list,
    ):
        """Test cell_fill_threshold parameter.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        cell_fill_t : int
            Number of cells around a specific empty cell needed for this cell
            to be filled.
        expected : Type[RaisesContext]
            Expected raise result.
        fills : list
            List of flags for specific cells, indicating 1 if cell has been
            filled or 0 if they have not been filled. Filling behaviour will
            change based on `cell_fill_t`.

        Note
        ----
        `fills` is not used in this function, but it has to be included in
        the signature as they are in the parameters.

        """
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, cell_fill_threshold=cell_fill_t
                )
                is not None
            )

    def test_cell_fill_output(
        self,
        dummy_pop_array: str,
        bbox: gpd.GeoDataFrame,
        cluster_centre: tuple,
        cell_fill_t: int,
        expected: Type[RaisesContext],
        fills: list,
    ):
        """Test fill output.

        Parameters
        ----------
        dummy_pop_array : str
            Filepath to dummy raster data.
        bbox : gpd.GeoDataFrame
            Boundaries of the bounding box to filter the raster.
        cluster_centre : tuple
            Coordinates for the centre of the cluster.
        cell_fill_t : int
            Number of cells around a specific empty cell needed for this cell
            to be filled.
        expected : Type[RaisesContext]
            Expected raise result.
        fills : list
            List of flags for specific cells, indicating 1 if cell has been
            filled or 0 if they have not been filled. Filling behaviour will
            change based on `cell_fill_t`.

        Note
        ----
        `expected` is not used in this function, but it has to be included in
        the signature as they are in the parameters.

        """
        if fills != []:
            uc = ucc.UrbanCentre(dummy_pop_array)
            uc.get_urban_centre(
                bbox, cluster_centre, cell_fill_threshold=cell_fill_t
            )
            # fills with 5 and 7
            assert uc._UrbanCentre__filled_array[1, 3] == fills[0]
            # fills with 5 but not 7
            assert uc._UrbanCentre__filled_array[1, 4] == fills[1]
            # doesn't fill (checks if outside bounds are 0)
            assert uc._UrbanCentre__filled_array[4, 0] == fills[2]


# test nodata parameter
@pytest.mark.parametrize(
    "v_nodata, expected",
    [
        (-200, does_not_raise()),
        (
            -200.5,
            pytest.raises(
                TypeError, match=(r"`nodata` expected integer, got float")
            ),
        ),
        (
            "str",
            pytest.raises(
                TypeError, match=(r"`nodata` expected integer, got str")
            ),
        ),
    ],
)
def test_v_nodata(
    dummy_pop_array: str,
    bbox: gpd.GeoDataFrame,
    cluster_centre: tuple,
    v_nodata: int,
    expected: Type[RaisesContext],
):
    """Test vector_nodata parameter.

    Parameters
    ----------
    dummy_pop_array : str
        Filepath to dummy raster data.
    bbox : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    cluster_centre : tuple
        Coordinates for the centre of the cluster.
    v_nodata : int
        Value to fill cells with no data.
    expected : Type[RaisesContext]
        Expected raise result.

    """
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, cluster_centre, vector_nodata=v_nodata
            )
            is not None
        )


# test buffer parameter
@pytest.mark.parametrize(
    "buffer, expected",
    [
        (10000, does_not_raise()),
        (
            -10000,
            pytest.raises(
                ValueError,
                match=(r"`buffer_size` expected positive non-zero integer"),
            ),
        ),
        (
            10000.5,
            pytest.raises(
                TypeError, match=(r"`buffer_size` expected int, got float")
            ),
        ),
        (
            "str",
            pytest.raises(
                TypeError, match=(r"`buffer_size` expected int, got str")
            ),
        ),
    ],
)
def test_buffer(
    dummy_pop_array: str,
    bbox: gpd.GeoDataFrame,
    cluster_centre: tuple,
    buffer: int,
    expected: Type[RaisesContext],
):
    """Test buffer parameter.

    Parameters
    ----------
    dummy_pop_array : str
        Filepath to dummy raster data.
    bbox : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    cluster_centre : tuple
        Coordinates for the centre of the cluster.
    buffer : int
        Size of the buffer around urban centres, in metres.
    expected : Type[RaisesContext]
        Expected raise result.

    """
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, cluster_centre, buffer_size=buffer
            )
            is not None
        )


# test intermediate output types
@pytest.mark.parametrize(
    "output, expected",
    [
        ("_UrbanCentre__windowed_array", np.ndarray),
        ("aff", affine.Affine),
        ("crs", rio.crs.CRS),
        ("_UrbanCentre__pop_filt_array", np.ndarray),
        ("_UrbanCentre__cluster_array", np.ndarray),
        ("_UrbanCentre__num_clusters", int),
        ("_UrbanCentre__urban_centres_array", np.ndarray),
        ("_UrbanCentre__filled_array", np.ndarray),
        ("_UrbanCentre__vectorized_uc", gpd.GeoDataFrame),
        ("_UrbanCentre__buffer", gpd.GeoDataFrame),
        ("_UrbanCentre__uc_buffer_bbox", gpd.GeoDataFrame),
    ],
)
def test_output_types(
    dummy_pop_array: str,
    bbox: gpd.GeoDataFrame,
    cluster_centre: tuple,
    output: str,
    expected: Type[type],
):
    """Test intermediate outputs.

    Parameters
    ----------
    dummy_pop_array : str
        Filepath to dummy raster data.
    bbox : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    cluster_centre : tuple
        Coordinates for the centre of the cluster.
    output : str
        Name of the intermediate output attribute.
    expected : Type[type]
        Expected type.

    """
    obj = ucc.UrbanCentre(dummy_pop_array)
    obj.get_urban_centre(bbox, cluster_centre)

    assert type(getattr(obj, output)) == expected


# test final output characteristics using defaults
def test_final_output(
    dummy_pop_array: str, bbox: gpd.GeoDataFrame, cluster_centre: tuple
):
    """Test final output.

    Parameters
    ----------
    dummy_pop_array : str
        Filepath to dummy raster data.
    bbox : gpd.GeoDataFrame
        Boundaries of the bounding box to filter the raster.
    cluster_centre : tuple
        Coordinates for the centre of the cluster.

    """
    out = ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
        bbox, cluster_centre
    )

    # uc expected coordinates
    # coordinates will need to be recalculated if array fixture changes
    # you can just do list(Polygon.exterior.coords) to get coordinates
    uc_coords = [
        (-243000.0, 6056000.0),
        (-243000.0, 6052000.0),
        (-240000.0, 6052000.0),
        (-240000.0, 6053000.0),
        (-238000.0, 6053000.0),
        (-238000.0, 6056000.0),
        (-243000.0, 6056000.0),
    ]
    assert out.loc[0][1] == Polygon(uc_coords)

    # bbox expected coordinates
    bbox_coords = [
        (-253000.0, 6042000.0),
        (-228000.0, 6042000.0),
        (-228000.0, 6066000.0),
        (-253000.0, 6066000.0),
        (-253000.0, 6042000.0),
    ]
    assert out.loc[2][1] == Polygon(bbox_coords)

    # type of output
    assert type(out) == gpd.GeoDataFrame

    # shape of output
    assert out.shape == (3, 2)

    # names of columns
    assert list(out.columns) == ["label", "geometry"]

    # check that all geometry are Polygons (not MultiPolygons)
    assert dict(out.type) == {0: "Polygon", 1: "Polygon", 2: "Polygon"}

    # check output crs
    assert out.crs == "ESRI: 54009"
