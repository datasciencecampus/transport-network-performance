"""Unit tests for transport_performance/urban_centres/urban_centres_class.

TODO: add docs.

Note: in the class parameterised tests below there are some arguments that are
not used across all tests within them. This is a deliberate design choice,
since pytest expects all parameterised arguments to be passed - removing or
excluding from a signle test triggers errors. The alternative would be to
separate the tests and reparameterise each separetly, but this would lead to a
larger codebase that is more difficult to maintain.
"""

import os

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from pytest_lazyfixture import lazy_fixture
from shapely.geometry import Polygon

import transport_performance.urban_centres.raster_uc as ucc


# fixtures
@pytest.fixture
def dummy_pop_array(tmp_path: str):
    """Create dummy population array."""
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
    """Create dummy bbox."""
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
    """Create dummy bbox that does not overlap with raster."""
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
    """Create dummy bbox with wrong crs."""
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
    """Create dummy cluster centre."""
    return (-242000.0, 6055000.0)


# tests
# test exceptions for file path
@pytest.mark.parametrize(
    "filepath, func, expected",
    [
        (lazy_fixture("dummy_pop_array"), "str", does_not_raise()),
        (lazy_fixture("dummy_pop_array"), "path", does_not_raise()),
        # wrong path
        ("wrongpath.tif", "str", pytest.raises(IOError)),
        # wrong extension
        ("wrongpath", "str", pytest.raises(ValueError)),
    ],
)
def test_file(filepath, func, bbox, cluster_centre, expected):
    """Test filepath."""
    if func == "str":
        filepath = str(filepath)
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
        ("string", pytest.raises(TypeError)),
        (pd.DataFrame(), pytest.raises(TypeError)),
        # badly defined bbox
        (gpd.GeoDataFrame(), pytest.raises(AttributeError)),
        # bbox not overlapping
        (lazy_fixture("non_overlapping_bbox"), pytest.raises(ValueError)),
        # wrong crs bbox
        (lazy_fixture("wrong_crs_bbox"), pytest.raises(ValueError)),
    ],
)
def test_bbox(dummy_pop_array, window, cluster_centre, expected):
    """Test bounding box."""
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
        ((-235000.0, 6055000.0), None, pytest.raises(ValueError)),
        # outside bbox
        ((-200000.0, 6055000.0), None, pytest.raises(IndexError)),
        # check tuple constrains
        ((50, 3), None, pytest.raises(TypeError)),
        ((50, 3, 3), None, pytest.raises(ValueError)),
        (50, None, pytest.raises(TypeError)),
        ("(50, 3)", None, pytest.raises(TypeError)),
    ],
)
def test_centre(dummy_pop_array, bbox, centre_coords, centre_crs, expected):
    """Test centre."""
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
        (1.5, pytest.raises(TypeError)),
        (2, pytest.raises(IndexError)),
        ("2", pytest.raises(TypeError)),
    ],
)
def test_band_n(dummy_pop_array, bbox, cluster_centre, band, expected):
    """Test raster band parameter."""
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
        (1500.5, pytest.raises(TypeError), []),
        ("1500", pytest.raises(TypeError), []),
        # tests value that would not create any cluster
        (150000, pytest.raises(ValueError), []),
    ],
)
class TestCellPop:
    """Class to test effect of cell pop threshold on output."""

    def test_cell_pop_t(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        cell_pop_t,
        expected,
        flags,
    ):
        """Test cell_pop_threshold parameter."""
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, cell_pop_threshold=cell_pop_t
                )
                is not None
            )

    def test_cell_pop_t_output(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        cell_pop_t,
        expected,
        flags,
    ):
        """Test cell_pop_threshold output."""
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
        (1, pytest.raises(TypeError), 0, 0),
        ("True", pytest.raises(TypeError), 0, 0),
    ],
)
class TestDiag:
    """Class to test effect of diagonal boolean on output."""

    def test_diag(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        diagonal,
        expected,
        cluster,
        num_clusters,
    ):
        """Test diag parameter."""
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, diag=diagonal
                )
                is not None
            )

    def test_diag_output(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        diagonal,
        expected,
        cluster,
        num_clusters,
    ):
        """Test diag parameter output."""
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
        (50000.5, pytest.raises(TypeError), []),
        ("50000", pytest.raises(TypeError), []),
        # test value that would filter out all clusters
        (1000000, pytest.raises(ValueError), []),
    ],
)
class TestClusterPop:
    """Class to test effect of clustering pop threshold on output."""

    def test_cluster_pop_t(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        cluster_pop_t,
        expected,
        clusters,
    ):
        """Test pop_threshold parameter."""
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, cluster_pop_threshold=cluster_pop_t
                )
                is not None
            )

    def test_cluster_pop_t_output(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        cluster_pop_t,
        expected,
        clusters,
    ):
        """Test pop_threshold outputs."""
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
        (5.5, pytest.raises(TypeError), []),
        ("5", pytest.raises(TypeError), []),
        # test values outside boundaries
        (11, pytest.raises(ValueError), []),
        (0, pytest.raises(ValueError), []),
    ],
)
class TestFill:
    """Class to test effect of fill threshold on output."""

    def test_cell_fill_t(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        cell_fill_t,
        expected,
        fills,
    ):
        """Test cell_fill_threshold parameter."""
        with expected:
            assert (
                ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                    bbox, cluster_centre, cell_fill_threshold=cell_fill_t
                )
                is not None
            )

    def test_cell_fill_output(
        self,
        dummy_pop_array,
        bbox,
        cluster_centre,
        cell_fill_t,
        expected,
        fills,
    ):
        """Test fill output."""
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
        (-200.5, pytest.raises(TypeError)),
        ("str", pytest.raises(TypeError)),
    ],
)
def test_v_nodata(dummy_pop_array, bbox, cluster_centre, v_nodata, expected):
    """Test vector_nodata parameter."""
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
        (-10000, pytest.raises(ValueError)),
        (10000.5, pytest.raises(TypeError)),
        ("str", pytest.raises(TypeError)),
    ],
)
def test_buffer(dummy_pop_array, bbox, cluster_centre, buffer, expected):
    """Test buffer parameter."""
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
def test_output_types(dummy_pop_array, bbox, cluster_centre, output, expected):
    """Test intermediate outputs."""
    obj = ucc.UrbanCentre(dummy_pop_array)
    obj.get_urban_centre(bbox, cluster_centre)

    assert type(getattr(obj, output)) == expected


# test final output characteristics using defaults
def test_final_output(dummy_pop_array, bbox, cluster_centre):
    """Test final output."""
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
