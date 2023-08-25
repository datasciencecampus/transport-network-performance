"""Unit tests for transport_performance/urban_centres/urban_centres_class.

TODO: add docs.
"""

import os

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio

from contextlib import nullcontext as does_not_raise
from pytest_lazyfixture import lazy_fixture
from shapely.geometry import Polygon

import transport_performance.urban_centres.raster_uc as ucc


@pytest.fixture
def dummy_pop_array(tmp_path: str):
    """Create dummy population array."""
    aff = rio.Affine(1000.0, 0.0, -243000.0, 0.0, -1000.0, 6056000.0)
    crs = rio.CRS.from_string("ESRI: 54009")
    dummy = np.array(
        [
            [5000, 5000, 5000, 1500, 1500, 0, 0, 0, 5000, 5000],
            [5000, 5000, 5000, 0, 0, 0, 0, 0, 0, 0],
            [5000, 5000, 5000, 1500, 1500, 0, 0, 0, 0, 0],
            [1500, 1500, 1500, 0, 0, 0, 0, 0, 0, 0],
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
    return (51.74, -3.25)


@pytest.fixture
def outside_cluster_centre():
    """Create dummy cluster centre outside of raster boundaries."""
    return (41.74, -13.25)


# test exceptions for input parameters
@pytest.mark.parametrize(
    "filepath, expected",
    [
        (lazy_fixture("dummy_pop_array"), does_not_raise()),
        ("wrongpath", pytest.raises(IOError)),
    ],
)
def test_file(filepath, bbox, cluster_centre, expected):
    """Test filepath."""
    with expected:
        assert (
            ucc.UrbanCentre(filepath).get_urban_centre(bbox, cluster_centre)
            is not None
        )


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


@pytest.mark.parametrize(
    "centre_coords, expected",
    [
        (lazy_fixture("cluster_centre"), does_not_raise()),
        (lazy_fixture("outside_cluster_centre"), pytest.raises(IndexError)),
        ((50, 3), pytest.raises(TypeError)),
        ((50, 3, 3), pytest.raises(ValueError)),
        (50, pytest.raises(TypeError)),
        ("(50, 3)", pytest.raises(TypeError)),
    ],
)
def test_centre(dummy_pop_array, bbox, centre_coords, expected):
    """Test centre."""
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, centre=centre_coords
            )
            is not None
        )


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


@pytest.mark.parametrize(
    "cell_pop_t, expected",
    [
        (1500, does_not_raise()),
        (1500.5, pytest.raises(TypeError)),
        ("1500", pytest.raises(TypeError)),
        # tests value that would not create any cluster
        (150000, pytest.raises(ValueError)),
    ],
)
def test_cell_pop_t(
    dummy_pop_array, bbox, cluster_centre, cell_pop_t, expected
):
    """Test cell_pop_threshold parameter."""
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, cluster_centre, cell_pop_threshold=cell_pop_t
            )
            is not None
        )


@pytest.mark.parametrize(
    "diagonal, expected",
    [
        (True, does_not_raise()),
        (False, does_not_raise()),
        ("True", pytest.raises(TypeError)),
    ],
)
def test_diag(dummy_pop_array, bbox, cluster_centre, diagonal, expected):
    """Test diag parameter."""
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, cluster_centre, diag=diagonal
            )
            is not None
        )


@pytest.mark.parametrize(
    "cluster_pop_t, expected",
    [
        (50000, does_not_raise()),
        (50000.5, pytest.raises(TypeError)),
        ("50000", pytest.raises(TypeError)),
        # test value that would filter out all clusters
        (1000000, pytest.raises(ValueError)),
    ],
)
def test_cluster_pop_t(
    dummy_pop_array, bbox, cluster_centre, cluster_pop_t, expected
):
    """Test pop_threshold parameter."""
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, cluster_centre, cluster_pop_threshold=cluster_pop_t
            )
            is not None
        )


@pytest.mark.parametrize(
    "cell_fill_t, expected",
    [
        (5, does_not_raise()),
        (5.5, pytest.raises(TypeError)),
        ("5", pytest.raises(TypeError)),
        # test values outside boundaries
        (11, pytest.raises(ValueError)),
        (0, pytest.raises(ValueError)),
    ],
)
def test_cell_fill_t(
    dummy_pop_array, bbox, cluster_centre, cell_fill_t, expected
):
    """Test cell_fill_threshold parameter."""
    with expected:
        assert (
            ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
                bbox, cluster_centre, cell_fill_treshold=cell_fill_t
            )
            is not None
        )


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


def test_final_output(dummy_pop_array, bbox, cluster_centre):
    """Test final output."""
    out = ucc.UrbanCentre(dummy_pop_array).get_urban_centre(
        bbox, cluster_centre
    )
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