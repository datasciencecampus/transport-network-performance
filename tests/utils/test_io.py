"""Tests for utils/io.py."""

import geopandas as gpd
import os
import pytest
import re

from geopandas.testing import assert_geodataframe_equal
from pyprojroot import here
from _pytest.python_api import RaisesContext
from shapely.geometry import Point
from typing import Type

from transport_performance.utils.io import to_pickle, from_pickle


@pytest.fixture
def test_gdf() -> gpd.GeoDataFrame:
    """A small geodataframe fixture to test pickling IO functions."""
    data = {
        "name": ["A", "B"],
        "var": [1, 2],
        "state": [True, False],
        "geometry": [Point(0, 0), Point(90, 0)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


class TestIOPickle:
    """A class to test the `to_pickle()` and `from_pickle` functions."""

    def test_to_from_pickle_on_pass(
        self, test_gdf: gpd.GeoDataFrame, tmp_path: str
    ) -> None:
        """Check correct behaviour of `to_pickle` and `from_pickle`.

        Parameters
        ----------
        test_gdf : gpd.GeoDataFrame
            Test fixture to pickle and check against.
        tmp_path : str
            Temporary directory to work within.

        """
        # create a working directory, with a new sub folder and a test file
        temp_file_path = os.path.join(tmp_path, "test_dir", "test.pkl")

        # test to pickle, and make sure the file exists
        to_pickle(test_gdf, temp_file_path)
        assert os.path.exists(temp_file_path)

        # re-read in file, and make sure it is equivalent to the input fixture
        gdf_in = from_pickle(temp_file_path)
        assert_geodataframe_equal(test_gdf, gdf_in)

    def test_to_pickle_on_warns(
        self, test_gdf: gpd.GeoDataFrame, tmp_path: str
    ) -> None:
        """Check `to_pickle()` warns when an invalid file extension is given.

        Parameters
        ----------
        test_gdf : gpd.GeoDataFrame
            Test fixture to pickle.
        tmp_path : str
            Path to a temporary working directory.

        """
        # use a false .test file extension and ensure correct coercion to .pkl
        test_file_path = os.path.join(tmp_path, "test.test")
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "Format .test provided. Expected ['pkl', 'pickle'] "
                "for path given to 'path'. Path defaulted to .pkl"
            ),
        ):
            to_pickle(test_gdf, test_file_path)
            assert os.path.exists(os.path.join(tmp_path, "test.pkl"))

    def test_to_pickle_on_fail(
        self, test_gdf: gpd.GeoDataFrame, tmp_path: str
    ) -> None:
        """Check `to_pickle()` fails when an invalid path type is given.

        Parameters
        ----------
        test_gdf : gpd.GeoDataFrame
            Test fixture to pickle.
        tmp_path : str
            Path to a temporary working directory.

        """
        # use a float as the path
        with pytest.raises(
            TypeError, match="`path` expected path-like, found .*float.*"
        ):
            to_pickle(test_gdf, 0.0)

    @pytest.mark.parametrize(
        "path, expected",
        [
            (
                "test.pkl",  # does not exist
                pytest.raises(
                    FileNotFoundError,
                    match="not found",
                ),
            ),
            (
                "test",  # no file extension
                pytest.raises(
                    ValueError,
                    match="No file extension was found in",
                ),
            ),
            (
                here("tests/data/newport-2023-06-13.osm.pbf"),  # wrong ext
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "`path` expected file extension ['.pkl', '.pickle']. "
                        "Found .pbf"
                    ),
                ),
            ),
            (
                0.0,  # wrong type
                pytest.raises(
                    TypeError,
                    match="expected .*str.*pathlib.Path.* Got .*float",
                ),
            ),
        ],
    )
    def test_from_pickle_on_fail(
        self, path: str, expected: Type[RaisesContext]
    ) -> None:
        """Test `from_pickle()` failure cases.

        Parameters
        ----------
        path : str
            Test path to use with `from_pickle()`.
        expected : Type[RaisesContext]
            Expected raises.

        """
        with expected:
            from_pickle(path)
