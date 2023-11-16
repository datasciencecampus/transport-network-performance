"""Common metrics pytest fixtures.

These have been grouped to improve modularity and reuse of fixtures.

The pytest fixture scope has been set to 'session' where possible such that
they are constructed only once (performance benefit) and then used as inputs
where needed. Fixures marked with the 'session' scope are not modified by unit
tests that use them, so reuse between tests will not present issues.

"""

import geopandas as gpd
import os
import pathlib
import pandas as pd
import pytest

from pyprojroot import here
from typing import Union

from transport_performance.utils.io import from_pickle


@pytest.fixture(scope="session")
def uc_fixture() -> gpd.GeoDataFrame:
    """Retrieve mock urban centre test fixture."""
    UC_FIXTURE_PATH = here("tests/data/metrics/mock_urban_centre.pkl")
    return from_pickle(UC_FIXTURE_PATH)


@pytest.fixture(scope="session")
def centroid_gdf_fixture() -> gpd.GeoDataFrame:
    """Retrieve mock centroid_gdf test fixture."""
    CENTROID_GDF_FIXTURE_PATH = here(
        "tests/data/metrics/mock_centroid_gdf.pkl"
    )
    return from_pickle(CENTROID_GDF_FIXTURE_PATH)


@pytest.fixture(scope="session")
def pop_gdf_fixture() -> gpd.GeoDataFrame:
    """Retrieve mock pop_gdf test fixture."""
    POP_GDF_FIXTURE_PATH = here("tests/data/metrics/mock_pop_gdf.pkl")
    return from_pickle(POP_GDF_FIXTURE_PATH)


@pytest.fixture(scope="session")
def tt_fixture() -> pathlib.Path:
    """Retrieve mock travel times test fixture."""
    return here("tests/data/metrics/mock_tt.parquet")


@pytest.fixture(scope="session")
def expected_transport_performance() -> Union[
    list, pd.DataFrame, pd.DataFrame
]:
    """Transport performance results fixture.

    Expected results when using centroid_gdf_fixture, pop_gdf_fixture,
    uc_fixture, and tt_fixture as inputs to `transport_performance()` and
    tp_utils.py functions.

    """
    TEST_COLS = [
        "id",
        "accessible_population",
        "proximity_population",
        "transport_performance",
    ]
    expected_tp_df = pd.DataFrame(
        [
            [5, 32, 46, (32 / 46 * 100)],
            [6, 26, 42, (26 / 42 * 100)],
            [9, 20, 39, (20 / 39 * 100)],
            [10, 20, 41, (20 / 41 * 100)],
        ],
        columns=TEST_COLS,
    )
    expected_stats_df = pd.DataFrame(
        [
            [
                "name",
                "country",
                0.04,
                34,
                expected_tp_df.transport_performance.min(),
                expected_tp_df.transport_performance.quantile(0.25),
                expected_tp_df.transport_performance.median(),
                expected_tp_df.transport_performance.quantile(0.75),
                expected_tp_df.transport_performance.max(),
            ],
        ],
        columns=[
            "urban centre name",
            "urban centre country",
            "urban centre area",
            "urban centre population",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ],
    )

    return TEST_COLS, expected_tp_df, expected_stats_df


@pytest.fixture
def multi_tt_fixture(tt_fixture, tmp_path) -> pathlib.Path:
    """Build a mock travel time input across multiple parquet files."""
    multi_tt_path = os.path.join(tmp_path, "mock_multi_tt")
    os.makedirs(multi_tt_path)
    tt = pd.read_parquet(tt_fixture)
    for id in tt.to_id.unique():
        tt[tt.to_id == id].to_parquet(
            os.path.join(multi_tt_path, f"mock_tt_id{id}.parquet")
        )

    return pathlib.Path(multi_tt_path)


@pytest.fixture()
def non_parquet_extension(tmp_path) -> pathlib.Path:
    """Create an directory fixture with a parquet and csv file within a dir."""
    non_parquet_dir = os.path.join(tmp_path, "non_parquet_dir")
    os.makedirs(non_parquet_dir)
    pd.DataFrame().to_parquet(os.path.join(non_parquet_dir, "test.parquet"))
    pd.DataFrame().to_csv(os.path.join(non_parquet_dir, "test.csv"))

    return pathlib.Path(non_parquet_dir)


@pytest.fixture()
def empty_directory(tmp_path) -> pathlib.Path:
    """Create an empty path for use in a fixture."""
    # tmp_path not useable inside a parameterised test
    return pathlib.Path(tmp_path)
