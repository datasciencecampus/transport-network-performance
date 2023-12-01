"""Fixtures for transport_performance/analyse_network/analyse network."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from pyprojroot import here
from shapely.geometry import Point

import transport_performance.analyse_network as an


############
# fixtures #
############

# gtfs file
@pytest.fixture(scope="module")
def dummy_gtfs():
    """Create fixture with dummy gtfs file.

    Returns
    -------
    list
        List with paths to dummy GTFS files.

    """
    return [here("tests/data/gtfs/newport-20230613_gtfs.zip")]


# pbf file
@pytest.fixture(scope="module")
def dummy_osm():
    """Create fixture with dummy osm file.

    Returns
    -------
    pathlib.Path
        Path to dummy PBF file.

    """
    return here("tests/data/newport-2023-06-13.osm.pbf")


# centroids
@pytest.fixture(scope="module")
def dummy_gdf_centroids():
    """Create fixture with dummy gtfs file.

    Returns
    -------
    gpd.GeoDataFrame
        Dummy geodataframe with four centroids within the boundaries of PBF
        and GTFS dummy files.

    """
    points = {
        "id": [1, 2, 3, 4],
        "geometry": [
            Point(-2.999681677472678, 51.58859432106067),
            Point(-3.002464225582309, 51.59023553677067),
            Point(-2.9966994427043026, 51.58796089656915),
            Point(-3.0026994196377954, 51.587209140699315),
        ],
        "within_urban_centre": [0, 1, 0, 1],
    }
    return gpd.GeoDataFrame(points, crs="epsg: 4326")


# AnalyseNetwork object
@pytest.fixture(scope="module")
def dummy_transport_network(dummy_gdf_centroids, dummy_osm, dummy_gtfs):
    """Create fixture with AnalyseNetwork object.

    Returns
    -------
    AnalyseNetwork
        Initialised AnalyseNetwork object created from PBF, GTFS and centroids
        fixtures.

    Notes
    -----
    This object is mainly used to call analyse_network methods in tests.

    """
    return an.AnalyseNetwork(dummy_gdf_centroids, dummy_osm, dummy_gtfs)


# r5py TransportNetwork object
@pytest.fixture(scope="module")
def dummy_r5py_tn(dummy_transport_network):
    """Create fixture with dummy r5py TransportNetwork object.

    Returns
    -------
    r5py.TransportNetwork
        r5py.TransportNetwork object.

    Notes
    -----
    This fixture is used to test defences in the internal function
    _calculate_transport_network.

    """
    return dummy_transport_network.transport_network


# small o-d matrix dataframe fixture
@pytest.fixture(scope="module")
def dummy_od_matrix():
    """Create fixture with dummy O-D matrix.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with two origins and two destinations.

    Notes
    -----
    This object is a pandas DataFrame and not a GeoDataFrame, as used by
    internal functions.

    """
    points = {
        "id_orig": [1, 2, 1, 2],
        "geometry_orig": [
            Point(-2.999681677472678, 51.58859432106067),
            Point(-3.002464225582309, 51.59023553677067),
            Point(-2.999681677472678, 51.58859432106067),
            Point(-3.002464225582309, 51.59023553677067),
        ],
        "id_dest": [3, 3, 4, 4],
        "geometry_dest": [
            Point(-2.9966994427043026, 51.58796089656915),
            Point(-2.9966994427043026, 51.58796089656915),
            Point(-3.0026994196377954, 51.587209140699315),
            Point(-3.0026994196377954, 51.587209140699315),
        ],
    }
    return pd.DataFrame(points)


# big dataframe to test splitting
@pytest.fixture(scope="module")
def dummy_big_df():
    """Create fixture with a random big dataframe.

    Returns
    -------
    pd.DataFrame
        Big dataframe.

    Notes
    -----
    This is created to test the calculate partitions behaviour of internal
    function _estimate_num_partitions.

    """
    df = pd.DataFrame(np.random.rand(10000, 1000))
    # parquet doesn't like non-string column names
    df.columns = [str(x) for x in df.columns]
    return df


# temporary directory to save parquet
@pytest.fixture
def dummy_filepath(tmp_path, scope="function"):
    """Create fixture with output path.

    Returns
    -------
    pathlib.Path
        Path to the temporary directory to save files.

    """
    tempdir = tmp_path / "parquet_files"
    tempdir.mkdir()
    return tempdir
