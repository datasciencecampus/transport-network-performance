"""Unit tests for transport_performance/analyse_network/analyse network."""
import datetime
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pathlib
import pytest

from contextlib import nullcontext as does_not_raise
from haversine import Unit
from pyprojroot import here
from pytest_lazyfixture import lazy_fixture
from r5py import TransportNetwork, TransportMode
from shapely.geometry import Point
from typing import Type, Union
from _pytest.python_api import RaisesContext

import transport_performance.analyse_network.analyse_network as an


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


#############################
# test class initialisation #
#############################


@pytest.mark.parametrize(
    "gdf, osm, gtfs, expected",
    [
        # no error raised
        (
            lazy_fixture("dummy_gdf_centroids"),
            lazy_fixture("dummy_osm"),
            lazy_fixture("dummy_gtfs"),
            does_not_raise(),
        ),
        # wrong centroids gdf
        (
            "not a gdf",
            lazy_fixture("dummy_osm"),
            lazy_fixture("dummy_gtfs"),
            pytest.raises(
                TypeError,
                match=(r"`gdf` expected .*GeoDataFrame.*" r"Got .*str.*"),
            ),
        ),
        # wrong osm path
        (
            lazy_fixture("dummy_gdf_centroids"),
            ["not a path or string"],
            lazy_fixture("dummy_gtfs"),
            pytest.raises(
                TypeError,
                match=(r"`pth` expected .*str.*Path.*" r"Got .*list.*"),
            ),
        ),
        # wrong gtfs list
        (
            lazy_fixture("dummy_gdf_centroids"),
            lazy_fixture("dummy_osm"),
            {"not a list"},
            pytest.raises(
                TypeError, match=(r"`gtfs` expected .*list.*" r"Got .*set.*")
            ),
        ),
    ],
)
def test_init(
    gdf: gpd.GeoDataFrame,
    osm: Union[str, pathlib.Path],
    gtfs: list,
    expected: Type[RaisesContext],
):
    """Tests AnalyseNetwork class initialisation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with centroids within transport network boundaries.
    osm : Union[str, pathlib.Path]
        Path to pbf file.
    gtfs : list
        List with path(s) to GTFS files.
    expected : Type[RaisesContext]
        Expected raise result.

    """
    with expected:
        assert an.AnalyseNetwork(gdf, osm, gtfs)


###########################
# test internal functions #
###########################

# _calculate_transport_network
@pytest.mark.parametrize(
    "transport_network, r5py_tn, origins, destinations, departure, expected",
    [
        # no error raised
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_r5py_tn"),
            lazy_fixture("dummy_gdf_centroids"),
            lazy_fixture("dummy_gdf_centroids"),
            datetime.datetime(2023, 6, 13, 8, 0),
            does_not_raise(),
        ),
        # wrong r5py.TransportNetwork object
        (
            lazy_fixture("dummy_transport_network"),
            "not an r5py transport network",
            lazy_fixture("dummy_gdf_centroids"),
            lazy_fixture("dummy_gdf_centroids"),
            datetime.datetime(2023, 6, 13, 8, 0),
            pytest.raises(
                TypeError,
                match=(
                    r"`transport_network` expected .*r5py.*TransportNetwork.*"
                    r"Got .*str.*"
                ),
            ),
        ),
        # wrong origins
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_r5py_tn"),
            "not an origins gdf",
            lazy_fixture("dummy_gdf_centroids"),
            datetime.datetime(2023, 6, 13, 8, 0),
            pytest.raises(
                TypeError,
                match=(r"`origins` expected .*GeoDataFrame.*" r"Got .*str.*"),
            ),
        ),
        # date out of range
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_r5py_tn"),
            lazy_fixture("dummy_gdf_centroids"),
            lazy_fixture("dummy_gdf_centroids"),
            datetime.datetime(2015, 6, 13, 8, 0),
            pytest.raises(
                IndexError,
                match=(
                    r"Date provided is outside of the time range included in "
                    r"the GTFS provided, or TransportNetwork does not contain "
                    r"a valid GTFS."
                ),
            ),
        ),
        # wrong destinations
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_r5py_tn"),
            lazy_fixture("dummy_gdf_centroids"),
            "not a destinations gdf",
            datetime.datetime(2023, 6, 13, 8, 0),
            pytest.raises(
                TypeError,
                match=(
                    r"`destinations` expected .*GeoDataFrame.*" r"Got .*str.*"
                ),
            ),
        ),
    ],
)
class Test_calculate_transport_network:
    """Class to test the _calculate_transport_network internal function."""

    def test__calculate_transport_network_inputs(
        self,
        transport_network: an.AnalyseNetwork,
        r5py_tn: TransportNetwork,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        departure: datetime.datetime,
        expected: Type[RaisesContext],
    ):
        """Test _calculate_transport_network inputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        r5py_tn : TransportNetwork
            r5py TransportNetwork object.
        origins :  gpd.GeoDataFrame
            Geodataframe with origin points.
        destinations :  gpd.GeoDataFrame
            Geodataframe with destination points.
        departure : datetime.datetime
            Departure time.
        expected : Type[RaisesContext]
            Expected raise result.

        """
        with expected:
            assert [
                transport_network._calculate_transport_network(
                    r5py_tn, origins, destinations, departure=departure
                )
            ]

    def test__calculate_transport_network_outputs(
        self,
        transport_network: an.AnalyseNetwork,
        r5py_tn: TransportNetwork,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        departure: datetime.datetime,
        expected: Type[RaisesContext],
    ):
        """Test _calculate_transport_network output.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        r5py_tn : TransportNetwork
            r5py TransportNetwork object.
        origins :  gpd.GeoDataFrame
            Geodataframe with origin points.
        destinations :  gpd.GeoDataFrame
            Geodataframe with destination points.
        departure : datetime.datetime
            Departure time.
        expected : Type[RaisesContext]
            Expected raise result.

        """
        if expected == does_not_raise():
            output = transport_network._calculate_transport_network(
                r5py_tn, origins, destinations, departure=departure
            )
            assert isinstance(output, pd.DataFrame)
            assert len(output) == 16
            assert list(output.columns) == ["from_id", "to_id", "travel_time"]
            assert min(output["travel_time"]) == 0
            assert max(output["travel_time"]) == 12
            assert output.travel_time.mean() == 5.625


# _gdf_batch_origins
class Test_gdf_batch_origins:
    """Class to test the _gdf_batch_origins internal function."""

    @pytest.mark.parametrize(
        "transport_network, gdf, destination_col, distance, num_origins,"
        "unit, expected",
        [
            # no error raised
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                11.25,
                1,
                Unit.KILOMETERS,
                does_not_raise(),
            ),
            # wrong gdf
            (
                lazy_fixture("dummy_transport_network"),
                "not a geodataframe",
                "within_urban_centre",
                11.25,
                1,
                Unit.KILOMETERS,
                pytest.raises(
                    TypeError,
                    match=(r"`gdf` expected .*GeoDataFrame.* Got .*str.*"),
                ),
            ),
            # wrong column name
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "not_a_column_in_gdf",
                11.25,
                1,
                Unit.KILOMETERS,
                pytest.raises(
                    IndexError,
                    match=(
                        r"'not_a_column_in_gdf' is not a column in the "
                        r"dataframe"
                    ),
                ),
            ),
            # wrong distance
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                "wrong distance",
                1,
                Unit.KILOMETERS,
                pytest.raises(
                    TypeError,
                    match=(r"`distance` expected .*float.* Got .*str.*"),
                ),
            ),
            # wrong num_origins type
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                11.25,
                "wrong number",
                Unit.KILOMETERS,
                pytest.raises(
                    TypeError,
                    match=(r"`num_origins` expected .*int.* Got .*str.*"),
                ),
            ),
            # wrong num_origins number
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                11.25,
                17,
                Unit.KILOMETERS,
                pytest.raises(
                    ValueError,
                    match=(r"`num_origins` should be between 1 and 4, got 17"),
                ),
            ),
            # wrong unit
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                11.25,
                1,
                "not a unit",
                pytest.raises(
                    TypeError, match=(r"`unit` expected .*Unit.* Got .*str.*")
                ),
            ),
        ],
    )
    def test__gdf_batch_origins_inputs(
        self,
        transport_network: an.AnalyseNetwork,
        gdf: gpd.GeoDataFrame,
        destination_col: str,
        distance: float,
        num_origins: int,
        unit: Unit,
        expected: Type[RaisesContext],
    ):
        """Test _gdf_batch_origins inputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        gdf : gpd.GeoDataFrame
            Geodataframe with location ids.
        destination_col : str
            Column with flag indicating what points should be considered as
            destinations.
        distance : float
            Distance to filter destinations.
        num_origins : int
            Number of origins to consider in each loop.
        unit : Unit
            Unit to calculate distance.
        expected : Type[RaisesContext]
            Expected raise result.

        """
        generator = transport_network._gdf_batch_origins(
            gdf, destination_col, distance, num_origins, unit
        )
        with expected:
            assert next(generator)

    @pytest.mark.parametrize(
        "transport_network, gdf, destination_col, distance, num_origins,"
        "exp_output",
        [
            # defaults
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                11.25,
                1,
                [[1], [2, 4]],
            ),
            # changed distance
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                0.25,
                1,
                [[1], []],
            ),
            # changed batch size
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                11.25,
                4,
                [[1, 2, 3, 4], [2, 4]],
            ),
            # changed batch size and distance
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_gdf_centroids"),
                "within_urban_centre",
                0.25,
                4,
                [[1, 2, 3, 4], [2, 4]],
            ),
        ],
    )
    def test__gdf_batch_origins_outputs(
        self,
        transport_network: an.AnalyseNetwork,
        gdf: gpd.GeoDataFrame,
        destination_col: str,
        distance: float,
        num_origins: int,
        exp_output: list,
    ):
        """Test _gdf_batch_origins outputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        gdf : gpd.GeoDataFrame
            Geodataframe with location ids.
        destination_col : str
            Column with flag indicating what points should be considered as
            destinations.
        distance : float
            Distance to filter destinations.
        num_origins : int
            Number of origins to consider in each loop.
        exp_output : list
            Expected yield outputs for the first iteration of the generator.

        """
        generator = transport_network._gdf_batch_origins(
            gdf, destination_col, distance, num_origins
        )
        outputs = next(generator)
        assert list(outputs[0]) == exp_output[0]
        assert list(outputs[1]) == exp_output[1]


# _haversine_gdf
@pytest.mark.parametrize(
    "transport_network, df, orig, dest, unit, expected, output",
    [
        # no error raised
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_od_matrix"),
            "geometry_orig",
            "geometry_dest",
            Unit.KILOMETERS,
            does_not_raise(),
            np.array([0.21773847, 0.47178888, 0.25921125, 0.3369124]),
        ),
        # wrong df
        (
            lazy_fixture("dummy_transport_network"),
            "not a df",
            "geometry_orig",
            "geometry_dest",
            Unit.KILOMETERS,
            pytest.raises(
                TypeError, match=(r"`df` expected .*DataFrame.* Got .*str.*")
            ),
            [],
        ),
        # wrong geometry origin column
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_od_matrix"),
            "not_in_df",
            "geometry_dest",
            Unit.KILOMETERS,
            pytest.raises(
                IndexError,
                match=(r"'not_in_df' is not a column in the dataframe"),
            ),
            [],
        ),
        # wrong geometry destination column
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_od_matrix"),
            "geometry_orig",
            "not_in_df",
            Unit.KILOMETERS,
            pytest.raises(
                IndexError,
                match=(r"'not_in_df' is not a column in the dataframe"),
            ),
            [],
        ),
        # wrong unit
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_od_matrix"),
            "geometry_orig",
            "geometry_dest",
            "not a unit",
            pytest.raises(
                TypeError, match=(r"`unit` expected .*Unit.* Got .*str.*")
            ),
            [],
        ),
    ],
)
class Test_haversine_df:
    """Class to test the _haversine_gdf internal function."""

    def test__haversine_df_inputs(
        self,
        transport_network: an.AnalyseNetwork,
        df: pd.DataFrame,
        orig: str,
        dest: str,
        unit: Unit,
        expected: Type[RaisesContext],
        output: list,
    ):
        """Test _gdf_batch_origins inputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        df : pd.DataFrame
            Dataframe with coordinates for origins and destinations.
        orig : str
            Name of the column containing origin coordinates.
        dest : str
            Name of the column containing destination coordinates.
        unit : Unit
            Unit to calculate distance.
        expected : Type[RaisesContext]
            Expected raise result.
        output : list
            List with expected distances between pairs of points.

        """
        with expected:
            assert [transport_network._haversine_df(df, orig, dest, unit)]

    def test__haversine_df_outputs(
        self,
        transport_network: an.AnalyseNetwork,
        df: pd.DataFrame,
        orig: str,
        dest: str,
        unit: Unit,
        expected: Type[RaisesContext],
        output,
    ):
        """Test _gdf_batch_origins outputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        df : pd.DataFrame
            Dataframe with coordinates for origins and destinations.
        orig : str
            Name of the column containing origin coordinates.
        dest : str
            Name of the column containing destination coordinates.
        unit : Unit
            Unit to calculate distance.
        expected : Type[RaisesContext]
            Expected raise result.
        output : list
            List with expected distances between pairs of points.

        """
        if expected == does_not_raise():
            assert (
                transport_network._haversine_df(df, orig, dest, unit) == output
            )


# _estimate_num_partitions
@pytest.mark.parametrize(
    "transport_network, df, partition_size, expected, partitions",
    [
        # no error raised, 1 partition
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_big_df"),
            200,
            does_not_raise(),
            1,
        ),
        # no error raised, 8 partitions
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_big_df"),
            10,
            does_not_raise(),
            8,
        ),
        # wrong df
        (
            lazy_fixture("dummy_transport_network"),
            "not a dataframe",
            10,
            pytest.raises(
                TypeError, match=(r"`df` expected .*DataFrame.* Got .*str.*")
            ),
            0,
        ),
        # wrong partition size
        (
            lazy_fixture("dummy_transport_network"),
            lazy_fixture("dummy_big_df"),
            "not an integer",
            pytest.raises(
                TypeError,
                match=(r"`partition_size` expected .*int.* Got .*str.*"),
            ),
            0,
        ),
    ],
)
class Test_estimate_num_partitions:
    """Class to test the _estimate_num_partitions internal function."""

    def test__estimate_num_partitions_inputs(
        self,
        transport_network: an.AnalyseNetwork,
        df: pd.DataFrame,
        partition_size: int,
        expected: Type[RaisesContext],
        partitions: int,
    ):
        """Test _estimate_num_partitions inputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        df : pd.DataFrame
            Big dataframe with random numbers.
        partition_size : int
            Maximum size in MB for each parquet file partition.
        expected : Type[RaisesContext]
            Expected raise result.
        partitions : int
            Expected output.

        """
        with expected:
            assert transport_network._estimate_num_partitions(
                df, partition_size
            )

    def test__estimate_num_partitions_outputs(
        self,
        transport_network: an.AnalyseNetwork,
        df: pd.DataFrame,
        partition_size: int,
        expected: Type[RaisesContext],
        partitions: int,
    ):
        """Test _estimate_num_partitions outputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        df : pd.DataFrame
            Big dataframe with random numbers.
        partition_size : int
            Maximum size in MB for each parquet file partition.
        expected : Type[RaisesContext]
            Expected raise result.
        partitions : int
            Expected output.

        """
        if expected == does_not_raise():
            assert (
                transport_network._estimate_num_partitions(df, partition_size)
                == partitions
            )


# _save_to_parquet


class Test_save_to_parquet:
    """Class to test the _save_to_parquet internal function."""

    @pytest.mark.parametrize(
        "transport_network, df, out_name_func, out_path, npartitions, "
        "expected",
        [
            # wrong od matrix
            (
                lazy_fixture("dummy_transport_network"),
                "not a dataframe",
                "file_name",
                lazy_fixture("dummy_filepath"),
                1,
                pytest.raises(
                    TypeError,
                    match=(r"`od_matrix` expected .*DataFrame.* Got .*str.*"),
                ),
            ),
            # wrong filename func
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_big_df"),
                12345,
                lazy_fixture("dummy_filepath"),
                1,
                pytest.raises(
                    TypeError,
                    match=(r"`out_name_func` expected .*str.* Got .*int.*"),
                ),
            ),
            # wrong npartition
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_big_df"),
                "file_name",
                lazy_fixture("dummy_filepath"),
                "not an int",
                pytest.raises(
                    TypeError,
                    match=(r"`npartitions` expected .*int.* Got .*str.*"),
                ),
            ),
            # wrong out_path
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_big_df"),
                "file_name",
                ["wrong path"],
                1,
                pytest.raises(
                    TypeError,
                    match=(r"`out_path` expected path-like, found.*list.*"),
                ),
            ),
        ],
    )
    def test__save_to_parquet_input(
        self,
        transport_network: an.AnalyseNetwork,
        df: pd.DataFrame,
        out_name_func: str,
        out_path: pathlib.Path,
        npartitions: int,
        expected: Type[RaisesContext],
    ):
        """Test _save_to_parquet inputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        df : pd.DataFrame
            Big dataframe with random numbers.
        out_name_func : str
            String to add to the parquet filenames.
        out_path : pathlib.Path
            Path to temporary directory to save parquet files.
        npartitions : int
            Number of partitions to divide the parquet file.
        expected : Type[RaisesContext]
            Expected raise result.

        """
        if expected != does_not_raise():
            with expected:
                assert transport_network._save_to_parquet(
                    df, out_name_func, out_path, npartitions
                )

    @pytest.mark.parametrize(
        "transport_network, df, out_name_func, out_path, npartitions",
        [
            # no error raised, 1 partition
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_big_df"),
                "file_name",
                lazy_fixture("dummy_filepath"),
                1,
            ),
            # no error raised, 3 partitions
            (
                lazy_fixture("dummy_transport_network"),
                lazy_fixture("dummy_big_df"),
                "file_name",
                lazy_fixture("dummy_filepath"),
                11,
            ),
        ],
    )
    def test__save_to_parquet_output(
        self,
        transport_network: an.AnalyseNetwork,
        df: pd.DataFrame,
        out_name_func: str,
        out_path: pathlib.Path,
        npartitions,
    ):
        """Test _save_to_parquet outputs.

        Parameters
        ----------
        transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        df : pd.DataFrame
            Big dataframe with random numbers.
        out_name_func : str
            String to add to the parquet filenames.
        out_path : pathlib.Path
            Path to temporary directory to save parquet files.
        npartitions : int
            Number of partitions to divide the parquet file.

        """
        transport_network._save_to_parquet(
            df, out_name_func, out_path, npartitions
        )
        assert (
            len(glob.glob(os.path.join(out_path, f"*{out_name_func}*")))
            == npartitions
        )


#######################
# test module outputs #
#######################
def test_od_matrix(
    dummy_transport_network: an.AnalyseNetwork, dummy_filepath: pathlib.Path
):
    """Test main od_matrix method outputs.

    Parameters
    ----------
    dummy_transport_network : AnalyseNetwork
        Initialised AnalyseNetwork object created from PBF, GTFS and
        centroids fixtures.
    dummy_filepath : pathlib.Path
        Path to temporary directory to save parquet files.


    """
    tn = dummy_transport_network
    tn.od_matrix(
        num_origins=4,
        out_path=dummy_filepath,
        partition_size=200,
        destination_col="within_urban_centre",
        distance=11.25,
        unit=Unit.KILOMETERS,
        departure=datetime.datetime(2023, 6, 13, 8, 0),
        departure_time_window=datetime.timedelta(hours=1),
        max_time=datetime.timedelta(minutes=45),
        transport_modes=[TransportMode.TRANSIT],
    )

    assert glob.glob(os.path.join(dummy_filepath, "*.parquet")) == [
        os.path.join(dummy_filepath, "batch-1-0.parquet")
    ]
    loaded_output = pd.read_parquet(dummy_filepath)
    assert list(loaded_output["from_id"]) == [1, 1, 2, 2, 3, 3, 4, 4]
    assert list(loaded_output["to_id"]) == [2, 4, 2, 4, 2, 4, 2, 4]
    assert list(loaded_output["travel_time"]) == [9, 4, 0, 7, 12, 7, 7, 0]
