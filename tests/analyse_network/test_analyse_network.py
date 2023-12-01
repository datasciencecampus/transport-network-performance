"""Unit tests for transport_performance/analyse_network/analyse network."""
import datetime
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pathlib
import pytest
import r5py

from contextlib import nullcontext as does_not_raise
from haversine import Unit
from r5py import TransportMode
from typing import Type, Any
from _pytest.python_api import RaisesContext

import transport_performance.analyse_network as an

# import metrics fixtures via pytest_plugins
pytest_plugins = ["tests.analyse_network.analyse_network_fixtures"]

#############################
# test class initialisation #
#############################


@pytest.mark.parametrize(
    "arg_name, arg_value, expected",
    [
        # no error raised
        (
            None,
            None,
            does_not_raise(),
        ),
        # wrong centroids gdf
        (
            "gdf",
            "not a gdf",
            pytest.raises(
                TypeError,
                match=(r"`gdf` expected .*GeoDataFrame.*" r"Got .*str.*"),
            ),
        ),
        # wrong osm path
        (
            "osm",
            ["not a path or string"],
            pytest.raises(
                TypeError,
                match=(r"`pth` expected .*str.*Path.*" r"Got .*list.*"),
            ),
        ),
        # wrong gtfs list
        (
            "gtfs",
            {"not a list"},
            pytest.raises(
                TypeError, match=(r"`gtfs` expected .*list.*" r"Got .*set.*")
            ),
        ),
    ],
)
def test_init(
    arg_name: str,
    arg_value: Any,
    expected: Type[RaisesContext],
    dummy_gdf_centroids: gpd.GeoDataFrame,
    dummy_osm: pathlib.Path,
    dummy_gtfs: list,
):
    """Tests AnalyseNetwork class initialisation.

    Parameters
    ----------
    arg_name : str
        Name of function argument to test.
    arg_value : Any
        Value to use for argument being tested.
    expected : Type[RaisesContext]
        Expected raise result.
    dummy_gdf_centroids : gpd.GeoDataFrame
        Fixture with dummy centroid coordinates.
    dummy_osm : pathlib.Path
        Fixture with path to dummy pbf file.
    dummy_gtfs : list
        Fixture with path to dummy gtfs file. It is in a list as that's the
        format needed by the function (as it can take a list of paths).

    """
    # dict of default args
    default_args = {
        "gdf": dummy_gdf_centroids,
        "osm": dummy_osm,
        "gtfs": dummy_gtfs,
    }

    if arg_name is not None:
        default_args[arg_name] = arg_value

    with expected:
        assert an.AnalyseNetwork(**default_args)


###########################
# test internal functions #
###########################

# _calculate_transport_network
class Test_calculate_transport_network:
    """Class to test the _calculate_transport_network internal function."""

    @pytest.mark.parametrize(
        "arg_name, arg_value, expected",
        [
            # wrong r5py.TransportNetwork object
            (
                "transport_network",
                "not an r5py transport network",
                pytest.raises(
                    TypeError,
                    match=(
                        r"`transport_network` expected "
                        r".*r5py.*TransportNetwork.* Got .*str.*"
                    ),
                ),
            ),
            # wrong origins
            (
                "origins",
                "not an origins gdf",
                pytest.raises(
                    TypeError,
                    match=(r"`origins` expected .*GeoDataFrame.* Got.*str.*"),
                ),
            ),
            # wrong destinations
            (
                "destinations",
                "not a destinations gdf",
                pytest.raises(
                    TypeError,
                    match=(
                        r"`destinations` expected .*GeoDataFrame.* Got .*str.*"
                    ),
                ),
            ),
            # date out of range
            (
                "departure",
                datetime.datetime(2015, 6, 13, 8, 0),
                pytest.raises(
                    IndexError,
                    match=(
                        r"Date provided is outside of the time range included "
                        r"in the GTFS provided, or TransportNetwork does not "
                        r"contain a valid GTFS."
                    ),
                ),
            ),
        ],
    )
    def test__calculate_transport_network_inputs(
        self,
        arg_name: str,
        arg_value: Any,
        expected: Type[RaisesContext],
        dummy_transport_network: an.AnalyseNetwork,
        dummy_r5py_tn: r5py.TransportNetwork,
        dummy_gdf_centroids: gpd.GeoDataFrame,
    ):
        """Test _calculate_transport_network inputs.

        Parameters
        ----------
        arg_name : str
            Name of function argument to test.
        arg_value : Any
            Value to use for argument being tested.
        expected : Type[RaisesContext]
            Expected raise result.
        dummy_transport_network : an.AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_r5py_tn : r5py.TransportNetwork
            Fixture with r5py.TransportNetwork object.
        dummy_gdf_centroids : gpd.GeoDataFrame
            Fixture with dummy centroid coordinates.

        """
        # transport network
        analyse_network = dummy_transport_network

        # dict of default args
        default_args = {
            "transport_network": dummy_r5py_tn,
            "origins": dummy_gdf_centroids,
            "destinations": dummy_gdf_centroids,
            "departure": datetime.datetime(2023, 6, 13, 8, 0),
        }

        default_args[arg_name] = arg_value

        with expected:
            assert [
                analyse_network._calculate_transport_network(**default_args)
            ]

    def test__calculate_transport_network_outputs(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_r5py_tn: r5py.TransportNetwork,
        dummy_gdf_centroids: gpd.GeoDataFrame,
    ):
        """Test _calculate_transport_network output.

        Parameters
        ----------
        dummy_transport_network : an.AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_r5py_tn : r5py.TransportNetwork
            Fixture with r5py.TransportNetwork object.
        dummy_gdf_centroids : gpd.GeoDataFrame
            Fixture with dummy centroid coordinates.

        """
        # transport network
        analyse_network = dummy_transport_network

        # dict of default args
        default_args = {
            "transport_network": dummy_r5py_tn,
            "origins": dummy_gdf_centroids,
            "destinations": dummy_gdf_centroids,
            "departure": datetime.datetime(2023, 6, 13, 8, 0),
        }

        output = analyse_network._calculate_transport_network(**default_args)
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
        "arg_name, arg_value, expected",
        [
            # wrong gdf
            (
                "gdf",
                "not a geodataframe",
                pytest.raises(
                    TypeError,
                    match=(r"`gdf` expected .*GeoDataFrame.* Got .*str.*"),
                ),
            ),
            # wrong column name
            (
                "destination_col",
                "not_a_column_in_gdf",
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
                "distance",
                "wrong distance",
                pytest.raises(
                    TypeError,
                    match=(r"`distance` expected .*float.* Got .*str.*"),
                ),
            ),
            # wrong num_origins type
            (
                "num_origins",
                "wrong number",
                pytest.raises(
                    TypeError,
                    match=(r"`num_origins` expected .*int.* Got .*str.*"),
                ),
            ),
            # wrong num_origins number
            (
                "num_origins",
                17,
                pytest.raises(
                    ValueError,
                    match=(r"`num_origins` should be between 1 and 4, got 17"),
                ),
            ),
            # wrong unit
            (
                "unit",
                "not a unit",
                pytest.raises(
                    TypeError, match=(r"`unit` expected .*Unit.* Got .*str.*")
                ),
            ),
        ],
    )
    def test__gdf_batch_origins_inputs(
        self,
        arg_name: str,
        arg_value: Any,
        expected: Type[RaisesContext],
        dummy_transport_network: an.AnalyseNetwork,
        dummy_gdf_centroids: gpd.GeoDataFrame,
    ):
        """Test _gdf_batch_origins inputs.

        Parameters
        ----------
        arg_name : str
            Name of function argument to test.
        arg_value : Any
            Value to use for argument being tested.
        expected : Type[RaisesContext]
            Expected raise result.
        dummy_transport_network : an.AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_gdf_centroids : gpd.GeoDataFrame
            Fixture with dummy centroid coordinates.

        """
        # transport network
        analyse_network = dummy_transport_network

        default_args = {
            "gdf": dummy_gdf_centroids,
            "destination_col": "within_urban_centre",
            "distance": 11.25,
            "num_origins": 1,
            "unit": Unit.KILOMETERS,
        }

        default_args[arg_name] = arg_value

        generator = analyse_network._gdf_batch_origins(**default_args)
        with expected:
            assert next(generator)

    @pytest.mark.parametrize(
        "distance, num_origins, exp_output",
        [
            # defaults
            (
                11.25,
                1,
                [[1], [2, 4]],
            ),
            # changed distance
            (
                0.25,
                1,
                [[1], []],
            ),
            # changed batch size
            (
                11.25,
                4,
                [[1, 2, 3, 4], [2, 4]],
            ),
            # changed batch size and distance
            (
                0.25,
                4,
                [[1, 2, 3, 4], [2, 4]],
            ),
        ],
    )
    def test__gdf_batch_origins_outputs(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_gdf_centroids: gpd.GeoDataFrame,
        distance: float,
        num_origins: int,
        exp_output: list,
    ):
        """Test _gdf_batch_origins outputs.

        Parameters
        ----------
        dummy_transport_network : AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_gdf_centroids : gpd.GeoDataFrame
            Fixture with dummy centroid coordinates.
        distance : float
            Distance to filter destinations.
        num_origins : int
            Number of origins to consider in each loop.
        exp_output : list
            Expected yield outputs for the first iteration of the generator.

        """
        analyse_network = dummy_transport_network
        gdf = dummy_gdf_centroids

        generator = analyse_network._gdf_batch_origins(
            gdf, "within_urban_centre", distance, num_origins
        )
        outputs = next(generator)
        assert list(outputs[0]) == exp_output[0]
        assert list(outputs[1]) == exp_output[1]


# _haversine_gdf
class Test_haversine_df:
    """Class to test the _haversine_gdf internal function."""

    @pytest.mark.parametrize(
        "arg_name, arg_value, expected",
        [
            # wrong df
            (
                "df",
                "not a df",
                pytest.raises(
                    TypeError,
                    match=(r"`df` expected .*DataFrame.* Got .*str.*"),
                ),
            ),
            # wrong geometry origin column
            (
                "orig",
                "not_in_df",
                pytest.raises(
                    IndexError,
                    match=(r"'not_in_df' is not a column in the dataframe"),
                ),
            ),
            # wrong geometry destination column
            (
                "dest",
                "not_in_df",
                pytest.raises(
                    IndexError,
                    match=(r"'not_in_df' is not a column in the dataframe"),
                ),
            ),
            # wrong unit
            (
                "unit",
                "not a unit",
                pytest.raises(
                    TypeError, match=(r"`unit` expected .*Unit.* Got .*str.*")
                ),
            ),
        ],
    )
    def test__haversine_df_inputs(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_od_matrix: pd.DataFrame,
        arg_name: str,
        arg_value: Any,
        expected: Type[RaisesContext],
    ):
        """Test _gdf_batch_origins inputs.

        Parameters
        ----------
        dummy_transport_network : AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_od_matrix : pd.DataFrame
            Fixture with dummy O-D matrix.
        arg_name : str
            Name of function argument to test.
        arg_value : Any
            Value to use for argument being tested.
        expected : Type[RaisesContext]
            Expected raise result.

        """
        # transport network
        analyse_network = dummy_transport_network

        default_args = {
            "df": dummy_od_matrix,
            "orig": "geometry_orig",
            "dest": "geometry_dest",
            "unit": Unit.KILOMETERS,
        }

        if arg_name is not None:
            default_args[arg_name] = arg_value

        with expected:
            assert [analyse_network._haversine_df(**default_args)]

    def test__haversine_df_outputs(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_od_matrix: pd.DataFrame,
    ):
        """Test _gdf_batch_origins outputs.

        Parameters
        ----------
        dummy_transport_network : AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_od_matrix : pd.DataFrame
            Fixture with dummy O-D matrix.

        """
        # transport network
        analyse_network = dummy_transport_network

        default_args = {
            "df": dummy_od_matrix,
            "orig": "geometry_orig",
            "dest": "geometry_dest",
            "unit": Unit.KILOMETERS,
        }

        output = np.array([0.21773847, 0.47178888, 0.25921125, 0.3369124])

        assert np.array_equal(
            np.round(analyse_network._haversine_df(**default_args), 8), output
        )


# _estimate_num_partitions
class Test_estimate_num_partitions:
    """Class to test the _estimate_num_partitions internal function."""

    @pytest.mark.parametrize(
        "arg_name, arg_value, expected",
        [
            # wrong df
            (
                "df",
                "not a dataframe",
                pytest.raises(
                    TypeError,
                    match=(r"`df` expected .*DataFrame.* Got .*str.*"),
                ),
            ),
            # wrong partition size
            (
                "partition_size",
                "not an integer",
                pytest.raises(
                    TypeError,
                    match=(r"`partition_size` expected .*int.* Got .*str.*"),
                ),
            ),
        ],
    )
    def test__estimate_num_partitions_inputs(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_big_df: pd.DataFrame,
        arg_name: str,
        arg_value: Any,
        expected: Type[RaisesContext],
    ):
        """Test _estimate_num_partitions inputs.

        Parameters
        ----------
        dummy_transport_network : AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_big_df : pd.DataFrame
            Fixture with dummy big dataframe.
        arg_name : str
            Name of function argument to test.
        arg_value : Any
            Value to use for argument being tested.
        expected : Type[RaisesContext]
            Expected raise result.

        """
        # transport network
        analyse_network = dummy_transport_network

        default_args = {
            "df": dummy_big_df,
            "partition_size": 200,
        }

        default_args[arg_name] = arg_value

        with expected:
            assert analyse_network._estimate_num_partitions(**default_args)

    @pytest.mark.parametrize(
        "arg_name, arg_value, expected",
        [
            # wrong df
            ("partition_size", 200, 1),
            # wrong partition size
            ("partition_size", 10, 8),
        ],
    )
    def test__estimate_num_partitions_outputs(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_big_df: pd.DataFrame,
        arg_name: str,
        arg_value: Any,
        expected: Type[RaisesContext],
    ):
        """Test _estimate_num_partitions outputs.

        Parameters
        ----------
        dummy_transport_network : AnalyseNetwork
            Fixture with initialised AnalyseNetwork object.
        dummy_big_df : pd.DataFrame
            Fixture with dummy big dataframe.
        arg_name : str
            Name of function argument to test.
        arg_value : Any
            Value to use for argument being tested.
        expected : int
            Expected result.

        """
        # transport network
        analyse_network = dummy_transport_network

        default_args = {
            "df": dummy_big_df,
            "partition_size": 200,
        }

        default_args[arg_name] = arg_value

        assert (
            analyse_network._estimate_num_partitions(**default_args)
            == expected
        )


# _save_to_parquet
class Test_save_to_parquet:
    """Class to test the _save_to_parquet internal function."""

    @pytest.mark.parametrize(
        "arg_name, arg_value, expected",
        [
            # wrong od matrix
            (
                "od_matrix",
                "not a dataframe",
                pytest.raises(
                    TypeError,
                    match=(r"`od_matrix` expected .*DataFrame.* Got .*str.*"),
                ),
            ),
            # wrong filename func
            (
                "out_name_func",
                12345,
                pytest.raises(
                    TypeError,
                    match=(r"`out_name_func` expected .*str.* Got .*int.*"),
                ),
            ),
            # wrong npartition
            (
                "npartitions",
                "not an int",
                pytest.raises(
                    TypeError,
                    match=(r"`npartitions` expected .*int.* Got .*str.*"),
                ),
            ),
            # wrong out_path
            (
                "out_path",
                ["wrong path"],
                pytest.raises(
                    TypeError,
                    match=(r"`out_path` expected path-like, found.*list.*"),
                ),
            ),
        ],
    )
    def test__save_to_parquet_input(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_big_df: pd.DataFrame,
        dummy_filepath: pathlib.Path,
        arg_name: str,
        arg_value: Any,
        expected: Type[RaisesContext],
    ):
        """Test _save_to_parquet inputs.

        Parameters
        ----------
        dummy_transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        dummy_big_df : pd.DataFrame
            Fixture with dummy big dataframe.
        dummy_filepath : pathlib.Path
            Fixture with path to the temporary directory to save files.
        arg_name : str
            Name of function argument to test.
        arg_value : Any
            Value to use for argument being tested.
        expected : int
            Expected result.

        """
        # transport network
        analyse_network = dummy_transport_network

        default_args = {
            "od_matrix": dummy_big_df,
            "out_name_func": "file_name",
            "out_path": dummy_filepath,
            "npartitions": 1,
        }

        default_args[arg_name] = arg_value

        with expected:
            assert analyse_network._save_to_parquet(**default_args)

    @pytest.mark.parametrize(
        "arg_name, arg_value",
        [
            # no error raised, 1 partition
            ("npartitions", 2),
            # no error raised, 11 partitions
            ("npartitions", 11),
        ],
    )
    def test__save_to_parquet_output(
        self,
        dummy_transport_network: an.AnalyseNetwork,
        dummy_big_df: pd.DataFrame,
        dummy_filepath: pathlib.Path,
        arg_name: str,
        arg_value: Any,
    ):
        """Test _save_to_parquet outputs.

        Parameters
        ----------
        dummy_transport_network : AnalyseNetwork
            Initialised AnalyseNetwork object created from PBF, GTFS and
            centroids fixtures.
        dummy_big_df : pd.DataFrame
            Fixture with dummy big dataframe.
        dummy_filepath : pathlib.Path
            Fixture with path to the temporary directory to save files.
        arg_name : str
            Name of function argument to test.
        arg_value : Any
            Value to use for argument being tested.

        """
        # transport network
        analyse_network = dummy_transport_network

        default_args = {
            "od_matrix": dummy_big_df,
            "out_name_func": "file_name",
            "out_path": dummy_filepath,
            "npartitions": 1,
        }

        default_args[arg_name] = arg_value
        analyse_network._save_to_parquet(**default_args)

        assert (
            len(
                glob.glob(
                    os.path.join(
                        default_args["out_path"],
                        f"*{default_args['out_name_func']}*",
                    )
                )
            )
            == default_args["npartitions"]
        )


#######################
# test module outputs #
#######################
@pytest.mark.parametrize(
    "batch_orig, num_files",
    [
        # all origins
        (False, 1),
        # single origin
        (True, 4),
    ],
)
def test_od_matrix(
    dummy_transport_network: an.AnalyseNetwork,
    dummy_filepath: pathlib.Path,
    batch_orig: bool,
    num_files: int,
):
    """Test main od_matrix method outputs.

    Parameters
    ----------
    dummy_transport_network : AnalyseNetwork
        Initialised AnalyseNetwork object created from PBF, GTFS and
        centroids fixtures.
    dummy_filepath : pathlib.Path
        Path to temporary directory to save parquet files.
    batch_orig : bool
        Flag to indicate whether to calculate the transport network
        performance using the whole dataset or batching by origin
    num_files : int
        Number of files expected based on `batch_orig`.

    """
    tn = dummy_transport_network
    tn.od_matrix(
        out_path=dummy_filepath,
        batch_orig=batch_orig,
        partition_size=200,
        destination_col="within_urban_centre",
        distance=11.25,
        unit=Unit.KILOMETERS,
        departure=datetime.datetime(2023, 6, 13, 8, 0),
        departure_time_window=datetime.timedelta(hours=1),
        max_time=datetime.timedelta(minutes=45),
        transport_modes=[TransportMode.TRANSIT],
    )

    if batch_orig:
        # check 4 parquet files are saved
        assert (
            len(glob.glob(os.path.join(dummy_filepath, "*.parquet")))
            == num_files
        )
        # check 4 files with expected name pattern are saved
        assert sorted(
            glob.glob(os.path.join(dummy_filepath, "batch-[1-9]-0.parquet"))
        ) == [
            os.path.join(dummy_filepath, f"batch-{n}-0.parquet")
            for n in range(1, 5)
        ]
    else:
        # check 1 parquet file is saved
        assert (
            len(glob.glob(os.path.join(dummy_filepath, "*.parquet")))
            == num_files
        )
        # check name of file is as expected
        assert glob.glob(os.path.join(dummy_filepath, "*.parquet")) == [
            os.path.join(dummy_filepath, "batch-all-0.parquet")
        ]

    loaded_output = pd.read_parquet(dummy_filepath)
    assert list(loaded_output["from_id"]) == [1, 1, 2, 2, 3, 3, 4, 4]
    assert list(loaded_output["to_id"]) == [2, 4, 2, 4, 2, 4, 2, 4]
    assert list(loaded_output["travel_time"]) == [9, 4, 0, 7, 12, 7, 7, 0]
