"""Tests for transport_performance/metrics.py."""

import pytest

from pandas.testing import assert_frame_equal

from transport_performance.metrics import transport_performance

# import metrics fixtures via pytest_plugins
pytest_plugins = ["tests._metrics.metrics_fixtures"]


class TestTransportPerformance:
    """Collection of tests for `transport_performance()` function."""

    def test_transport_performance(
        self,
        uc_fixture,
        centroid_gdf_fixture,
        pop_gdf_fixture,
        tt_fixture,
        expected_transport_performance,
    ) -> None:
        """Test main behaviour of transport performance function.

        Parameters
        ----------
        uc_fixture
            A mock urban centre test fixture
        centroid_gdf_fixture
            A mock centroid test fixture
        pop_gdf_fixture
            A mock population test fixture
        tt_fixture
            A mock travel time test fixture
        expected_transport_performance
            Expected results fixture

        Notes
        -----
        1. See `scripts/prep-metrics-fixtures.py` for more details on the
        input fixtures.
        2. Expected results were manually calculated and QA-ed for this unit
        test.

        """
        # call transport_performance() using the test fixtures
        tp_df, stats_df = transport_performance(
            tt_fixture,
            centroid_gdf_fixture,
            pop_gdf_fixture,
            travel_time_threshold=3,
            distance_threshold=0.11,
            urban_centre_name="name",
            urban_centre_country="country",
            urban_centre_gdf=uc_fixture,
        )
        # upack expected results and confirm equivalence
        test_cols, expected_tp, expected_stats = expected_transport_performance

        # assert results are as expected
        assert_frame_equal(tp_df[test_cols], expected_tp)
        assert_frame_equal(stats_df, expected_stats)

    @pytest.mark.parametrize(
        "arg_name, arg_value, expected",
        [
            (
                "travel_times_path",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*travel_times_path)(?=.*str)(?=.*pathlib.Path)"
                        "(?=.*Got)(?=.*float).+"
                    ),
                ),
            ),
            (
                "centroid_gdf",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*centroid_gdf)(?=.*GeoDataFrame)(?=.*Got)"
                        "(?=.*float).+"
                    ),
                ),
            ),
            (
                "pop_gdf",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*pop_gdf)(?=.*GeoDataFrame)(?=.*Got)"
                        "(?=.*float).+"
                    ),
                ),
            ),
            (
                "travel_time_threshold",
                "45",
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*travel_time_threshold)(?=.*int)(?=.*Got)"
                        "(?=.*str).+"
                    ),
                ),
            ),
            (
                "distance_threshold",
                "11.25",
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*distance_threshold)(?=.*int)(?=.*float)(?=.*Got)"
                        "(?=.*str).+"
                    ),
                ),
            ),
            (
                "sources_col",
                0.0,
                pytest.raises(
                    TypeError,
                    match=("(?=.*sources_col)(?=.*str)(?=.*Got)(?=.*float).+"),
                ),
            ),
            (
                "destinations_col",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*destinations_col)(?=.*str)(?=.*Got)(?=.*float).+"
                    ),
                ),
            ),
            (
                "backend",
                0.0,
                pytest.raises(
                    TypeError,
                    match=("(?=.*backend)(?=.*str)(?=.*Got)(?=.*float).+"),
                ),
            ),
            (
                "descriptive_stats",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*descriptive_stats)(?=.*bool)(?=.*Got)"
                        "(?=.*float).+"
                    ),
                ),
            ),
            (
                "urban_centre_name",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*urban_centre_name)(?=.*str)(?=.*Got)"
                        "(?=.*float).+"
                    ),
                ),
            ),
            (
                "urban_centre_country",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*urban_centre_country)(?=.*str)(?=.*Got)"
                        "(?=.*float).+"
                    ),
                ),
            ),
            (
                "urban_centre_gdf",
                0.0,
                pytest.raises(
                    TypeError,
                    match=(
                        "(?=.*urban_centre_gdf)(?=.*GeoDataFrame)(?=.*Got)"
                        "(?=.*float).+"
                    ),
                ),
            ),
        ],
    )
    def test_transport_performance_type_defences(
        self,
        arg_name,
        arg_value,
        expected,
        uc_fixture,
        centroid_gdf_fixture,
        pop_gdf_fixture,
        tt_fixture,
    ):
        """Check `transport_performance()` type defences.

        Parameters
        ----------
        arg_name
            Name of function argument to test.
        arg_value
            Value to use for argument being tested.
        expected
            Expected raise statement for argument being tested.
        uc_fixture
            Mock urban centre fixture.
        centroid_gdf_fixture
            Mock centroids fixture.
        pop_gdf_fixture
            Mock population fixture.
        tt_fixture
            Mock travel time fixture.

        Notes
        -----
        1. `match` arugument in parameterisation checks each word is within
        the return error message independent of order. This is such that if the
        error message change, the key components are still captured.

        """
        # create a argument dictionary to store default values
        default_args = {
            "travel_times_path": tt_fixture,
            "centroid_gdf": centroid_gdf_fixture,
            "pop_gdf": pop_gdf_fixture,
            "travel_time_threshold": 45,
            "distance_threshold": 11.25,
            "sources_col": "from_id",
            "destinations_col": "to_id",
            "backend": "pandas",
            "descriptive_stats": True,
            "urban_centre_name": "name",
            "urban_centre_country": "country",
            "urban_centre_gdf": uc_fixture,
        }

        # change value of argument being tested and check raises
        default_args[arg_name] = arg_value
        with expected:
            transport_performance(**default_args)

    def test_transport_performance_invalid_backend(
        self, centroid_gdf_fixture, pop_gdf_fixture, tt_fixture
    ) -> None:
        """Test transport performance with an invalid backend.

        Parameters
        ----------
        centroid_gdf_fixture
            Mock centroid test fixture
        pop_gdf_fixture
            Mock population test fixture
        tt_fixture
            Mock travel time test fixture

        """
        # call transport_performance() using an invalid backend
        with pytest.raises(
            ValueError,
            match="Got `backend`=test. Expected one of:",
        ):
            transport_performance(
                tt_fixture,
                centroid_gdf_fixture,
                pop_gdf_fixture,
                backend="test",
            )
