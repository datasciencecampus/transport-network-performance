import pytest

from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from transport_performance._metrics.tp_utils import (
    _transport_performance_pandas,
    _transport_performance_stats,
)

# import metrics fixtures via pytest_plugins
pytest_plugins = ["tests._metrics.metrics_fixtures"]


class TestTransportPerformancePandas:
    """Unit tests for _transport_performance_pandas()."""

    @pytest.mark.parametrize(
        "tt_path",
        [lazy_fixture("tt_fixture"), lazy_fixture("multi_tt_fixture")],
    )
    def test__transport_performance_pandas(
        self,
        centroid_gdf_fixture,
        pop_gdf_fixture,
        tt_path,
        expected_transport_performance,
    ) -> None:
        """Test main behaviour of _transport_performance_pandas().

        Test with both single and multiple travel time input parquet files.

        Parameters
        ----------
        centroid_gdf_fixture
            A mock centroid test fixture.
        pop_gdf_fixture
            A mock population test fixture.
        tt_path
            A path to mock travel time fixture(s).
        expected_transport_performance
            A mock travel time test fixture.

        """
        # call transport_performance() using the test fixtures
        tp_df = _transport_performance_pandas(
            tt_path,
            centroid_gdf_fixture,
            pop_gdf_fixture,
            travel_time_threshold=3,
            distance_threshold=0.11,
        )

        # upack expected results and confirm equivalence
        test_subset_cols, expected_tp, _ = expected_transport_performance
        assert_frame_equal(tp_df[test_subset_cols], expected_tp)

    def test__transport_performance_pandas_source_dest_cols(
        self,
        centroid_gdf_fixture,
        pop_gdf_fixture,
        change_tt_cols_fixture,
        expected_transport_performance,
    ) -> None:
        """Test non default `sources_col` and `destinations_col`.

        Parameters
        ----------
        centroid_gdf_fixture
            A mock centroid test fixture.
        pop_gdf_fixture
            A mock population test fixture.
        change_tt_cols_fixture
            A mock travel time fixture with alternative column names. See
            `change_tt_cols_fixture` for more details.
        expected_transport_performance
            Expected transport performance results.

        """
        # call transport_performance() using the test fixtures
        tp_df = _transport_performance_pandas(
            change_tt_cols_fixture,
            centroid_gdf_fixture,
            pop_gdf_fixture,
            sources_col="from",
            destinations_col="to",
            travel_time_threshold=3,
            distance_threshold=0.11,
        )

        # upack expected results and confirm equivalence
        test_subset_cols, expected_tp, _ = expected_transport_performance
        assert_frame_equal(tp_df[test_subset_cols], expected_tp)


class TestTransportPerformanceStats:
    """Unit tests for `_transport_performance_stats()`."""

    def test_uc_incorrect_crs(
        self,
        expected_transport_performance,
        uc_fixture,
    ) -> None:
        """Check descriptive stats calculation with wrong CRS.

        Ensure the user warning is raisde, CRS conversion, and calculated area
        are correct when an urban centre with an invalid CRS is provided.

        Parameters
        ----------
        expected_transport_performance
            Mock transport performance input (using expected output).
        uc_fixture
            Mock urban centre fixture, who's CRS will be converted.

        """
        with pytest.warns(
            UserWarning,
            match="Unable to calculate the ubran centre area in CRS EPSG:4326",
        ):
            _, tp_df, expected_stats = expected_transport_performance
            stats_df = _transport_performance_stats(
                tp_df,
                urban_centre_name="name",
                urban_centre_country="country",
                urban_centre_gdf=uc_fixture.to_crs("EPSG:4326"),
            )
            assert_frame_equal(stats_df, expected_stats)
