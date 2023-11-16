import pytest

from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from transport_performance._metrics.tp_utils import (
    _transport_performance_pandas,
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
