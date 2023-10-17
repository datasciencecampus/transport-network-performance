"""Test GTFS utility functions."""

import os
import pytest
import re
import pathlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from plotly.graph_objects import Figure as PlotlyFigure
import numpy as np

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.gtfs_utils import (
    bbox_filter_gtfs,
    _add_validation_row,
    filter_gtfs_around_trip,
    convert_pandas_to_plotly,
    _get_validation_warnings,
)

# location of GTFS test fixture
GTFS_FIX_PTH = os.path.join(
    "tests", "data", "gtfs", "newport-20230613_gtfs.zip"
)


class TestBboxFilterGtfs(object):
    """Test bbox_filter_gtfs."""

    @pytest.fixture(scope="function")
    def bbox_list(self):
        """Tiny bounding box over newport train station."""
        return [-3.0017783334, 51.5874718209, -2.9964692194, 51.5907034241]

    def test_bbox_filter_gtfs_defence(self):
        """Check defensive behaviour for bbox_filter_gtfs."""
        with pytest.raises(
            TypeError,
            match="`units` expected <class 'str'>. Got <class 'bool'>",
        ):
            bbox_filter_gtfs(units=False)

    def test_bbox_filter_gtfs_writes_with_bbox_list(self, bbox_list, tmpdir):
        """Test bbox_filter_gtfs writes when a bbox list is passed."""
        tmp_out = os.path.join(
            tmpdir, "newport-train-station-bboxlist_gtfs.zip"
        )
        bbox_filter_gtfs(
            in_pth=os.path.join(
                "tests", "data", "gtfs", "newport-20230613_gtfs.zip"
            ),
            out_pth=pathlib.Path(tmp_out),
            bbox=bbox_list,
        )
        assert os.path.exists(
            tmp_out
        ), f"Expected {tmp_out} to exist but it did not."
        # check the output gtfs can be read
        feed = GtfsInstance(gtfs_pth=pathlib.Path(tmp_out))
        assert isinstance(
            feed, GtfsInstance
        ), f"Expected class `Gtfs_Instance but found: {type(feed)}`"

    def test_bbox_filter_gtfs_writes_with_bbox_gdf(self, bbox_list, tmpdir):
        """Test bbox_filter_gtfs writes when a bbox GDF is passed."""
        # convert bbox list to gdf
        bbox_gdf = gpd.GeoDataFrame(
            index=[0], crs="epsg:4326", geometry=[box(*bbox_list)]
        )
        tmp_out = os.path.join(
            tmpdir, "newport-train-station-bboxgdf_gtfs.zip"
        )

        bbox_filter_gtfs(
            in_pth=os.path.join(
                "tests", "data", "gtfs", "newport-20230613_gtfs.zip"
            ),
            out_pth=pathlib.Path(tmp_out),
            bbox=bbox_gdf,
        )

        assert os.path.exists(
            tmp_out
        ), f"Expected {tmp_out} to exist but it did not."
        # check the output gtfs can be read
        feed = GtfsInstance(gtfs_pth=pathlib.Path(tmp_out))
        assert isinstance(
            feed, GtfsInstance
        ), f"Expected class `Gtfs_Instance but found: {type(feed)}`"


class Test_AddValidationRow(object):
    """Tests for _add_validation_row()."""

    def test__add_validation_row_defence(self):
        """Defensive tests for _add_test_validation_row()."""
        gtfs = GtfsInstance(gtfs_pth=GTFS_FIX_PTH)
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "The validity_df does not exist as an "
                "attribute of your GtfsInstance object, \n"
                "Did you forget to run the .is_valid() method?"
            ),
        ):
            _add_validation_row(
                gtfs, _type="warning", message="test", table="stops"
            )

    def test__add_validation_row_on_pass(self):
        """General tests for _add_test_validation_row()."""
        gtfs = GtfsInstance(gtfs_pth=GTFS_FIX_PTH)
        gtfs.is_valid(far_stops=False)

        _add_validation_row(
            gtfs=gtfs, _type="warning", message="test", table="stops"
        )

        expected_row = ["warning", "test", "stops", []]
        found_row = list(gtfs.validity_df.iloc[-1].values)

        assert expected_row == found_row, (
            "_add_validation_row() failed to add the correct data to the "
            "validity df (GtfsInstance().validity_df)."
        )


class Test_FilterGtfsAroundTrip(object):
    """Tests for filter_gtfs_around_trip()."""

    def test_filter_gtfs_around_trip_defence(self):
        """Defensive tests for filter_gtfs_around_trip()."""
        # check trips with no shape id are filtered
        gtfs = GtfsInstance(gtfs_pth=GTFS_FIX_PTH)
        with pytest.raises(
            ValueError,
            match="'shape_id' not available for trip with trip_id: "
            "VJe1fb5120f04b2e3699a133007032117aed104794",
        ):
            filter_gtfs_around_trip(
                gtfs, trip_id="VJe1fb5120f04b2e3699a133007032117aed104794"
            )

    def test_filter_gtfs_around_trip_on_pass(self, tmpdir):
        """General tests for filter_gtfs_around_trip()."""
        gtfs = GtfsInstance(gtfs_pth=GTFS_FIX_PTH)
        out_pth = os.path.join(tmpdir, "test_gtfs.zip")

        # check gtfs can be created
        filter_gtfs_around_trip(
            gtfs,
            trip_id="VJbedb4cfd0673348e017d42435abbdff3ddacbf82",
            out_pth=out_pth,
        )
        assert os.path.exists(out_pth), "Failed to filtere GTFS around trip."
        # check the new gtfs can be read
        feed = GtfsInstance(gtfs_pth=out_pth)
        assert isinstance(
            feed, GtfsInstance
        ), f"Expected class `Gtfs_Instance but found: {type(feed)}`"


@pytest.fixture(scope="function")
def test_df():
    """A test fixture."""
    test_df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 1],
            "score": [45, 34, 23, 12, 23],
            "grade": ["A", "B", "C", "D", "C"],
        }
    )
    return test_df


class TestConvertPandasToPlotly(object):
    """Test convert_pandas_to_plotly()."""

    def test_convert_pandas_to_plotly_defences(self, test_df):
        """Test convert_pandas_to_plotly defences."""
        multi_index_df = test_df.groupby(["ID", "grade"]).agg(
            {"score": ["mean", "min", "max"]}
        )
        with pytest.raises(
            TypeError,
            match="Pandas dataframe must have a singular index, not "
            "MultiIndex. "
            "This means that 'df.columns' or 'df.index' does not return a "
            "MultiIndex.",
        ):
            convert_pandas_to_plotly(multi_index_df)

    def test_convert_pandas_to_plotly_on_pass(self, test_df):
        """Test convert_pandas_to_plotly() when defences pass."""
        # return_html
        html_return = convert_pandas_to_plotly(test_df, return_html=True)
        assert isinstance(html_return, str), re.escape(
            f"Expected type str but {type(html_return)} found"
        )

        # return plotly figure
        fig_return = convert_pandas_to_plotly(test_df, return_html=False)
        assert isinstance(fig_return, PlotlyFigure), re.escape(
            "Expected type plotly.graph_objects.Figure but "
            f"{type(fig_return)} found"
        )


class TestGetValidationWarnings(object):
    """Tests for _get_validation_warnings."""

    def test__get_validation_warnings_defence(self):
        """Test thhe defences of _get_validation_warnings."""
        with pytest.raises(
            TypeError, match=".* expected a GtfsInstance object"
        ):
            _get_validation_warnings(True, "test_msg")
        gtfs = GtfsInstance(gtfs_pth=GTFS_FIX_PTH)
        with pytest.raises(
            AttributeError, match="The gtfs has not been validated.*"
        ):
            _get_validation_warnings(gtfs, "test")
        gtfs.is_valid()
        with pytest.raises(
            ValueError, match=r"'return_type' expected one of \[.*\]\. Got .*"
        ):
            _get_validation_warnings(gtfs, "tester", "tester")

    def test__get_validation_warnings(self):
        """Test _get_validation_warnings on pass."""
        gtfs = GtfsInstance(GTFS_FIX_PTH)
        gtfs.is_valid()
        # test return types
        df_exp = _get_validation_warnings(
            gtfs, "test", return_type="dataframe"
        )
        assert isinstance(
            df_exp, pd.DataFrame
        ), f"Expected df, got {type(df_exp)}"
        ndarray_exp = _get_validation_warnings(gtfs, "test")
        assert isinstance(
            ndarray_exp, np.ndarray
        ), f"Expected np.ndarray, got {type(ndarray_exp)}"
        # test with valld regex (assertions on DF data without DF)
        regex_matches = _get_validation_warnings(
            gtfs, "Unrecognized column *.", return_type="dataframe"
        )
        assert len(regex_matches) == 5, (
            "Getting validaiton warnings returned"
            "unexpected number of warnings"
        )
        assert list(regex_matches["type"].unique()) == [
            "warning"
        ], "Dataframe type column not asd expected"
        assert list(regex_matches.table) == [
            "agency",
            "stop_times",
            "stops",
            "trips",
            "trips",
        ], "Dataframe table column not as expected"
        # test with matching message (no regex)
        exact_match = _get_validation_warnings(
            gtfs, "Unrecognized column agency_noc", return_type="Dataframe"
        )
        assert list(exact_match.values[0]) == [
            "warning",
            "Unrecognized column agency_noc",
            "agency",
            [],
        ], "Dataframe values not as expected"
        assert (
            len(exact_match) == 1
        ), f"Expected one match, found {len(exact_match)}"
        # test with no matches (regex)
        regex_no_match = _get_validation_warnings(
            gtfs, ".*This is a test.*", return_type="Dataframe"
        )
        assert len(regex_no_match) == 0, "No matches expected. Matches found"
        # test with no match (no regex)
        no_match = _get_validation_warnings(
            gtfs, "This is a test!!!", return_type="Dataframe"
        )
        assert len(no_match) == 0, "No matches expected. Matched found"
