"""Test GTFS utility functions."""

import os
import pytest
import re
import pathlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from plotly.graph_objects import Figure as PlotlyFigure

from transport_performance.gtfs.validation import GtfsInstance
from transport_performance.gtfs.gtfs_utils import (
    bbox_filter_gtfs,
    _add_validation_row,
    filter_gtfs_around_trip,
    convert_pandas_to_plotly,
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
        gtfs = GtfsInstance()
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
        gtfs = GtfsInstance()
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
        gtfs = GtfsInstance()
        with pytest.raises(
            ValueError,
            match="'shape_id' not available for trip with trip_id: "
            "VJd44c7f90d8e70b3b7332d7d0646690b7c118a7c0",
        ):
            filter_gtfs_around_trip(
                gtfs, trip_id="VJd44c7f90d8e70b3b7332d7d0646690b7c118a7c0"
            )

    def test_filter_gtfs_around_trip_on_pass(self, tmpdir):
        """General tests for filter_gtfs_around_trip()."""
        gtfs = GtfsInstance()
        out_pth = os.path.join(tmpdir, "test_gtfs.zip")

        # check gtfs can be created
        filter_gtfs_around_trip(
            gtfs,
            trip_id="VJ217b8849f1e5675d19ca46660a32d0719db12c80",
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
