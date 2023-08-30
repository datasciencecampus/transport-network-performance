"""Test GTFS utility functions."""

from pyprojroot import here
import os
import pytest
import pathlib
import re

import pandas as pd
from plotly.graph_objects import Figure as PlotlyFigure

from transport_performance.gtfs.gtfs_utils import (
    bbox_filter_gtfs,
    convert_pandas_to_plotly,
)
from transport_performance.gtfs.validation import GtfsInstance


class TestBboxFilterGtfs(object):
    """Test bbox_filter_gtfs."""

    def test_bbox_filter_gtfs_defence(self):
        """Check defensive behaviour for bbox_filter_gtfs."""
        with pytest.raises(
            TypeError, match="Expected string. Found <class 'bool'> : False"
        ):
            bbox_filter_gtfs(units=False)

    def test_bbox_filter_gtfs_writes_as_expected(self, tmpdir):
        """Test bbox_filter_gtfs writes out a filtered GTFS archive."""
        tmp_out = os.path.join(tmpdir, "newport-train-station_gtfs.zip")
        bbox_filter_gtfs(
            in_pth=here("tests/data/newport-20230613_gtfs.zip"),
            out_pth=pathlib.Path(tmp_out),
            bbox_list=[
                -3.0017783334,
                51.5874718209,
                -2.9964692194,
                51.5907034241,
            ],  # tiny bounding box over newport train station
        )
        assert os.path.exists(
            tmp_out
        ), f"Expected {tmp_out} to exist but it did not."
        # check the output gtfs can be read
        feed = GtfsInstance(gtfs_pth=pathlib.Path(tmp_out))
        assert isinstance(
            feed, GtfsInstance
        ), f"Expected class `Gtfs_Instance but found: {type(feed)}`"


class TestConvertPandasToPlotly(object):
    """Test convert_pandas_to_plotly()."""

    def test_convert_pandas_to_plotly_defences(self):
        """Test convert_pandas_to_plotly defences."""
        test_df = pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 1],
                "score": [45, 34, 23, 12, 23],
                "grade": ["A", "B", "C", "D", "C"],
            }
        )
        with pytest.raises(
            LookupError,
            match=re.escape(
                "dark is not a valid colour scheme."
                "Valid colour schemes include ['dsc']"
            ),
        ):
            convert_pandas_to_plotly(test_df, scheme="dark")

        multi_index_df = test_df.groupby(["ID", "grade"]).agg(
            {"score": ["mean", "min", "max"]}
        )
        with pytest.raises(
            TypeError,
            match="Pandas dataframe must have a single index,"
            "not MultiIndex",
        ):
            convert_pandas_to_plotly(multi_index_df)

    def test_convert_pandas_to_plotly_on_pass(self):
        """Test convert_pandas_to_plotly() when defences pass."""
        test_df = pd.DataFrame(
            {
                "ID": [1, 2, 3, 4, 1],
                "score": [45, 34, 23, 12, 23],
                "grade": ["A", "B", "C", "D", "C"],
            }
        )
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
