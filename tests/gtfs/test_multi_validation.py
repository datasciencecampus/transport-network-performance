"""Tests for multi_validation.py."""
import pytest
import os
import glob
import pathlib
import shutil

import numpy as np
import pandas as pd
import folium

from transport_performance.gtfs.multi_validation import (
    MultiGtfsInstance,
)
from transport_performance.gtfs.validation import GtfsInstance


@pytest.fixture(scope="function")
def multi_gtfs_paths():
    """Small test fixture for GTFS paths."""
    paths = [
        "tests/data/chester-20230816-small_gtfs.zip",
        "tests/data/gtfs/newport-20230613_gtfs.zip",
    ]
    return paths


@pytest.fixture(scope="function")
def multi_gtfs_fixture(multi_gtfs_paths):
    """Test fixture for MultiGtfsInstance."""
    m_gtfs = MultiGtfsInstance(multi_gtfs_paths)
    return m_gtfs


class TestMultiGtfsInstance(object):
    """Tests for the MultiGtfsInstance class."""

    def test_init_defences(self, tmp_path):
        """Defensive tests for the class constructor."""
        # path not expected type
        with pytest.raises(
            TypeError, match=".*path.*expected.*str.*list.*Got.*int.*"
        ):
            MultiGtfsInstance(12)
        # not enough files found (0)
        with pytest.raises(FileNotFoundError, match="No GTFS files found."):
            MultiGtfsInstance(f"{tmp_path}/*.zip")
        # not enough files found (1)
        with open(os.path.join(tmp_path, "test.txt"), "w") as f:
            f.write("This is a test.")
        # files of wrong type
        with open(os.path.join(tmp_path, "test2.txt"), "w") as f:
            f.write("This is a test.")
        with pytest.raises(
            ValueError, match=r".*path\[0\].*expected.*zip.*Found .txt"
        ):
            MultiGtfsInstance(f"{tmp_path}/*.txt")

    def test_init(self, multi_gtfs_paths):
        """General tests for the constructor."""
        m_gtfs = MultiGtfsInstance(multi_gtfs_paths)
        assert np.array_equal(
            np.sort(m_gtfs.paths), np.sort(multi_gtfs_paths)
        ), "Paths not as expected"
        assert len(m_gtfs.paths) == 2, "Unexpected number of GTFS paths"
        assert (
            len(m_gtfs.instances) == 2
        ), "Unexpected number of GTFS instances"
        for inst in m_gtfs.instances:
            assert isinstance(
                inst, GtfsInstance
            ), "GtfsInstance not instanciated"
        # test singular gtfs path
        test_path = pathlib.Path(
            os.path.join("tests", "data", "chester-20230816-small_gtfs.zip")
        )
        m_gtfs_2 = MultiGtfsInstance(test_path)
        assert len(m_gtfs_2.paths) == 1, "Too many/Not enough paths passed"
        assert (
            pathlib.Path(m_gtfs.paths[0]) == test_path
        ), "Test path not as expected in MGTFS"

    def test_save_feeds(self, multi_gtfs_paths, tmp_path):
        """Tests for .save_feeds()."""
        gtfs = MultiGtfsInstance(multi_gtfs_paths)
        save_dir = os.path.join(tmp_path, "save_test")
        gtfs.save_feeds(save_dir)
        # assert .save created parent dir
        assert os.path.exists(save_dir), "Save directory not created"
        # assert files saved
        expected_paths = [
            "chester-20230816-small_gtfs_new.zip",
            "newport-20230613_gtfs_new.zip",
        ]
        found_paths = [
            os.path.basename(fpath) for fpath in glob.glob(save_dir + "/*.zip")
        ]
        assert np.array_equal(
            np.sort(expected_paths), np.sort(found_paths)
        ), "GtfsInstances not saved as expected"

    def test_clean_feeds_defences(self, multi_gtfs_fixture):
        """Defensive tests for .clean_feeds()."""
        with pytest.raises(TypeError, match=".*clean_kwargs.*dict.*bool"):
            multi_gtfs_fixture.clean_feeds(True)

    def test_clean_feeds_on_pass(self, multi_gtfs_fixture):
        """General tests for .clean_feeds()."""
        # validate and do quick check on validity_df
        valid_df = multi_gtfs_fixture.is_valid()
        assert len(valid_df) == 12, "validity_df not as expected"
        # clean feed
        multi_gtfs_fixture.clean_feeds()
        # ensure cleaning has occured
        new_valid = multi_gtfs_fixture.is_valid()
        assert len(new_valid) == 9
        assert np.array_equal(
            list(new_valid.iloc[3][["type", "table"]].values),
            ["error", "routes"],
        ), "Validity df after cleaning not as expected"

    def test_is_valid_defences(self, multi_gtfs_fixture):
        """Defensive tests for .is_valid()."""
        with pytest.raises(TypeError, match=".*validation_kwargs.*dict.*bool"):
            multi_gtfs_fixture.is_valid(True)

    def test_is_valid_on_pass(self, multi_gtfs_fixture):
        """General tests for is_valid()."""
        valid_df = multi_gtfs_fixture.is_valid()
        assert len(valid_df) == 12, "Validation df not as expected"
        assert np.array_equal(
            list(valid_df.iloc[3][["type", "message"]].values),
            (["warning", "Fast Travel Between Consecutive Stops"]),
        )
        assert hasattr(
            multi_gtfs_fixture, "validity_df"
        ), "validity_df not created"
        assert isinstance(
            multi_gtfs_fixture.validity_df, pd.DataFrame
        ), "validity_df not a df"

    def test_validate_empty_feeds(self, multi_gtfs_fixture):
        """Tests for validate_empty_feeds."""
        # emulate filtering the feeds to a box with no routes by dropping all
        # stop times
        [
            i.feed.stop_times.drop(i.feed.stop_times.index, inplace=True)
            for i in multi_gtfs_fixture.instances
        ]
        assert (
            len(multi_gtfs_fixture.validate_empty_feeds()) == 2
        ), "Two empty feeds were not found"
        # ensure they weren't deleted
        assert len(multi_gtfs_fixture.instances) == 2, "Feeds deleted"
        with pytest.warns(
            UserWarning, match="MultiGtfsInstance has no feeds."
        ):
            multi_gtfs_fixture.validate_empty_feeds(delete=True)
        assert len(multi_gtfs_fixture.instances) == 0, "Feeds were not deleted"

    def test_validate_empty_feeds_outputs_correct_filenames(
        self, multi_gtfs_paths, tmp_path
    ):
        """Regression test, labelled filenames are correct. Need 3 GTFS."""
        multi_gtfs_paths = sorted(multi_gtfs_paths)
        dest_paths = [
            os.path.join(tmp_path, os.path.basename(pth))
            for pth in multi_gtfs_paths
        ]
        # copy GTFS fixtures into tmp
        for s, d in zip(multi_gtfs_paths, dest_paths):
            shutil.copyfile(src=s, dst=d)
        # add a copy of chester - need 3 files to test filename sequencing
        dupe_chester = os.path.join(
            tmp_path, "DUPE_" + os.path.basename(multi_gtfs_paths[0])
        )
        shutil.copyfile(multi_gtfs_paths[0], dupe_chester)
        # check that we have the correct file setup in tmp
        tmp_contents_pre = sorted(os.listdir(tmp_path))
        assert tmp_contents_pre == [
            "DUPE_chester-20230816-small_gtfs.zip",
            "chester-20230816-small_gtfs.zip",
            "newport-20230613_gtfs.zip",
        ], f"Expected 3 GTFS before empty feed rm, found: {tmp_contents_pre}"
        # instantiate multi gtfs and filter to a newport BBOX with delete empty
        # feeds, expecting a single poulated GTFS with correct newport filenm
        gtfs = MultiGtfsInstance(tmp_path)
        n_expected = len(tmp_contents_pre)
        n_found = len(gtfs.instances)
        assert (
            n_found == n_expected
        ), f"Expected {n_expected} instances but found {n_found}"
        # filter to newport train station
        gtfs.filter_to_bbox(
            [-3.004961, 51.586603, -2.995325, 51.591028],
            delete_empty_feeds=True,
        )
        n_filtered = len(gtfs.instances)
        assert (
            n_filtered == 1
        ), f"Expected 1 instance after delete_empty_feeds, found {n_filtered}"
        # save the multi feed in a new directory in tmp
        out_pth = os.path.join(tmp_path, "NO_CHESTER")
        gtfs.save_feeds(out_pth)
        out_contents = os.listdir(out_pth)
        assert out_contents == [
            "newport-20230613_gtfs_new.zip"
        ], f"Saved feeds expected single Newport GTFS, found: {out_contents}"

    def test_filter_to_date_defences(self, multi_gtfs_fixture):
        """Defensive tests for .filter_to_date()."""
        with pytest.raises(
            TypeError, match=".*dates.*expected.*str.*list.*int.*"
        ):
            multi_gtfs_fixture.filter_to_date(12)
        with pytest.raises(
            ValueError,
            match=(
                r"In GTFS tests/data/chester-20230816-small_gtfs.zip."
                r"\n{\'20040109\'} passed to \'filter_dates\' not present in "
                r"feed dates.*"
            ),
        ):
            multi_gtfs_fixture.filter_to_date(["20040109"])

    def test_filter_to_date(self, multi_gtfs_fixture):
        """Tests for .filter_to_date()."""
        # assert original contents
        assert (
            len(multi_gtfs_fixture.instances[0].feed.stop_times) == 34249
        ), "Gtfs inst[0] not as expected"
        assert (
            len(multi_gtfs_fixture.instances[1].feed.stop_times) == 7765
        ), "Gtfs inst[1] not as expected"
        multi_gtfs_fixture.filter_to_date("20230806")
        # assert filtered contents
        assert (
            len(multi_gtfs_fixture.instances[0].feed.stop_times) == 984
        ), "Gtfs inst[0] not as expected after filter"
        assert (
            len(multi_gtfs_fixture.instances[1].feed.stop_times) == 151
        ), "Gtfs inst[1] not as expected after filter"

    def test_filter_to_bbox_defences(self, multi_gtfs_fixture):
        """Defensive tests for .filter_to_bbox()."""
        with pytest.raises(
            TypeError, match=".*bbox.*expected.*list.*GeoDataFrame.*int.*"
        ):
            multi_gtfs_fixture.filter_to_bbox(12)
        with pytest.raises(
            TypeError, match=".*crs.*expected.*str.*int.*.*DataFrame"
        ):
            multi_gtfs_fixture.filter_to_bbox(
                [12.0, 12.0, 13.0, 13.0], pd.DataFrame()
            )
        with pytest.raises(
            TypeError, match=".*delete_empty_feeds.*expected.*bool.*str.*"
        ):
            multi_gtfs_fixture.filter_to_bbox(
                [1.0, 1.0, 1.0, 1.0], 4326, "test"
            )
        # assert error is raised if bbox empties multi GTFS
        with pytest.raises(
            ValueError, match="BBOX.*has filtered.*to contain no data.*"
        ):
            multi_gtfs_fixture.filter_to_bbox(
                [1.0, 1.0, 1.0, 1.0], "epsg:4326", True
            )

    def test_filter_to_bbox(self, multi_gtfs_fixture):
        """Tests for .filter_to_bbox()."""
        # assert original contents
        assert (
            len(multi_gtfs_fixture.instances[0].feed.stop_times) == 34249
        ), "Gtfs inst[0] not as expected"
        assert (
            len(multi_gtfs_fixture.instances[1].feed.stop_times) == 7765
        ), "Gtfs inst[1] not as expected"
        # filter to bbox
        # (out of scope of Chester, so Chester GTFS should return 0)
        with pytest.warns(UserWarning):
            multi_gtfs_fixture.filter_to_bbox(
                [-2.985535, 51.551459, -2.919617, 51.606077]
            )
        # assert filtered contents
        assert (
            len(multi_gtfs_fixture.instances[0].feed.stop_times) == 0
        ), "Gtfs inst[0] not as expected after filter"
        assert (
            len(multi_gtfs_fixture.instances[1].feed.stop_times) == 217
        ), "Gtfs inst[1] not as expected after filter"

    @pytest.mark.parametrize(
        "which, summ_ops, raises, match",
        (
            ["route", True, TypeError, ".*summ_ops.*list.*bool"],
            [True, [np.max], TypeError, ".*which.*str.*bool"],
            [
                "not_which",
                [np.max],
                ValueError,
                ".*which.*trips.*routes.*not_which.*",
            ],
        ),
    )
    def test__summarise_core_defence(
        self, multi_gtfs_fixture, which, summ_ops, raises, match
    ):
        """Defensive tests for _summarise_core()."""
        with pytest.raises(raises, match=match):
            multi_gtfs_fixture._summarise_core(which=which, summ_ops=summ_ops)

    def test__summarise_core(self, multi_gtfs_fixture):
        """General tests for _summarise_core()."""
        # test summarising routes
        summary = multi_gtfs_fixture._summarise_core(
            which="routes", summ_ops=[np.max, np.mean]
        )
        assert isinstance(
            summary, pd.DataFrame
        ), "_summarise_core() did not return a df."
        assert (
            len(summary) == 14
        ), f"Number of rows in route summary df is {len(summary)}. Expected 14"
        friday_exp = [["friday", 37, 31.0, 3], ["friday", 4, 4.0, 200]]
        assert np.array_equal(
            friday_exp,
            [list(x) for x in list(summary[summary.day == "friday"].values)],
        ), "Route summary for Friday not as expected"
        # test summarising trips
        summary = multi_gtfs_fixture._summarise_core(
            which="trips", summ_ops=[np.max, np.mean]
        )
        assert isinstance(
            summary, pd.DataFrame
        ), "_summarise_core() did not return a df."
        assert (
            len(summary) == 14
        ), f"Number of rows in trip summary df is {len(summary)}. Expected 14"
        friday_exp = [["friday", 3, 955, 792.0], ["friday", 200, 22, 22.0]]
        assert np.array_equal(
            friday_exp,
            [list(x) for x in list(summary[summary.day == "friday"].values)],
        ), "trip summary for Friday not as expected"
        dated = multi_gtfs_fixture._summarise_core(
            which="routes", return_summary=False
        )
        assert isinstance(
            dated, pd.DataFrame
        ), "Returned dated table not DataFrame"
        assert len(dated) == 202955, "Returned dated table not as expected"
        dated_sum = multi_gtfs_fixture._summarise_core(
            which="routes", to_days=False
        )
        assert len(dated_sum) == 590, "Dated route counts not as expected"
        assert (
            dated_sum[dated_sum.date == "2024-04-06"].route_count.iloc[0]
            == 159
        ), "Unexpecteed number of routes on 2024-04-06"

    def test_summarise_trips(self, multi_gtfs_fixture):
        """General tests for summarise_trips()."""
        # assert that the summary is returned
        summary = multi_gtfs_fixture.summarise_trips()
        assert isinstance(summary, pd.DataFrame)
        assert hasattr(multi_gtfs_fixture, "daily_trip_summary")

    def test_summarise_routes(self, multi_gtfs_fixture):
        """General tests for summarise_routes()."""
        # assert that the summary is returned
        summary = multi_gtfs_fixture.summarise_routes()
        assert isinstance(summary, pd.DataFrame)
        assert hasattr(multi_gtfs_fixture, "daily_route_summary")

    @pytest.mark.parametrize(
        "path, return_viz, filtered_only, raises, match",
        (
            [
                True,
                True,
                True,
                TypeError,
                ".*path.*expected.*str.*Path.*None.*Got.*bool.*",
            ],
            [
                "test.html",
                12,
                True,
                TypeError,
                ".*return_viz.*expected.*bool.*None.*Got.*int.*",
            ],
            [
                None,
                None,
                True,
                ValueError,
                "Both .*path.*return_viz.* parameters are of NoneType.",
            ],
            [
                "test.html",
                True,
                12,
                TypeError,
                ".*filtered_only.*expected.*bool.*Got.*int.*",
            ],
        ),
    )
    def test_viz_stops_defences(
        self,
        multi_gtfs_fixture,
        path,
        return_viz,
        filtered_only,
        raises,
        match,
    ):
        """Defensive tests for .viz_stops()."""
        with pytest.raises(raises, match=match):
            multi_gtfs_fixture.viz_stops(
                path=path, return_viz=return_viz, filtered_only=filtered_only
            )

    def test_viz_stops(self, multi_gtfs_fixture, tmp_path):
        """General tests for .viz_stops()."""
        # saving without returning
        save_path = os.path.join(tmp_path, "save_test.html")
        returned = multi_gtfs_fixture.viz_stops(
            path=save_path, return_viz=False
        )
        assert os.path.exists(save_path)
        assert isinstance(returned, type(None))
        # saving with returning
        save_path = os.path.join(tmp_path, "save_test2.html")
        returned = multi_gtfs_fixture.viz_stops(
            path=save_path, return_viz=True
        )
        assert os.path.exists(save_path)
        assert isinstance(returned, folium.Map)
        # returning without save
        returned = multi_gtfs_fixture.viz_stops(return_viz=True)
        assert isinstance(returned, folium.Map)
        files = glob.glob(f"{tmp_path}/*.html")
        assert len(files) == 2, "More files saved than expected"
