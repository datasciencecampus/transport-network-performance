"""Tests for multi_validation.py."""
import pytest
import os
import glob

import numpy as np

from transport_performance.gtfs.multi_validation import (
    MultiGtfsInstance,
    FileCountError,
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
        with pytest.raises(
            FileCountError, match="At least 2 files expected.*Found.*0"
        ):
            MultiGtfsInstance(f"{tmp_path}/*.zip")
        # not enough files found (1)
        with open(os.path.join(tmp_path, "test.txt"), "w") as f:
            f.write("This is a test.")
        with pytest.raises(
            FileCountError, match="At least 2 files expected.*Found.*1"
        ):
            MultiGtfsInstance(f"{tmp_path}/*.txt")
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
            m_gtfs.paths, multi_gtfs_paths
        ), "Paths not as expected"
        assert len(m_gtfs.paths) == 2, "Unexpected number of GTFS paths"
        assert (
            len(m_gtfs.instances) == 2
        ), "Unexpected number of GTFS instances"
        for inst in m_gtfs.instances:
            assert isinstance(
                inst, GtfsInstance
            ), "GtfsInstance not instanciated"

    def test_save(self, multi_gtfs_paths, tmp_path):
        """Tests for .save()."""
        gtfs = MultiGtfsInstance(multi_gtfs_paths)
        save_dir = os.path.join(tmp_path, "save_test")
        gtfs.save(save_dir)
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
            expected_paths, found_paths
        ), "GtfsInstances not saved as expected"

    def test_clean_feed_defences(self):
        """Defensive tests for .clean_feed()."""
        # To be completed once PR 195 is merged as there are breaking changes.
        # https://github.com/datasciencecampus/transport-network-performance/
        # pull/195
        pass

    def test_clean_feed_on_pasas(self):
        """General tests for .clean_feed()."""
        # To be completed once PR 195 is merged as there are breaking changes.
        # https://github.com/datasciencecampus/transport-network-performance/
        # pull/195
        pass

    def test_is_valid_defences(self):
        """Defensive tests for .is_valid()."""
        # To be completed once PR 195 is merged as there are breaking changes.
        # https://github.com/datasciencecampus/transport-network-performance/
        # pull/195
        pass

    def test_is_valid_on_pass(self):
        """General tests for is_valid()."""
        # To be completed once PR 195 is merged as there are breaking changes.
        # https://github.com/datasciencecampus/transport-network-performance/
        # pull/195
        pass

    def test_filter_to_date_defences(self, multi_gtfs_fixture):
        """Defensive tests for .filter_to_date()."""
        with pytest.raises(
            TypeError, match=".*dates.*expected.*str.*list.*int.*"
        ):
            multi_gtfs_fixture.filter_to_date(12)

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
        with pytest.raises(TypeError, match=".*crs.*expected.*str.*int.*"):
            multi_gtfs_fixture.filter_to_bbox([12, 12, 13, 13], 12)

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
