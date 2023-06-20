"""Tests for r5 utility functions."""
import os
import pytest
from heimdall_transport.gtfs import r5_helpers


class TestCheckR5Setup(object):
    """Class to group tests for r5_helpers.test_r5_setup."""

    @pytest.mark.setup
    def test_r5py_setup(self) -> None:
        """Check if r5 has created the expected db files from test fixture."""
        x, y = r5_helpers.check_r5_setup()
        assert os.path.exists(
            x
        ), f"r5py did not create the expected file at:/ {x}"
        assert os.path.exists(
            y
        ), f"r5py did not create the expected file at:/ {y}"
