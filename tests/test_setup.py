"""test_setup.py.

Unit tests for testing initial setup. The intention is these tests won't be
part of the main test suite, and will only be run as needed.

This test module can be run with the pytest flag --runsetup.
"""

import pytest
import os


class TestSetup:
    """A class of tests to check correct set-up.

    A set of tests to:
        1. check `transport_performance` is installed.
        2. check `r5py` is installed and java interface is functional
    """

    @pytest.mark.setup
    def test_transport_performance_install(self) -> None:
        """Check `transport_performance is installed.

        This is a simple test to check the package has been installed and is
        available for use.
        """
        try:
            import transport_performance as tp

            assert tp.__name__ == "transport_performance"
        except ImportError:
            pytest.fail("Unable to find `transport_performance`.")

    @pytest.mark.setup
    def test_r5py_setup(self) -> None:
        """Check development environment will cope with r5py requirements."""
        # this import is only here to prevent java messages being printed to
        # the CLI when set-up pytests are not run
        from r5py import TransportNetwork

        # search the ext dir for pbf & gtfs
        test_data = os.path.join("tests", "data")
        gtfs_data = os.path.join(test_data, "gtfs")
        found_gtfs = os.listdir(gtfs_data)
        foundf = os.listdir(test_data)
        gtfs = [
            os.path.join(gtfs_data, x)
            for x in found_gtfs
            if x.endswith(".zip")
        ][0]
        pbf = [
            os.path.join(test_data, x) for x in foundf if x.endswith(".pbf")
        ][0]

        # needs wrapping in try but specific exception to raise unknown.
        # Examining r5py exception classes, I'll go with the below.
        try:
            TransportNetwork(pbf, [gtfs])
        except RuntimeError:
            print("RuntimeError encountered")
            pass
        except MemoryError:
            print("Memory error encountered")
            pass

        # has a .mapdb file been created in the external directory?
        mapdb_f = pbf + ".mapdb"
        mapdb_p_f = pbf + ".mapdb.p"
        assert os.path.exists(
            mapdb_f
        ), f"r5py did not create the expected file at:/ {mapdb_f}"
        assert os.path.exists(
            mapdb_p_f
        ), f"r5py did not create the expected file at:/ {mapdb_p_f}"
