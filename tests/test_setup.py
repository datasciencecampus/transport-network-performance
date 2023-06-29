"""test_setup.py.

Unit tests for testing initial setup. The intention is these tests won't be
part of the main test suite, and will only be run as needed.

This test module can be run with the pytest flag --runsetup.
"""

import pytest
import os
from r5py import TransportNetwork


class TestSetup:
    """A class of tests to check correct set-up.

    A set of tests to:
        1. check `heimdall_transport` is installed.
    """

    @pytest.mark.setup
    def test_heimdall_transport_install(self) -> None:
        """Check `heimdall_transport is installed.

        This is a simple test to check the package has been installed and is
        available for use.
        """
        try:
            import heimdall_transport as ht

            assert ht.__name__ == "heimdall_transport"
        except ImportError:
            pytest.fail("Unable to find `heimdall_transport`.")

    @pytest.mark.setup
    def test_r5py_setup(self) -> None:
        """Check development environment will cope with r5py requirements."""
        # search the ext dir for pbf & gtfs
        search_pth = os.path.join("tests", "data")
        foundf = os.listdir(search_pth)
        gtfs = [
            os.path.join(search_pth, x) for x in foundf if x.endswith(".zip")
        ][0]
        pbf = [
            os.path.join(search_pth, x) for x in foundf if x.endswith(".pbf")
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
