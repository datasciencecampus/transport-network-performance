"""test_setup.py.

Unit tests for testing initial setup. The intention is these tests won't be
part of the main test suite, and will only by run as needed.
Runs with pytest flag --runsetup
"""

import pytest


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
