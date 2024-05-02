"""conftest.py.

`pytest` configuration file. Currently used to flag tests for set-up only.
Reworked example from pytest docs:
https://docs.pytest.org/en/latest/example/simple.html.
"""

import pandas as pd
import pytest


def pytest_addoption(parser):
    """Adapt pytest cli args, and give more info when -h flag is used."""
    parser.addoption(
        "--runsetup",
        action="store_true",
        default=False,
        help="run set-up tests",
    )
    parser.addoption(
        "--runinteg",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--runexpensive",
        action="store_true",
        default=False,
        help="run expensive tests",
    )
    parser.addoption(
        "--sanitycheck",
        action="store_true",
        default=False,
        help="run sanity checks",
    )


def pytest_configure(config):
    """Add ini value line."""
    config.addinivalue_line("markers", "setup: mark test to run during setup")
    config.addinivalue_line(
        "markers", "runinteg: mark test to run for integration tests"
    )
    config.addinivalue_line(
        "markers", "runexpensive: mark test to run expensive tests"
    )
    config.addinivalue_line(
        "markers",
        "sanitycheck: mark test to run checks in dependencies' code.",
    )


def pytest_collection_modifyitems(config, items):  # noqa:C901
    """Handle switching based on cli args."""
    if (
        config.getoption("--runsetup")
        & config.getoption("--runinteg")
        & config.getoption("--runexpensive")
        & config.getoption("--sanitycheck")
    ):
        # do full test suite when all flags are given
        return

    # do not add setup marks when the runsetup flag is given
    if not config.getoption("--runsetup"):
        skip_setup = pytest.mark.skip(reason="need --runsetup option to run")
        for item in items:
            if "setup" in item.keywords:
                item.add_marker(skip_setup)

    # do not add integ marks when the runinteg flag is given
    if not config.getoption("--runinteg"):
        skip_runinteg = pytest.mark.skip(
            reason="need --runinteg option to run"
        )
        for item in items:
            if "runinteg" in item.keywords:
                item.add_marker(skip_runinteg)

    # do not add expensive marks when the runexpensive flag is given
    if not config.getoption("--runexpensive"):
        skip_runexpensive = pytest.mark.skip(
            reason="need --runexpensive option to run"
        )
        for item in items:
            if "runexpensive" in item.keywords:
                item.add_marker(skip_runexpensive)

    # do not add sanitycheck marks when the sanitycheck flag is given
    if not config.getoption("--sanitycheck"):
        skip_sanitycheck = pytest.mark.skip(
            reason="need --sanitycheck option to run"
        )
        for item in items:
            if "sanitycheck" in item.keywords:
                item.add_marker(skip_sanitycheck)


@pytest.fixture(scope="session")
def _EXPECTED_VALIDITY_DF():
    validity_df = pd.DataFrame(
        {
            "type": [
                "warning",
                "warning",
                "warning",
                "warning",
                "warning",
                "warning",
            ],
            "message": [
                "Unrecognized column agency_noc",
                "Feed expired",
                "Unrecognized column platform_code",
                "Unrecognized column vehicle_journey_code",
                "Fast Travel Between Consecutive Stops",
                "Fast Travel Over Multiple Stops",
            ],
            "table": [
                "agency",
                "calendar",
                "stops",
                "trips",
                "full_stop_schedule",
                "multiple_stops_invalid",
            ],
            "rows": [
                [],
                [],
                [],
                [],
                [457, 458, 4596, 4597, 5788, 5789],
                [0, 1, 2],
            ],
        }
    )
    return validity_df
