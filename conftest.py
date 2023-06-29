"""conftest.py.

`pytest` configuration file. Currently used to flag tests for set-up only.
Reworked example from pytest docs:
https://docs.pytest.org/en/latest/example/simple.html.
"""

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


def pytest_configure(config):
    """Add ini value line."""
    config.addinivalue_line("markers", "setup: mark test to run during setup")
    config.addinivalue_line(
        "markers", "runinteg: mark test to run for integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Handle switching based on cli args."""
    if config.getoption("--runsetup"):
        # --runsetup given in cli: do not skip slow tests
        return
    skip_setup = pytest.mark.skip(reason="need --runsetup option to run")
    for item in items:
        if "setup" in item.keywords:
            item.add_marker(skip_setup)

    if config.getoption("--runinteg"):
        return
    skip_runinteg = pytest.mark.skip(reason="need --runinteg option to run")
    for item in items:
        if "runinteg" in item.keywords:
            item.add_marker(skip_runinteg)
