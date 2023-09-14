"""Test scripts for the GTFS report utility functions."""

import os
import pathlib
import re

import pytest
from pyprojroot import here

from transport_performance.gtfs.report.report_utils import (
    TemplateHTML,
    _set_up_report_dir,
)


@pytest.fixture(scope="function")
def template_fixture():
    """Fixture for test funcs expecting a valid feed object."""
    template = TemplateHTML(
        path=here("tests/data/gtfs/report/html_template.html")
    )
    return template


class TestTemplateHTML(object):
    """Tests related to the TemplateHTML class."""

    def test_init(self, template_fixture):
        """Test initialising the TemplateHTML class."""
        expected_template = """<!DOCTYPE html>
<html lang="en">

<body>
    <div>[test_placeholder] Tester [test_placeholder]</div>
</body>
"""
        assert (
            expected_template == template_fixture._get_template()
        ), "Test template not as expected"

    def test_insert_defence(self, template_fixture):
        """Test defences for .insert()."""
        with pytest.raises(
            ValueError,
            match=(
                "You have selected not to replace multiple"
                "placeholders of the same value, however"
                "placeholders occur more than once. \n"
                "If you would like to allow this, set the"
                "replace_multiple param to True"
            ),
        ):
            template_fixture._insert("test_placeholder", "test_value")

    def test_insert_on_pass(self, template_fixture):
        """Test functionality for .insert() when defences are passed."""
        expected_template = """<!DOCTYPE html>
<html lang="en">

<body>
    <div>test_value Tester test_value</div>
</body>
"""
        template_fixture._insert(
            placeholder="test_placeholder",
            value="test_value",
            replace_multiple=True,
        )
        assert (
            expected_template == template_fixture._get_template()
        ), "Test placeholder replacement not acting as expected"


class TestSetUpReportDir(object):
    """Test setting up a dir for a report."""

    def test_set_up_report_dir_defence(self):
        """Test the defences for set_up_report_dir()."""
        with pytest.raises(
            FileExistsError,
            match=(
                re.escape(
                    "Report already exists at path: "
                    "[tests/data/gtfs/report]."
                    "Consider setting overwrite=True"
                    "if you'd like to overwrite this."
                )
            ),
        ):
            _set_up_report_dir("tests/data/gtfs/report")

    def test_set_up_report_dir_on_pass(self, tmp_path):
        """Test set_up_report_dir() when defences are passed."""
        # create original report
        _set_up_report_dir(
            pathlib.Path(os.path.join(tmp_path)), overwrite=False
        )
        assert os.path.exists(
            os.path.join(tmp_path, "gtfs_report")
        ), "Failed to create report in temporary directory"
        # attempt to overwrite the previous report
        _set_up_report_dir(
            pathlib.Path(os.path.join(tmp_path)), overwrite=True
        )
        assert os.path.exists(
            os.path.join(tmp_path, "gtfs_report")
        ), "Failed to create report in temporary directory"
