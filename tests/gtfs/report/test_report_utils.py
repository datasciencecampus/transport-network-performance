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
        assert expected_template.replace(
            r"\n", ""
        ) == template_fixture._get_template().replace(
            r"\n", ""
        ), "Test template not as expected"

    def test__insert_defence(self, template_fixture):
        """Test defences for ._insert()."""
        with pytest.raises(
            ValueError,
            match=(
                "`replace_multiple` requires True as found \n"
                "multiple placeholder matches in template."
            ),
        ):
            template_fixture._insert("test_placeholder", "test_value")

    def test__insert_on_pass(self, template_fixture):
        """Test functionality for ._insert() when defences are passed."""
        template_fixture._insert(
            placeholder="test_placeholder",
            value="test_value_insert_test",
            replace_multiple=True,
        )
        assert (
            "test_value_insert_test"
            in template_fixture._get_template().replace(r"\n", "")
        ), ("Test placeholder replacement not acting as expected")


class TestSetUpReportDir(object):
    """Test setting up a dir for a report."""

    def test__set_up_report_dir_defence(self, tmp_path):
        """Test the defences for _set_up_report_dir()."""
        _set_up_report_dir(os.path.join(tmp_path))
        with pytest.raises(
            FileExistsError,
            match=(
                re.escape(
                    "Report already exists at path: "
                    f"[{tmp_path}]."
                    "Consider setting overwrite=True"
                    "if you'd like to overwrite this."
                )
            ),
        ):
            _set_up_report_dir(os.path.join(tmp_path), overwrite=False)

    def test__set_up_report_dir_on_pass(self, tmp_path):
        """Test _set_up_report_dir() when defences are passed."""
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
        # attempt to create report in different paths
        _set_up_report_dir(os.path.join(tmp_path, "testing"))
        assert os.path.exists(
            os.path.join(tmp_path, "testing", "gtfs_report")
        ), (
            f"Failed to create report dir in {tmp_path}/testing/" "gtfs_report"
        )
        _set_up_report_dir(os.path.join(tmp_path, "testing", "testing"))
        assert os.path.exists(
            os.path.join(tmp_path, "testing", "testing", "gtfs_report")
        ), (
            f"Failed to create report dir in {tmp_path}/testing/testing/"
            "gtfs_report"
        )
