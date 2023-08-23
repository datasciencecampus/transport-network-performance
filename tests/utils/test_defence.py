"""Tests for defence.py. These internals may be covered elsewhere."""
import re
import os
import shutil

import pytest
from pyprojroot import here

from transport_performance.utils.defence import (
    _check_list,
    _check_parent_dir_exists,
)


class Test_CheckList(object):
    """Test internal _check_list."""

    def test__check_list_only(self):
        """Func raises as expected when not checking list elements."""
        with pytest.raises(
            TypeError,
            match="`some_bool` should be a list. Instead found <class 'bool'>",
        ):
            _check_list(ls=True, param_nm="some_bool", check_elements=False)

    def test__check_list_elements(self):
        """Func raises as expected when checking list elements."""
        with pytest.raises(
            TypeError,
            match=(
                "`mixed_list` must contain <class 'int'> only. Found "
                "<class 'str'> : 2"
            ),
        ):
            _check_list(
                ls=[1, "2", 3],
                param_nm="mixed_list",
                check_elements=True,
                exp_type=int,
            )

    def test__check_list_passes(self):
        """Test returns None when pass conditions met."""
        assert (
            _check_list(ls=[1, 2, 3], param_nm="int_list", exp_type=int)
            is None
        )
        assert (
            _check_list(
                ls=[False, True], param_nm="bool_list", check_elements=False
            )
            is None
        )


class Test_CheckParentDirExists(object):
    """Assertions for check_parent_dir_exists."""

    def test_check_parent_dir_exists_defence(self):
        """Check defence for _check_parent_dir_exists()."""
        with pytest.raises(
            FileNotFoundError,
            match="Parent directory .*missing not found on disk.",
        ):
            _check_parent_dir_exists(
                pth="missing/file.someext", param_nm="not_found", create=False
            )

    def test_check_parents_dir_exists(self):
        """Test that a parent directory is created."""
        # test without create
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                "Parent directory D:\\DSC\\transport-network-performance"
                "\\data\\interim\\test_dir not found on disk"
            ),
        ):
            _check_parent_dir_exists(
                pth="data/interim/test_dir/test_dir.html",
                param_nm="test_prm",
                create=False,
            )

        # test creating the parent directory (1 level)
        _check_parent_dir_exists(
            pth="data/interim/test_dir/test_dir.html",
            param_nm="test_prm",
            create=True,
        )

        assert os.path.exists(here("data/interim/test_dir/")), (
            "_check_parent_dir_exists did not make parent dir"
            "when 'create=True' (single level)"
        )

        shutil.rmtree("data/interim/test_dir/")

        # test creating the parent directory (2 levels)
        _check_parent_dir_exists(
            pth="data/interim/test_dir/test_dir/test_dir.html",
            param_nm="test_prm",
            create=True,
        )

        assert os.path.exists(here("data/interim/test_dir/test_dir/")), (
            "_check_parent_dir_exists did not make parent dir"
            "when 'create=True' (multiple levels)"
        )

        shutil.rmtree("data/interim/test_dir")
