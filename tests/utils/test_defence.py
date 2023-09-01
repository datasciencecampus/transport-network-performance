"""Tests for defence.py. These internals may be covered elsewhere."""
import re
import os
import pathlib

import pytest

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
                "<class 'str'>"
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

        error_pth = "test_folder\\test_file.py"
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Please specify string paths with single forward"
                " slashes only."
                f" Got {repr(error_pth)}"
            ),
        ):
            _check_parent_dir_exists(
                pth="test_folder\\test_file.py",
                param_nm="test_prm",
                create=False,
            )

    def test_check_parents_dir_exists(self, tmp_path):
        """Test that a parent directory is created."""
        # test without create
        expected_error_path = os.path.join(tmp_path, "data_path", "data_path")
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                rf"Parent directory {expected_error_path}"
                r" not found on disk."
            ),
        ):
            _check_parent_dir_exists(
                pth=pathlib.Path(
                    os.path.join(
                        tmp_path, "data_path", "data_path", "test.html"
                    )
                ),
                param_nm="test_prm",
                create=False,
            )

        # test creating the parent directory (1 level)
        _check_parent_dir_exists(
            pth=pathlib.Path(
                os.path.join(tmp_path, "test_dir_single", "test.html")
            ),
            param_nm="test_prm",
            create=True,
        )

        assert os.path.exists(
            pathlib.Path(os.path.join(tmp_path, "test_dir_single"))
        ), (
            "_check_parent_dir_exists did not make parent dir"
            " when 'create=True' (single level)"
        )

        # test creating the parent directory (2 levels)
        _check_parent_dir_exists(
            pth=pathlib.Path(
                os.path.join(
                    tmp_path, "test_dir_multi", "test_dir_multi", "test.html"
                )
            ),
            param_nm="test_prm",
            create=True,
        )

        assert os.path.exists(
            pathlib.Path(
                os.path.join(tmp_path, "test_dir_multi", "test_dir_multi")
            )
        ), (
            "_check_parent_dir_exists did not make parent dir"
            " when 'create=True' (multiple levels)"
        )
