"""Tests for defence.py. These internals may be covered elsewhere."""
import re
import os
import pathlib

import pytest

from transport_performance.utils.defence import (
    _check_list,
    _check_parent_dir_exists,
    _type_defence,
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


class Test_TypeDefence(object):
    """Assertions for _type_defence()."""

    def test_type_defence_raises_on_single_types(self):
        """Assert func raises for single values to the `types` parameter."""
        with pytest.raises(
            TypeError,
            match="`empty_list` expected <class 'str'>. Got <class 'list'>",
        ):
            _type_defence(list(), "empty_list", str)
        with pytest.raises(
            TypeError,
            match="`int_1` expected <class 'list'>. Got <class 'int'>",
        ):
            _type_defence(1, "int_1", list)
        with pytest.raises(
            TypeError,
            match="`string_1` expected <class 'int'>. Got <class 'str'>",
        ):
            _type_defence("1", "string_1", int)
        with pytest.raises(
            TypeError,
            match="`float_1` expected <class 'int'>. Got <class 'float'>",
        ):
            _type_defence(1.0, "float_1", int)
        with pytest.raises(
            TypeError,
            match="`empty_dict` expected <class 'tuple'>. Got <class 'dict'>",
        ):
            _type_defence(dict(), "empty_dict", tuple)
        with pytest.raises(
            TypeError,
            match="`empty_tuple` expected <class 'dict'>. Got <class 'tuple'>",
        ):
            _type_defence(tuple(), "empty_tuple", dict)
        with pytest.raises(
            TypeError,
            match="`None` expected <class 'int'>. Got <class 'NoneType'>",
        ):
            _type_defence(None, "None", int)

    def test_type_defence_raises_on_multiple_types(object):
        """Assert func raises for multiple values to the `types` parameter."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "pected (<class 'str'>, <class 'NoneType'>). Got <class 'int'>"
            ),
        ):
            _type_defence(1, "int_1", (str, type(None)))
        with pytest.raises(
            TypeError,
            match=re.escape(
                "`str_1` expected (<class 'int'>, <class 'float'>, <class 'Non"
            ),
        ):
            _type_defence("1", "str_1", (int, float, type(None)))
        with pytest.raises(
            TypeError,
            match=re.escape(
                "`float_1` expected (<class 'int'>, <class 'str'>, <class 'Non"
            ),
        ):
            _type_defence(1.0, "float_1", (int, str, type(None)))
        with pytest.raises(
            TypeError,
            match=re.escape(
                "`empty_dict` expected (<class 'NoneType'>, <class 'str'>, <cl"
            ),
        ):
            _type_defence(
                dict(), "empty_dict", (type(None), str, bool, list, tuple)
            )
        with pytest.raises(
            TypeError,
            match=re.escape(
                "`empty_list` expected (<class 'NoneType'>, <class 'str'>, <cl"
            ),
        ):
            _type_defence(list(), "empty_list", (type(None), str, dict, tuple))
        with pytest.raises(
            TypeError,
            match=re.escape(
                "`empty_tuple` expected (<class 'NoneType'>, <class 'list'>, <"
            ),
        ):
            _type_defence(
                tuple(),
                "empty_tuple",
                (type(None), list, dict, str, int, float),
            )
        with pytest.raises(
            TypeError,
            match=re.escape(
                "`None` expected (<class 'int'>, <class 'str'>, <class 'float'"
            ),
        ):
            _type_defence(None, "None", (int, str, float))

    def test_type_defence_passes_on_single_types(self):
        """Assert func passes on single values to the `types` parameter."""
        _type_defence(1, "int_1", int)
        _type_defence(1.0, "float_1", float)
        _type_defence("1", "str_1", str)
        _type_defence(dict(), "empty_dict", dict)
        _type_defence(tuple(), "empty_tuple", tuple)
        _type_defence(list(), "empty_list", list)
        _type_defence(None, "None", type(None))

    def test_type_defence_passes_on_multiple_types(self):
        """Assert func passes on multiple values to the `types` parameter."""
        _type_defence(1, "int_1", (tuple, int))
        _type_defence("1", "str_1", (int, float, str))
        _type_defence(1.0, "float_1", (float, type(None)))
        _type_defence(dict(), "empty_dict", (tuple, dict))
        _type_defence(list(), "empty_list", (type(None), list))
        _type_defence(tuple(), "empty_tuple", (tuple, dict))
        _type_defence(None, "None", (list, dict, type(None)))
