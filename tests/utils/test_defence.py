"""Tests for defence.py. These internals may be covered elsewhere."""
import re
import os
import pathlib

import pytest
import pandas as pd
from pyprojroot import here

from transport_performance.utils.defence import (
    _check_list,
    _check_parent_dir_exists,
    _type_defence,
    _check_column_in_df,
    _check_item_in_list,
    _check_attribute,
    _is_expected_filetype,
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


@pytest.fixture(scope="function")
def test_df():
    """A test fixture for an example dataframe."""
    test_df = pd.DataFrame(
        {"test_col_1": [1, 2, 3, 4], "test_col_2": [True, True, False, True]}
    )
    return test_df


class Test_CheckColumnInDf(object):
    """Tests for _check_column_in_df()."""

    def test__check_column_in_df_defence(self, test_df):
        """Defensive tests for _check_colum_in_df()."""
        with pytest.raises(
            IndexError, match="'test' is not a column in the dataframe."
        ):
            _check_column_in_df(df=test_df, column_name="test")

    def test__check_column_in_df_on_pass(self, test_df):
        """General tests for _check_colum_in_df()."""
        _check_column_in_df(test_df, "test_col_1")
        _check_column_in_df(test_df, "test_col_2")


@pytest.fixture(scope="function")
def test_list():
    """Test fixture."""
    my_list = ["test", "test2", "tester", "definitely_testing"]
    return my_list


class TestCheckItemInList(object):
    """Tests for _check_item_in_list()."""

    def test_check_item_in_list_defence(self, test_list):
        """Defensive tests for check_item_in_list()."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'test' expected one of the following:"
                f"{test_list} Got not_in_test"
            ),
        ):
            _check_item_in_list(
                item="not_in_test", _list=test_list, param_nm="test"
            )

    def test_check_item_in_list_on_pass(self, test_list):
        """General tests for check_item_in_list()."""
        _check_item_in_list(item="test", _list=test_list, param_nm="test")


@pytest.fixture(scope="function")
def dummy_obj():
    """Fixture to assist with tests."""

    class dummy:
        """Dummy class for testing."""

        def __init__(self) -> None:
            """Intialise dummy object."""
            self.tester = "test"
            self.tester_also = "also_test"

    new_dummy = dummy()
    return new_dummy


class TestCheckAttribute(object):
    """Tests for _check_item_in_list()."""

    def test_check_attribute_defence(self, dummy_obj):
        """Defensive tests for check_attribute."""
        with pytest.raises(AttributeError, match="dummy test msg"):
            _check_attribute(
                obj=dummy_obj, attr="not_in_test", message="dummy test msg"
            )

    def test_check_attribute_on_pass(self, dummy_obj):
        """General tests for check_attribute()."""
        _check_attribute(dummy_obj, "tester")


class Test_IsExpectedFiletype(object):
    """Tests for _is_expected_filetype."""

    def test_is_expected_filetype_raises_single(self):
        """Test when `exp_ext` is a single string."""
        with pytest.raises(
            ValueError,
            match="`raster` expected file extension .gif. Found .tiff",
        ):
            _is_expected_filetype(
                "some-raster.tiff",
                "raster",
                check_existing=False,
                exp_ext=".gif",
            )
        with pytest.raises(
            ValueError,
            match="`gtfs.zip` expected file extension .tiff. Found .zip",
        ):
            _is_expected_filetype(
                here("tests/data/newport-20230613_gtfs.zip"),
                param_nm="gtfs.zip",
                check_existing=True,
                exp_ext=".tiff",
            )

    def test_is_expected_filetype_raises_multiple(self):
        """Test raises when `exp_ext` is a list of multiple file extensions."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "`raster` expected file extension ['.gif', '.jiff']. Found .ti"
                "ff"
            ),
        ):
            _is_expected_filetype(
                "some_raster.tiff",
                "raster",
                check_existing=False,
                exp_ext=[".gif", ".jiff"],
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "`osm.pbf` expected file extension ['.zip', '.gif', '.pdf']. F"
                "ound .pbf"
            ),
        ):
            _is_expected_filetype(
                "tests/data/newport-2023-06-13.osm.pbf",
                "osm.pbf",
                check_existing=True,
                exp_ext=[".zip", ".gif", ".pdf"],
            )

    def test_is_expected_filetype_on_pass(self):
        """Test when `exp_ext` passes."""
        result = _is_expected_filetype(
            "some_raster.tiff",
            "raster",
            check_existing=False,
            exp_ext=[".gif", ".tiff"],
        )
        assert result is None
        result = _is_expected_filetype(
            "tests/data/newport-2023-06-13.osm.pbf",
            "osm.pbf",
            check_existing=True,
            exp_ext=[".zip", ".gif", ".pbf"],
        )
        assert result is None
