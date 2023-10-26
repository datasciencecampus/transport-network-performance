"""Tests for defence.py. These internals may be covered elsewhere."""
import re
import os
import pathlib
from typing import Union, Type

import pytest
from _pytest.python_api import RaisesContext
import pandas as pd
from pyprojroot import here

from transport_performance.utils.defence import (
    _check_iterable,
    _check_parent_dir_exists,
    _gtfs_defence,
    _type_defence,
    _check_column_in_df,
    _check_item_in_iter,
    _check_attribute,
    _handle_path_like,
    _is_expected_filetype,
    _enforce_file_extension,
)


class Test_CheckIter(object):
    """Test internal _check_iterable."""

    def test__check_iter_only(self):
        """Func raises as expected when not checking iterable elements."""
        # not iterable
        with pytest.raises(
            TypeError,
            match="`some_bool` expected .*Iterable.* Got .*bool.*",
        ):
            _check_iterable(
                iterable=True,
                param_nm="some_bool",
                iterable_type=list,
                check_elements=False,
            )

        # iterable does not match provided type
        with pytest.raises(
            TypeError,
            match="`some_tuple` expected .*list.* Got .*tuple.*",
        ):
            _check_iterable(
                iterable=(1, 2, 3),
                param_nm="some_tuple",
                iterable_type=list,
                check_elements=False,
            )

        # iterable_type is not type
        with pytest.raises(
            TypeError,
            match="`iterable_type` expected .*type.* Got .*str.*",
        ):
            _check_iterable(
                iterable=(1, 2, 3),
                param_nm="some_tuple",
                iterable_type="tuple",
                check_elements=False,
            )

    def test__check_iter_elements(self):
        """Func raises as expected when checking list elements."""
        # mixed types
        with pytest.raises(
            TypeError,
            match=(
                "`mixed_list` must contain <class 'int'> only. Found "
                "<class 'str'> : 2"
            ),
        ):
            _check_iterable(
                iterable=[1, "2", 3],
                param_nm="mixed_list",
                iterable_type=list,
                check_elements=True,
                exp_type=int,
            )

        # wrong expected types
        with pytest.raises(
            TypeError,
            match=("`exp_type` expected .*type.*tuple.*" "Got .*str.*"),
        ):
            _check_iterable(
                iterable=["1", "2", "3"],
                param_nm="param",
                iterable_type=list,
                check_elements=True,
                exp_type="str",
            )

        # wrong types in exp_type tuple
        with pytest.raises(
            TypeError,
            match=("`exp_type` must contain types only.* Found .*str.*: str"),
        ):
            _check_iterable(
                iterable=[1, "2", 3],
                param_nm="param",
                iterable_type=list,
                check_elements=True,
                exp_type=(int, "str"),
            )

    def test__check_iter_passes(self):
        """Test returns None when pass conditions met."""
        # check list and element type
        assert (
            _check_iterable(
                iterable=[1, 2, 3],
                param_nm="int_list",
                iterable_type=list,
                exp_type=int,
            )
            is None
        )

        # check list and multiple element types
        assert (
            _check_iterable(
                iterable=[1, "2", 3],
                param_nm="int_list",
                iterable_type=list,
                exp_type=(int, str),
            )
            is None
        )

        # check tuple
        assert (
            _check_iterable(
                iterable=(False, True),
                param_nm="bool_list",
                iterable_type=tuple,
                check_elements=False,
            )
            is None
        )

    def test__check_iter_length(self):
        """Func raises as expected when length of iterable does not match."""
        # wrong length
        with pytest.raises(
            ValueError,
            match=("`list_3` is of length 3. Expected length 2."),
        ):
            _check_iterable(
                iterable=[1, 2, 3],
                param_nm="list_3",
                iterable_type=list,
                check_elements=False,
                check_length=True,
                length=2,
            )

    def test__check_iter_length_pass(self):
        """Test returns None when pass conditions met."""
        _check_iterable(
            iterable=[1, 2, 3],
            param_nm="list_3",
            iterable_type=list,
            check_elements=False,
            check_length=True,
            length=3,
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


def test__gtfs_defence():
    """Tests for _gtfs_defence()."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "'test' expected a GtfsInstance object. " "Got <class 'str'>"
        ),
    ):
        _gtfs_defence("tester", "test")


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
    """Tests for _check_item_in_iter()."""

    def test_check_item_in_iter_defence(self, test_list):
        """Defensive tests for check_item_in_iter()."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'test' expected one of the following: "
                f"{test_list}. Got not_in_test: <class 'str'>"
            ),
        ):
            _check_item_in_iter(
                item="not_in_test", iterable=test_list, param_nm="test"
            )

    def test_check_item_in_iter_on_pass(self, test_list):
        """General tests for check_item_in_iter()."""
        _check_item_in_iter(item="test", iterable=test_list, param_nm="test")


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
    """Tests for _check_item_in_iter()."""

    def test_check_attribute_defence(self, dummy_obj):
        """Defensive tests for check_attribute."""
        with pytest.raises(AttributeError, match="dummy test msg"):
            _check_attribute(
                obj=dummy_obj, attr="not_in_test", message="dummy test msg"
            )

    def test_check_attribute_on_pass(self, dummy_obj):
        """General tests for check_attribute()."""
        _check_attribute(dummy_obj, "tester")


class Test_HandlePathLike(object):
    """Tests for _handle_path_like()."""

    # Paremetrize tests. First dictionary contains string like paths.
    pth_str = {
        "unix_pth": ["foo/bar", "/transport-network-performance/foo/bar"],
        "unix_symlink": [
            "foo/bar/../baz",
            "/transport-network-performance/foo/baz",
        ],
        "windows_single": [
            r"foo\bar",
            "/transport-network-performance/foo/bar",
        ],
        "windows_double": [
            "foo\\bar",
            "/transport-network-performance/foo/bar",
        ],
        "windows_mixed": [
            r"foo\\bar\baz",
            "transport-network-performance/foo/bar/baz",
        ],
        "windows_symlink": [
            r"foo\\bar\\..\baz",
            "/transport-network-performance/foo/baz",
        ],
    }
    # second dict contains unix like, representing the user passing
    # pyprojroot.here values to _handle_path_like()
    pth_posix = {
        "unix_here": [
            here(pth_str["unix_pth"][0]),
            "/transport-network-performance/foo/bar",
        ],
        "unix_here_symlink": [
            here(pth_str["unix_symlink"][0]),
            "/transport-network-performance/foo/baz",
        ],
        "windows_single_here": [
            here(rf"{pth_str['windows_single'][0]}"),
            "/transport-network-performance/foo/bar",
        ],
        "windows_double_here": [
            here(pth_str["windows_double"][0]),
            "/transport-network-performance/foo/bar",
        ],
        "windows_mixed_here": [
            here(rf"{pth_str['windows_mixed'][0]}"),
            "/transport-network-performance/foo/bar/baz",
        ],
        "windows_here_symlink": [
            here(pth_str["windows_symlink"][0]),
            "/transport-network-performance/foo/baz",
        ],
    }

    @pytest.mark.parametrize(
        "param_nm, path, expected",
        [(k, v[0], v[1]) for k, v in pth_str.items()],
    )
    def test__handle_path_like_with_strings(self, param_nm, path, expected):
        """For all keys in pth_str, test the path against the expected path."""
        if os.name == "nt":
            expected = expected.replace("/", "\\")
        pth = _handle_path_like(path, param_nm).__str__()
        assert pth.endswith(expected), f"Expected: {expected}, Found: {pth}"

    @pytest.mark.parametrize(
        "param_nm, path, expected",
        [(k, v[0], v[1]) for k, v in pth_posix.items()],
    )
    def test__handle_path_like_with_posix_pths(self, param_nm, path, expected):
        """For all keys in posix_pth, test path against expected."""
        # when all is said and done, on Windows you'll get backward slashes
        exp_return_class = pathlib.PosixPath
        if os.name == "nt":
            expected = expected.replace("/", "\\")
            exp_return_class = pathlib.WindowsPath
        pth = _handle_path_like(path, param_nm)
        pth_str = pth.__str__()
        assert pth_str.endswith(
            expected
        ), f"Expected: {expected}, Found: {pth}"
        assert isinstance(
            pth, exp_return_class
        ), f"Expected {exp_return_class}, found: {type(pth)}"

    def test__handle_path_like_raises(self):
        """Func raises if pth is not a path-like or str."""
        with pytest.raises(
            TypeError,
            match="`empty_tuple` expected path-like, found <class 'tuple'>",
        ):
            _handle_path_like(tuple(), "empty_tuple")


class Test_IsExpectedFiletype(object):
    """Tests for _is_expected_filetype."""

    def test_is_expected_filetype_raises_single(self):
        """Test raises when `exp_ext` is a single string."""
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
                os.path.join(
                    "tests", "data", "gtfs", "newport-20230613_gtfs.zip"
                ),
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

    def test_is_expected_filetype_defence(self):
        """Test defensive behaviour."""
        with pytest.raises(
            ValueError, match="No file extension was found in .*noextension"
        ):
            _is_expected_filetype(
                pth="noextension", param_nm="noextension", check_existing=False
            )
        # check warnings for adding '.' to exp_ext if forgotten
        with pytest.warns(
            UserWarning, match="'.' was prepended to the `exp_ext`."
        ):
            _is_expected_filetype("foo.bar", "foobar", False, exp_ext="BaR")
        with pytest.warns(
            UserWarning, match="'.' was prepended to `exp_ext` value 'pbf'."
        ):
            _is_expected_filetype(
                "bar.baZ", "barbaz", False, exp_ext=["PBF", ".BAz"]
            )


class Test_EnforceFileExtension(object):
    """Tests for _enforce_file_extension()."""

    @pytest.mark.parametrize(
        (
            "path, exp_ext, default_ext, param_nm, msg, expected_warn,"
            " expected_pth"
        ),
        [
            (
                "invalid.txt",
                ".html",
                ".html",
                "test",
                None,
                pytest.warns(
                    UserWarning,
                    match=(
                        re.escape(
                            "Format .txt provided. Expected ['html'] for path"
                            " given to 'test'. Path defaulted to .html"
                        )
                    ),
                ),
                "invalid.html",
            ),
            # test with custom error message
            (
                "invalid.txt",
                ".html",
                ".html",
                "test",
                "custom message test",
                pytest.warns(
                    UserWarning, match=(re.escape("custom message test"))
                ),
                "invalid.html",
            ),
            # list of acceptable types
            (
                "invalid.txt",
                [".html", ".xml"],
                ".xml",
                "test",
                None,
                pytest.warns(
                    UserWarning,
                    match=(
                        re.escape(
                            "Format .txt provided. Expected ['html', 'xml'] "
                            "for path given to 'test'. Path defaulted to .xml"
                        )
                    ),
                ),
                "invalid.xml",
            ),
        ],
    )
    def test__enforce_file_extension_warns(
        self,
        path: Union[str, pathlib.Path],
        exp_ext: Union[str, list],
        default_ext: str,
        param_nm: str,
        msg: str,
        expected_warn: Type[RaisesContext],
        expected_pth: Union[str, pathlib.Path],
        tmp_path,
    ) -> None:
        """Tests for _enforce_file_extension() raising warnings."""
        path = os.path.join(tmp_path, path)
        expected_pth = os.path.join(tmp_path, expected_pth)
        with expected_warn:
            new_pth = _enforce_file_extension(
                path, exp_ext, default_ext, param_nm, msg
            )
            assert str(new_pth) == expected_pth, "new path not as expected"

    @pytest.mark.parametrize(
        ("path, exp_ext, default_ext, param_nm, msg, expected_pth"),
        [
            # single accepted type
            ("valid.txt", ".txt", ".txt", "test", None, "valid.txt"),
            # list of accepted types
            ("valid.py", [".txt", ".py"], ".txt", "test", None, "valid.py"),
        ],
    )
    def test__enforce_file_extension_on_pass(
        self,
        path: Union[str, pathlib.Path],
        exp_ext: Union[str, list],
        default_ext: str,
        param_nm: str,
        msg: str,
        expected_pth: Union[str, pathlib.Path],
        tmp_path,
    ) -> None:
        """Tests for _enforce_file_extension() on pass."""
        path = os.path.join(tmp_path, path)
        expected_pth = os.path.join(tmp_path, expected_pth)
        new_path = _enforce_file_extension(
            path, exp_ext, default_ext, param_nm, msg
        )
        assert str(new_path) == expected_pth, "New path not as expected"
