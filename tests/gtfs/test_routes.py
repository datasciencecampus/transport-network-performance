"""Testing routes module."""
import pytest
import pandas as pd
from pyprojroot import here
from unittest.mock import call
from typing import Union, Type
import pathlib
from _pytest.python_api import RaisesContext
import re
import pickle
import os


from transport_performance.gtfs.routes import (
    scrape_route_type_lookup,
    get_saved_route_type_lookup,
)
from transport_performance.utils.constants import PKG_PATH


def mocked__get_response_text(*args):
    """Mock _get_response_text.

    Returns
    -------
        str: Minimal text representation of url tables.

    """
    k1 = "https://gtfs.org/schedule/reference/"
    v1 = "<td> <br><code>0</code> - Tram."
    k2 = (
        "https://developers.google.com/transit/gtfs/reference/"
        "extended-route-types"
    )
    v2 = """<table class="nice-table">
    <tbody>
      <tr>
        <th>Code</th>
        <th>Description</th>
        <th>Supported</th>
        <th>Examples</th>
      </tr>
      <tr>
        <td><strong>100</strong></td>
        <td><strong>Railway Service</strong></td>
        <td>Yes</td>
        <td>Not applicable (N/A)</td>
      </tr>"""

    return_vals = {k1: v1, k2: v2}
    return return_vals[args[0]]


class TestScrapeRouteTypeLookup(object):
    """Test scrape_route_type_lookup."""

    def test_defensive_exceptions(self):
        """Test the defensive checks raise as expected."""
        with pytest.raises(
            TypeError,
            match=r"`url` expected <class 'str'>. Got <class 'int'>",
        ):
            scrape_route_type_lookup(gtfs_url=1)
        with pytest.raises(
            TypeError,
            match=r"`url` expected <class 'str'>. Got <class 'bool'>",
        ):
            scrape_route_type_lookup(ext_spec_url=False)
        with pytest.raises(
            ValueError,
            match="url string expected protocol, instead found foobar",
        ):
            scrape_route_type_lookup(gtfs_url="foobar")
        with pytest.raises(
            TypeError,
            match=r"`extended_schema` .* <class 'bool'>. Got <class 'str'>",
        ):
            scrape_route_type_lookup(extended_schema="True")

    def test_table_without_extended_schema(self, mocker):
        """Check the return object when extended_schema = False."""
        patch_resp_txt = mocker.patch(
            "transport_performance.gtfs.routes._get_response_text",
            side_effect=mocked__get_response_text,
        )
        result = scrape_route_type_lookup(extended_schema=False)
        # did the mocker get used
        found = patch_resp_txt.call_args_list
        assert found == [
            call("https://gtfs.org/schedule/reference/")
        ], f"Expected mocker was called with specific url but found: {found}"
        assert isinstance(
            result, pd.core.frame.DataFrame
        ), f"Expected DF but found: {type(result)}"
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame({"route_type": "0", "desc": "Tram."}, index=[0]),
        )

    def test_table_with_extended_schema(self, mocker):
        """Check return table when extended schema = True."""
        patch_resp_txt = mocker.patch(
            "transport_performance.gtfs.routes._get_response_text",
            side_effect=mocked__get_response_text,
        )
        result = scrape_route_type_lookup()
        found = patch_resp_txt.call_args_list
        assert found == [
            call("https://gtfs.org/schedule/reference/"),
            call(
                (
                    "https://developers.google.com/transit/gtfs/reference/"
                    "extended-route-types"
                )
            ),
        ], f"Expected mocker to be called with specific urls. Found: {found}"

        assert isinstance(
            result, pd.core.frame.DataFrame
        ), f"Expected DF. Found: {type(result)}"
        pd.testing.assert_frame_equal(
            result,
            pd.DataFrame(
                {
                    "route_type": ["0", "100"],
                    "desc": ["Tram.", "Railway Service"],
                },
                index=[0, 1],
            ),
        )

    @pytest.mark.runinteg
    def test_lookup_is_stable(self):
        """Check if the tables at the urls have changed content."""
        # import the expected fixtures
        lookup_fix = pd.read_pickle(
            os.path.join(PKG_PATH, "data", "gtfs", "route_lookup.pkl")
        )
        lookup = scrape_route_type_lookup()
        pd.testing.assert_frame_equal(lookup, lookup_fix)


def _create_pkl(obj, out_pth: Union[str, pathlib.Path]) -> None:
    """Private function used to create .pkl. Not exported."""
    # NOTE: not including defences as this is a private function that is only
    # used for testing
    with open(out_pth, "wb") as f:
        pickle.dump(obj, f)

    return None


class Test_GetSavedRouteTypeLookup(object):
    """Tests for get_saved_route_type_lookup()."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            # test raises from key _is_expected_filetype() defences
            (
                pathlib.Path(
                    os.path.join(
                        "tests", "data", "gtfs", "newport-20230613_gtfs.zip"
                    )
                ),
                pytest.raises(
                    ValueError,
                    match=r"`path` expected file extension .pkl. "
                    r"Found .zip",
                ),
            ),
            (
                here("tests/data/test_file.pkl"),
                pytest.raises(
                    FileNotFoundError,
                    match=re.escape(
                        f"{here('tests/data/test_file.pkl')} not "
                        "found on file."
                    ),
                ),
            ),
        ],
    )
    def test_get_saved_route_type_lookup_raises(
        self, path: Union[str, pathlib.Path], expected: Type[RaisesContext]
    ):
        """Test raises."""
        with expected:
            get_saved_route_type_lookup(path=path)

    @pytest.mark.parametrize(
        "pkl_name, test_obj, expected",
        [
            # invalid object once .pkl unserialized
            (
                "list_pkl.pkl",
                [1, 2, 3, 4, 5],
                pytest.raises(
                    TypeError,
                    match=re.escape(
                        "Serialized object in specified .pkl file is of type: "
                        "<class 'list'>. Expected (<class 'dict'>, "
                        "<class 'pandas.core.frame.DataFrame'>)"
                    ),
                ),
            ),
            # empty dataframe
            (
                "empty_pkl.pkl",
                pd.DataFrame({"test_col": []}),
                pytest.warns(
                    UserWarning, match="Route type lookup has length of 0"
                ),
            ),
            # df with invalid columns
            (
                "invalid_col_pkl.pkl",
                pd.DataFrame(
                    {"route_type": [0, 1], "route_desc": ["car", "bus"]}
                ),
                pytest.warns(
                    UserWarning,
                    match=(
                        "Unexpected column 'route_desc' in route type lookup"
                    ),
                ),
            ),
        ],
    )
    def test_get_saved_route_type_lookup_invalid_pkl(
        self, tmp_path, pkl_name, test_obj, expected
    ):
        """Test defences when unserialized pickle is invalid obj type."""
        out_pth = os.path.join(tmp_path, pkl_name)
        _create_pkl(obj=test_obj, out_pth=out_pth)
        with expected:
            get_saved_route_type_lookup(out_pth)

    def test_get_saved_route_type_lookup_on_pass(self, tmp_path):
        """Test get_saved_route_type_lookup_on_pass()."""
        found_lkp = get_saved_route_type_lookup()
        assert isinstance(found_lkp, pd.DataFrame), ".pkl did not return df"
        assert len(found_lkp) == 91, ".pkl is an unexpected length"
        assert list(found_lkp.columns) == [
            "route_type",
            "desc",
        ], ".pkl columns are not as expected"
        # test a .pkl containing a serialized dict
        test_dict = {"route_type": [0, 1, 2], "desc": ["bus", "train", "car"]}
        out_pth = os.path.join(tmp_path, "dict_pkl.pkl")
        _create_pkl(test_dict, out_pth=out_pth)
        dict_lkp = get_saved_route_type_lookup(path=out_pth)
        assert isinstance(
            dict_lkp, pd.DataFrame
        ), "Dict pkl did not convert to pd.DataFrame"
        assert len(dict_lkp) == 3, "Dict .pkl length not as expected"
        assert list(dict_lkp.columns) == [
            "route_type",
            "desc",
        ], ".pkl columns are not as expected"
