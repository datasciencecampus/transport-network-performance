"""Test validate_osm."""
import pytest
import re
from pyprojroot import here

from transport_performance.osm.validate_osm import (
    _compile_tags,
    _check_dict_values_all_equal,
    _filter_target_dict_with_list,
    FindIds,
)


class Test_CompileTags(object):
    """Tests for _compile_tags internal."""

    def test__compile_tags_returns_as_expected(self):
        """Test compile_tags returns correct dictionary."""

        class _mock_osm_tags:
            """Fixture classes are not currently supported.

            This class presents the tags in the same format as osmium.
            """

            def __init__(self) -> None:
                self.tags = [
                    ["1", "foo", "2", "bar", "3", "baz", "4", "foobar"],
                    ["a", "fizz", "b", "buzz", "c", "fizzbuzz"],
                ]

        osm = _mock_osm_tags()
        out = _compile_tags(osm)
        assert isinstance(out, dict)
        assert len(out) == 7
        assert out["1"] == "foo"
        assert out["c"] == "fizzbuzz"
        type(out.keys())
        assert list(out.keys()) == ["1", "2", "3", "4", "a", "b", "c"]
        assert list(out.values()) == [
            "foo",
            "bar",
            "baz",
            "foobar",
            "fizz",
            "buzz",
            "fizzbuzz",
        ]


class Test_CheckDictValuesAllEqual(object):
    """Tests for _check_dict_values_all_equal internal."""

    def test__check_dict_values_all_equal(self):
        """Assert outcomes for dictionary value checks."""
        equal_dict = {"a": "foo", "b": "foo"}
        unequal_dict = {"a": "foo", "b": "bar"}
        assert _check_dict_values_all_equal(equal_dict, "foo")
        assert not _check_dict_values_all_equal(unequal_dict, "bar")


class Test_FilterTargetDictWithList(object):
    """Tests for _filter_target_dict_with_list internal."""

    def test_filter_target_dict_with_list_raises(self):
        """Check on raise conditions."""
        with pytest.raises(
            TypeError,
            match="`targets` expected <class 'dict'>. Got <class 'int'>",
        ):
            _filter_target_dict_with_list(
                targets=1, _list=list(), search_key="key", accepted_keys=list()
            )
        with pytest.raises(
            TypeError,
            match="`search_key` expected <class 'str'>. Got <class 'int'>",
        ):
            _filter_target_dict_with_list(
                targets={}, _list=list(), search_key=1, accepted_keys=list()
            )
        with pytest.raises(
            TypeError,
            match="`_list` should be a list. Instead found <class 'int'>",
        ):
            _filter_target_dict_with_list(
                targets=dict(), _list=1, search_key="key", accepted_keys=list()
            )
        with pytest.raises(
            TypeError,
            match="`_list` must contain <class 'int'> only. Found"
            " <class 'float'> : 3.0",
        ):
            _filter_target_dict_with_list(
                targets=dict(),
                _list=[1, 2, 3.0],
                search_key="key",
                accepted_keys=list(),
            )
        with pytest.raises(
            TypeError,
            match="`accepted_keys` should be a list. Instead found"
            " <class 'int'>",
        ):
            _filter_target_dict_with_list(
                targets=dict(), _list=[1], search_key="key", accepted_keys=1
            )
        with pytest.raises(
            TypeError,
            match="`accepted_keys` must contain <class 'str'> only. Found"
            " <class 'int'> : 2",
        ):
            _filter_target_dict_with_list(
                targets=dict(),
                _list=[1],
                search_key="key",
                accepted_keys=["1", 2, "3"],
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'search_key' expected one of the following:['list', 'of',"
                " 'keys'] Got unacceptable_key"
            ),
        ):
            _filter_target_dict_with_list(
                targets=dict(),
                _list=list(),
                search_key="unacceptable_key",
                accepted_keys=["list", "of", "keys"],
            )
        # check catching conditions where search key doesn't match the name
        # of any dictionary target
        empty_dict = {"some_dict": dict()}
        with pytest.raises(
            KeyError,
            match=re.escape(
                "`search_key`: no_match did not match keys in `targets`:"
                " dict_keys(['some_dict'])"
            ),
        ):
            _filter_target_dict_with_list(
                targets=empty_dict,
                _list=[1, 2, 3],
                search_key="no_match",
                accepted_keys=["no_match"],
            )
        # this check is to simulate providing the incorrect search key
        # resulting in an empty dictionary returned.
        with pytest.raises(
            ValueError,
            match="No tags found. Did you specify the correct search_key?",
        ):
            _filter_target_dict_with_list(
                targets=empty_dict,
                _list=[1, 2, 3],
                search_key="some_dict",
                accepted_keys=["some_dict"],
            )

    def test__filter_target_dict_with_list_returns(self):
        """Assert that target dict is filtered."""
        target_dicts = {
            "choose_me": {1: "keep", 2: "keep", 3: "keep", 4: "remove"},
            "do_not_choose_me": {1: "should", 2: "not", 3: "appear"},
        }
        out = _filter_target_dict_with_list(
            targets=target_dicts,
            _list=[1, 2, 3],
            search_key="choose_me",
            accepted_keys=["choose_me", "do_not_choose_me"],
        )
        assert isinstance(out, dict)
        assert list(out.keys()) == ["choose_me"]
        assert "remove" not in out.values()


# osm path
OSM_PTH = here("tests/data/newport-2023-06-13.osm.pbf")
# classes are costly, so instantiate only once
ids = FindIds(OSM_PTH)


@pytest.fixture(scope="function")
def _IDsFixture():
    """Return an instance of FindIds."""
    return ids


class TestFindIds(object):
    """Tests for FindIds api class."""

    def test_findids_init(self, _IDsFixture):
        """Test init behaviour for FindIds.

        Parameters
        ----------
        IDsFixture : pytest.fixture
            Object of class FindIds.

        """
        # check all the expected attributes
        expected_attrs = [
            "counts",
            "id_dict",
            "node_ids",
            "way_ids",
            "relations_ids",
            "area_ids",
        ]
        for attr in expected_attrs:
            assert hasattr(
                ids, attr
            ), f"The expected attribute `{attr}` was not found in {ids}"
        expected_methods = [
            "count_features",
            "get_feature_ids",
            "node",
            "way",
            "relation",
            "area",
        ]
        found_methods = [
            getattr(ids, m, "not_found") for m in expected_methods
        ]
        # check all these methods were found
        for i, method in enumerate(found_methods):
            assert (
                method != "not_found"
            ), f"The expected method `{expected_methods[i]}`"
            f" was not found in {ids}"
        # assert they are methods
        for i, method in enumerate(found_methods):
            assert callable(
                method
            ), f"The expected method `{expected_methods[i]}`"
            f" in {ids} is not callable"
        # check feature ID counts in osm test fixture
        e_nod = 256508
        f_nod = len(ids.node_ids)
        assert f_nod == e_nod, f"Expected {e_nod} nodes, found {f_nod} nodes."
        e_way = 51231
        f_way = len(ids.way_ids)
        assert f_way == e_way, f"Expected {e_way} ways, found {f_way} ways."
        e_rel = 286
        f_rel = len(ids.relations_ids)
        assert f_rel == e_rel, f"Expected {e_rel} rels, found {f_rel} rels."
        e_are = 37841
        f_are = len(ids.area_ids)
        assert f_are == e_are, f"Expected {e_are} areas, found {f_are} areas."
