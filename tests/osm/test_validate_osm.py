"""Test validate_osm."""
import pytest
import re
from pyprojroot import here

from transport_performance.osm.validate_osm import (
    _compile_tags,
    _filter_target_dict_with_list,
    FindIds,
    FindLocations,
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


osm_pth = here("tests/data/newport-2023-06-13.osm.pbf")
ids = FindIds(osm_pth)


class TestFindIds(object):
    """Tests for FindIds api class."""

    # classes are costly, so instantiate only once

    e_nod = 256508
    e_way = 51231
    e_rel = 286
    e_area = 37841

    def test_findids_init(self):
        """Test init behaviour for FindIds."""
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
        f_nod = len(ids.node_ids)
        assert (
            f_nod == self.e_nod
        ), f"Expected {self.e_nod} nodes, found {f_nod} nodes."
        f_way = len(ids.way_ids)
        assert (
            f_way == self.e_way
        ), f"Expected {self.e_way} ways, found {f_way} ways."
        f_rel = len(ids.relations_ids)
        assert (
            f_rel == self.e_rel
        ), f"Expected {self.e_rel} rels, found {f_rel} rels."
        f_area = len(ids.area_ids)
        assert (
            f_area == self.e_area
        ), f"Expected {self.e_area} areas, found {f_area} areas."

    def test_find_ids_count_features(self):
        """Test count_features method."""
        ids.count_features()
        assert isinstance(ids.counts, dict)
        f_nod = ids.counts["n_nodes"]
        f_way = ids.counts["n_ways"]
        f_rel = ids.counts["n_relations"]
        f_area = ids.counts["n_areas"]
        assert (
            f_nod == self.e_nod
        ), f"Expected {self.e_nod} nodes, found {f_nod} nodes."
        assert (
            f_way == self.e_way
        ), f"Expected {self.e_way} ways, found {f_way} ways."
        assert (
            f_rel == self.e_rel
        ), f"Expected {self.e_rel} rels, found {f_rel} rels."
        assert (
            f_area == self.e_area
        ), f"Expected {self.e_area} areas, found {f_area} areas."

    def test_get_feature_ids(self):
        """get_feature_ids returns correct IDs."""
        ids.get_feature_ids()
        assert isinstance(ids.id_dict, dict)
        assert isinstance(ids.id_dict["node_ids"], list)
        assert sorted(ids.id_dict["node_ids"])[0:5] == [
            127231,
            127233,
            127234,
            127235,
            127236,
            127237,
        ], "First 5 node IDs not as expected"
        assert sorted(ids.id_dict["way_ids"][0:5]) == [
            1881332,
            1881588,
            2372923,
            2954415,
            2954417,
            2954418,
        ], "First 5 way IDs not as expected"
        assert sorted(ids.id_dict["relation_ids"][0:5]) == [
            20990,
            22696,
            58437,
            62149,
            122733,
            128098,
        ], "First 5 relation IDs not as expected"
        assert sorted(ids.id_dict["area_ids"][0:5]) == [
            8418164,
            8418170,
            9622110,
            9622112,
            9625854,
            10172568,
        ], "First 5 area IDs not as expected"


class TestFindLocations(object):
    """Tests for FindLocations api class."""

    locs = FindLocations(osm_pth)

    def test_find_locations_init(self):
        """Test for FindLocations init behaviour."""
        expected_attrs = [
            "found_locs",
            "node_locs",
            "way_node_locs",
        ]
        for attr in expected_attrs:
            assert hasattr(
                self.locs, attr
            ), f"The expected attribute `{attr}` was not found in {self.locs}"
        expected_methods = [
            "check_locs_for_ids",
            "node",
            "way",
        ]
        found_methods = [
            getattr(self.locs, m, "not_found") for m in expected_methods
        ]
        # check all these methods were found
        for i, method in enumerate(found_methods):
            assert (
                method != "not_found"
            ), f"The expected method `{expected_methods[i]}`"
            f" was not found in {self.locs}"
        # assert they are methods
        for i, method in enumerate(found_methods):
            assert callable(
                method
            ), f"The expected method `{expected_methods[i]}`"
            f" in {self.locs} is not callable"

        assert self.locs.node_locs[10971292664] == {
            "lon": "-3.0019690",
            "lat": "51.5804167",
        }
        # 29 node locations for way ID 1881332
        assert len(self.locs.way_node_locs[1881332]) == 29

    def test_check_locs_for_ids(self):
        """Assert check_locs_for_ids."""
        ids.get_feature_ids()
        # check that the expected coordinates are returned for list of node IDs
        id_list = sorted(ids.id_dict["node_ids"])[0:5]
        self.locs.check_locs_for_ids(ids=id_list, feature_type="node")
        # len(locs.found_locs["node"])
        assert list(self.locs.found_locs.keys()) == ["node"]
        assert len(self.locs.found_locs["node"]) == 5
        # in all 5 nodes, check that floats are returned
        for n in self.locs.found_locs["node"]:
            for k, v in self.locs.found_locs["node"][n].items():
                assert isinstance(
                    v, float
                ), f"Expected value{v} to be type float. got {type(v)}"
