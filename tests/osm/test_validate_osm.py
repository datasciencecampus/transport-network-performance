"""Test validate_osm."""
import pytest
import re
from pyprojroot import here

from transport_performance.osm.validate_osm import (
    _compile_tags,
    _filter_target_dict_with_list,
    FindIds,
    FindLocations,
    FindTags,
)

osm_pth = here("tests/data/small-osm.pbf")


@pytest.fixture(scope="module")
def _tiny_osm_ids():
    """Findids too costly on standard fixture, so use _tiny_osm instead."""
    ids = FindIds(osm_pth)
    # many of the test classes here will depend on the get_feature_ids method
    # calculate now once rather than several times in test classes
    ids.get_feature_ids()
    return ids


@pytest.fixture(scope="module")
def _tiny_osm_locs():
    """Locations found within the _tiny_osm fixture."""
    locs = FindLocations(osm_pth)
    return locs


@pytest.fixture(scope="module")
def _tiny_osm_tags():
    """Tags found within the _tiny_osm fixture."""
    tags = FindTags(osm_pth)
    return tags


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
                targets={},
                _list=list(),
                search_key=1,
                accepted_keys=list("fookey"),
            )
        with pytest.raises(
            TypeError,
            match="`_list` expected <class 'collections.abc.Iterable'>. Got"
            " <class 'int'>",
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
            match="`accepted_keys` expected <class 'collections.abc.Iterable'>"
            ". Got <class 'int'>",
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
            match="'search_key'.*['list', 'of', 'keys'].*unacceptable_key:"
            " <class 'str'>",
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


def _class_atttribute_assertions(
    some_object, some_attributes: list, some_methods: list
) -> None:
    """Util for checking class internals.

    Asserts that the object contains the specified attributes & methods. Raises
    AssertionError if specified attributes & methods not found within object.
    """
    for attr in some_attributes:
        assert hasattr(
            some_object, attr
        ), f"The expected attribute `{attr}` was not found in {some_object}"
        found_methods = [
            getattr(some_object, m, "not_found") for m in some_methods
        ]
        # check all these methods were found
        for i, method in enumerate(found_methods):
            assert (
                method != "not_found"
            ), f"The expected method `{some_methods[i]}`"
            f" was not found in {some_object}"
        # assert they are methods
        for i, method in enumerate(found_methods):
            assert callable(method), f"The expected method `{some_methods[i]}`"
            f" in {some_object} is not callable"


class TestFindIds(object):
    """Tests for FindIds api class."""

    # the expected number of features within the _tiny_osm_ids
    e_nod = 1669
    e_way = 227
    e_rel = 12
    e_area = 11

    def test_findids_init(self, _tiny_osm_ids):
        """Test init behaviour for FindIds."""
        ids = _tiny_osm_ids
        # check all the expected attributes
        expected_attrs = [
            "counts",
            "id_dict",
            "node_ids",
            "way_ids",
            "relations_ids",
            "area_ids",
        ]
        expected_methods = [
            "count_features",
            "get_feature_ids",
            "node",
            "way",
            "relation",
            "area",
        ]
        _class_atttribute_assertions(ids, expected_attrs, expected_methods)
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

    def test_find_ids_count_features(self, _tiny_osm_ids):
        """Test count_features method."""
        ids = _tiny_osm_ids
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

    def test_get_feature_ids(self, _tiny_osm_ids):
        """get_feature_ids returns correct IDs."""
        ids = _tiny_osm_ids
        assert isinstance(ids.id_dict, dict)
        assert isinstance(ids.id_dict["node_ids"], list)
        assert sorted(ids.id_dict["node_ids"])[0:3] == [
            7727955,
            7727957,
            7727958,
        ], "First 3 node IDs not as expected"
        assert sorted(ids.id_dict["way_ids"][0:4]) == [
            4811009,
            4812745,
            4812746,
            4812791,
        ], "First 4 way IDs not as expected"
        assert sorted(ids.id_dict["relation_ids"][0:5]) == [
            305815,
            368267,
            368270,
            8200260,
            8208502,
        ], "First 5 relation IDs not as expected"
        assert sorted(ids.id_dict["area_ids"][0:6]) == [
            9625854,
            10172568,
            10172570,
            51392038,
            53588532,
            77922346,
        ], "First 5 area IDs not as expected"


class TestFindLocations(object):
    """Tests for FindLocations api class."""

    def test_find_locations_init(self, _tiny_osm_locs):
        """Test for FindLocations init behaviour."""
        locs = _tiny_osm_locs
        exp_attrs = ["found_locs", "node_locs", "way_node_locs"]
        exp_methods = ["check_locs_for_ids", "node", "way"]
        _class_atttribute_assertions(locs, exp_attrs, exp_methods)
        assert locs.node_locs[7727955] == {
            "lon": -3.0034452,
            "lat": 51.5677329,
        }
        # 2 node locations for way ID 4811009
        assert len(locs.way_node_locs[4811009]) == 2

    def test_check_locs_for_ids(self, _tiny_osm_locs, _tiny_osm_ids):
        """Assert check_locs_for_ids."""
        ids = _tiny_osm_ids
        locs = _tiny_osm_locs
        # check that the expected coordinates are returned for node IDs
        id_list = sorted(ids.id_dict["node_ids"])[0:5]
        locs.check_locs_for_ids(ids=id_list, feature_type="node")
        assert list(locs.found_locs.keys()) == ["node"]
        assert len(locs.found_locs["node"]) == 5
        # in all 5 nodes, check that floats are returned
        for n in locs.found_locs["node"]:
            for k, v in locs.found_locs["node"][n].items():
                assert isinstance(
                    v, float
                ), f"Expected coord {v} to be type float. got {type(v)}"
        # now check coordinates for a list of way IDs
        way_ids = sorted(ids.id_dict["way_ids"])[0:3]
        locs.check_locs_for_ids(ids=way_ids, feature_type="way")
        assert list(locs.found_locs.keys()) == ["way"]
        assert len(locs.found_locs["way"]) == 3
        # coords are nested deeper for ways than nodes as you need to access
        # way members' coordinates
        for w in locs.found_locs["way"]:
            for x in locs.found_locs["way"][w]:
                for k, v in x.items():
                    for coord in list(v.values()):
                        assert isinstance(
                            coord, float
                        ), f"Expected coord {coord} to be type float."
                        " got {type(coord)}"


class TestFindTags(object):
    """Test FindTags API class."""

    def test_find_tags_init(self, _tiny_osm_tags):
        """Check init behaviour for FindTags."""
        tags = _tiny_osm_tags
        expected_attrs = [
            "found_tags",
            "node_tags",
            "way_tags",
            "relation_tags",
            "area_tags",
        ]
        expected_methods = [
            "check_tags_for_ids",
            "node",
            "way",
            "relation",
            "area",
        ]
        _class_atttribute_assertions(tags, expected_attrs, expected_methods)

    def test_find_tags_check_tags_for_ids(self, _tiny_osm_tags, _tiny_osm_ids):
        """Test FindTags.check_tags_for_ids()."""
        ids = _tiny_osm_ids
        tags = _tiny_osm_tags
        node_ids = ids.id_dict["node_ids"][20:50]
        way_ids = ids.id_dict["way_ids"][0:4]
        rel_ids = ids.id_dict["relation_ids"][0:3]
        area_ids = ids.id_dict["area_ids"][0:2]
        # many node IDs are empty, so check a known ID for tags instead
        tags.check_tags_for_ids(ids=node_ids, feature_type="node")
        target_node = tags.found_tags["node"][7728862]
        tag_value_map = {
            "highway": "traffic_signals",
        }
        for k, v in tag_value_map.items():
            f = target_node[k]
            assert f == v, f"Expected node tag value {v} but found {f}"

        # check way tags
        tags.check_tags_for_ids(ids=way_ids, feature_type="way")
        # raise ValueError(way_ids)
        target_way = tags.found_tags["way"][4811009]
        # raise ValueError(target_way)
        assert len(tags.found_tags["way"]) == 4
        tag_value_map = {
            "highway": "primary",
            "lanes": "2",
            "name": "Kingsway",
            "oneway": "yes",
            "postal_code": "NP20",
            "ref": "A4042",
        }
        for k, v in tag_value_map.items():
            f = target_way[k]
            assert f == v, f"Expected way tag value {v} but found {f}"
        # check relation tags
        tags.check_tags_for_ids(ids=rel_ids, feature_type="relation")
        target_rel = tags.found_tags["relation"][rel_ids[0]]
        # raise ValueError(target_rel)
        tag_value_map = {
            "colour": "#5d2491",
            "name": "Cardiff-Newport 30",
            "operator": "Cardiff Bus;Newport Bus",
            "ref": "30",
            "route": "bus",
            "type": "route",
        }
        for k, v in tag_value_map.items():
            f = target_rel[k]
            assert f == v, f"Expected relation tag value {v} but found {f}"

        # check area tags
        tags.check_tags_for_ids(ids=area_ids, feature_type="area")
        target_area = tags.found_tags["area"][area_ids[0]]
        # raise ValueError(target_area)
        tag_value_map = {
            "highway": "primary",
            "junction": "roundabout",
            "maxspeed": "40 mph",
            "name": "Cardiff Road",
            "oneway": "yes",
            "postal_code": "NP10",
        }
        for k, v in tag_value_map.items():
            f = target_area[k]
            assert f == v, f"Expected area tag value {v} but found {f}"
        assert len(tags.found_tags["area"]) == 2
