"""Test validate_osm."""
from transport_performance.osm.validate_osm import (
    _compile_tags,
    _check_dict_values_all_equal,
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
