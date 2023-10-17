"""Test validate_osm."""
from transport_performance.osm.validate_osm import (
    _compile_tags,
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
