"""Test osm_utils module."""
import pytest
from pyprojroot import here
import os
from unittest.mock import patch, call

from transport_performance.osm.osm_utils import filter_osm


class TestFilterOsm(object):
    """Testing filter_osm()."""

    def test_filter_osm_defense(self):
        """Defensive behaviour for filter_osm."""
        with pytest.raises(
            FileExistsError, match="not/a/pbf/.nosiree not found on file."
        ):
            # file doesnt exist
            filter_osm(pbf_pth="not/a/pbf/.nosiree")
        with pytest.raises(
            ValueError,
            match="`pbf_pth` expected file extension .pbf. Found .zip",
        ):
            # file exists but is not a pbf
            filter_osm(pbf_pth=here("tests/data/newport-20230613_gtfs.zip"))
        with pytest.raises(
            TypeError,
            match="`out_pth` expected path-like, found <class 'bool'>.",
        ):
            # out_pth is not a path_like
            filter_osm(out_pth=False)
        with pytest.raises(
            TypeError,
            match="`tag_filter` expected <class 'bool'>. Got <class 'int'>",
        ):
            # check for boolean defense
            filter_osm(tag_filter=1)
        with pytest.raises(
            TypeError,
            match="`install_osmosis` .* <class 'bool'>. Got <class 'str'>",
        ):
            # check for boolean defense
            filter_osm(install_osmosis="False")
        with pytest.raises(
            ValueError,
            match="box longitude West 1.1 is not smaller than East 1.0",
        ):
            # check for bounding boxes that osmosis won't like - long problem
            filter_osm(bbox=[1.1, 0.0, 1.0, 0.1])
        with pytest.raises(
            ValueError,
            match="box latitude South 0.1 is not smaller than North 0.0",
        ):
            # lat problem
            filter_osm(bbox=[0.0, 0.1, 0.1, 0.0])
        with pytest.raises(
            TypeError,
            match="ox` must contain <class 'float'> only. Found <class 'int'>",
        ):
            # type problems with bbox
            filter_osm(bbox=[0, 1.1, 0.1, 1.2])

    @patch("builtins.print")
    def test_filter_osm_defense_missing_osmosis(self, mock_print, mocker):
        """Assert func behaves when osmosis is missing and install=False."""
        with pytest.raises(
            Exception, match="`osmosis` is not found. Please install."
        ):
            # imitate missing osmosis and install_osmosis is False
            mock_missing_osmosis = mocker.patch(
                "transport_performance.osm.osm_utils.subprocess.run",
                side_effect=FileNotFoundError("No osmosis here..."),
            )
            filter_osm()
        assert (
            mock_missing_osmosis.called
        ), "`mock_missing_osmosis` was not called."
        # check which call was passed to mocker. Need to extract the string
        # as filepaths in the osmosis command will change on runners.
        subprocess_cmd = mock_missing_osmosis.call_args_list[0].__str__()
        assert subprocess_cmd.startswith(
            r"call(['osmosis', '--read-pbf',"
        ), f"Expected command got by mocker changed. Got: {subprocess_cmd} "
        # collect print statements
        func_out = mock_print.mock_calls
        assert func_out == [
            call("Rejecting ways:  waterway, landuse & natural.")
        ], f"Expected print statement not encountered. Got: {func_out}"

    @pytest.mark.runinteg
    @patch("builtins.print")
    def test_filter_osm_with_osmosis(self, mock_print, tmpdir):
        """Assertions when osmosis is present."""
        target_pth = os.path.join(tmpdir, "test_output.osm.pbf")
        out = filter_osm(out_pth=target_pth, install_osmosis=True)
        # assert mock_missing_osmosis.called
        assert os.path.exists(
            target_pth
        ), f"Filtered pbf file not found: {target_pth}"
        func_out = mock_print.mock_calls
        assert (
            func_out[-1]
            .__str__()
            .startswith("call('Filter completed. Written to ")
        )
        assert out is None
