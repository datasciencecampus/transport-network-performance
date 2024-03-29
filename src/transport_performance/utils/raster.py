"""Utility functions to handle merging and resampling of raster data.

These were developed to support merging raster files together (e.g., to cover
a larger area), and resampling to a different grid size (e.g., 100x100m grids
to 200x200m grids). The original design intention is for these to form part of
gridded population data pre-processing.
"""

import os
import glob
import re
import rioxarray
import pathlib

from typing import Union
from rioxarray.merge import merge_arrays
from rasterio.warp import Resampling
from transport_performance.utils.defence import (
    _handle_path_like,
    _check_parent_dir_exists,
    _is_expected_filetype,
    _type_defence,
    _enforce_file_extension,
)


def merge_raster_files(
    input_dir: Union[str, pathlib.Path],
    output_dir: Union[str, pathlib.Path],
    output_filename: str,
    subset_regex: str = None,
) -> dict:
    """Merge raster files together.

    Takes an input directory and merges all `.tif` files within it. Writes
    merged raster to file.

    Parameters
    ----------
    input_dir : Union[str, pathlib.Path]
        Directory containing input raster files.
    output_dir : Union[str, pathlib.Path]
        Directory to write output, merged raster file.
    output_filename : str
        Filename of merged raster file (.tif extension required).
    subset_regex : str, optional
        Subset any raster files in the input directory using a regex, by
        default None which means no subsetting will occur.

    Returns
    -------
    bounds : dict
        A dictionary summarising the boundaries of all input rasters and the
        merged output. The "inputs" key is a list of the respective input
        boundaries. The "output" key is a list containing the bounds of the
        merged result. Useful to checking consistency of merged output.

    Raises
    ------
    FileNotFoundError
        In scenarios where `input_dir` does not exist and if there are no .tif
        detectable within `input_dir`.

    Notes
    -----
    1. This function does not provide any consistency checking of inputs and
    merged outputs (e.g., checking for overlapping inputs and 'gaps' in the
    merged outputs). This is primarily because merging is nuanced and numerous
    in the potential ways inputs can be merged. For this reason, it is down to
    the function's user to ensure the merged output is consistent with the
    respective inputs. To this end, the bounds dict is returned to allow user
    consistency testing.

    2. The default rioxarry behaviours are assumed when merging inputs, i.e.,
    the `CRS`, resolution and `nodata` values will be taken from the first
    input DataArray. See [1]_ for more details.

    References
    ----------
    .. [1] https://corteva.github.io/rioxarray/html/rioxarray.html#rioxarray.m
    erge.merge_arrays

    """
    # defend against case where the provided input dir does not exist
    # handle path like first to capture input dir type errors before os.join
    input_dir = _handle_path_like(input_dir, "input_dir")
    # then add a dummy file to purely check if provided parent directory exists
    dummy_path = os.path.join(input_dir, "dummy.txt")
    _check_parent_dir_exists(dummy_path, "input_dir", create=False)

    # get tif files in directory, ensure some exist and select subset via regex
    tif_filepaths = glob.glob(f"{input_dir}/*.tif")
    if len(tif_filepaths) == 0:
        raise FileNotFoundError(f"No `*.tif` files found in {input_dir}")

    # apply regex and ensure tif files exist after applying it
    # raise a unique FileNotFoundError to aid debugging
    _type_defence(subset_regex, "subset_regex", (str, type(None)))
    if subset_regex is not None:
        tif_filepaths = [
            fpath for fpath in tif_filepaths if re.search(subset_regex, fpath)
        ]
    if len(tif_filepaths) == 0:
        raise FileNotFoundError(
            f"No `*.tif` files found in {input_dir} after applying regex "
            f"'{subset_regex}'."
        )

    # build a list of input rioxarrays to be merged
    arrays = []
    for tif_filepath in sorted(tif_filepaths):
        arrays.append(rioxarray.open_rasterio(tif_filepath, masked=True))

    # merge the datasets together
    xds_merged = merge_arrays(arrays)

    # create full filepath for merged tif file and write to disk
    # check expected file type and parent dir exists (creating if not)
    # need to _handle_path_like before os.path.join
    output_dir = _handle_path_like(output_dir, "output_dir")
    _type_defence(output_filename, "output_filename", str)
    merged_dir = os.path.join(output_dir, output_filename)
    _check_parent_dir_exists(merged_dir, "merged_dir", create=True)
    merged_dir = _enforce_file_extension(
        merged_dir, ".tif", ".tif", "merged_dir"
    )

    xds_merged.rio.to_raster(merged_dir)

    # get boundaries of inputs and output raster
    bounds = {
        "inputs": [array.rio.bounds() for array in arrays],
        "output": [xds_merged.rio.bounds()],
    }

    return bounds


def sum_resample_file(
    input_filepath: Union[str, pathlib.Path],
    output_filepath: Union[str, pathlib.Path],
    resample_factor: int = 2,
) -> None:
    """Resample raster file (change grid resolution) by summing.

    Takes an input raster file, resamples to a different grid resolution by
    summing constituent cells, and writes the resampled raster to file.

    Parameters
    ----------
    input_filepath : Union[str, pathlib.Path]
        Input filpath of GeoTIFF file
    output_filepath : Union[str, pathlib.Path]
        Output filepath for resampled GeoTIFF file
    resample_factor : int, optional
        Factor to resample input raster by, by default 2 which means the
        resolution will be decreased by 2 (e.g., input=100x100m then output=200
        x200m)

    Raises
    ------
    FileNotFoundError
        Unable to find `input_filepath`

    """
    # defend against case where the provided input dir does not exist
    _is_expected_filetype(input_filepath, "input_filepath", exp_ext=".tif")

    xds = rioxarray.open_rasterio(input_filepath, masked=True)

    # resample based on scaling factor and using sum resampling
    _type_defence(resample_factor, "resample_factor", int)
    xds_resampled = xds.rio.reproject(
        xds.rio.crs,
        resolution=tuple(
            res * resample_factor for res in xds.rio.resolution()
        ),
        resampling=Resampling.sum,
    )

    # make output_filepath's directory if it does not exist and check the
    # output file extension
    _check_parent_dir_exists(output_filepath, "output_filepath", create=True)
    _is_expected_filetype(
        output_filepath,
        "output_filepath",
        exp_ext=".tif",
        check_existing=False,
    )

    xds_resampled.rio.to_raster(output_filepath)
