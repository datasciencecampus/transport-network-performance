"""Utility functions to handle merging and resampling of raster data.

These were developed to support merging raster files together (e.g., to cover
a larger area), and resampling to a different grid size (e.g., 100x100m grids
to 200x200m grids). The original design intention is for these to form part of
gridded population data pre-processing.
"""

import os
import glob
import re

from pyprojroot import here


def merge_raster_files(
    input_dir: str,
    output_dir: str,
    output_filename: str,
    subset_regex: str = None,
) -> None:
    """Merge raster files together.

    Takes an input directory and merges all `.tif` files within it. Writes
    merged raster to file.

    Parameters
    ----------
    input_dir : str
        Directory containing input raster files.
    output_dir : str
        Directory to write output, merged raster file.
    output_filename : str
        Filename of merged raster file (.tif extension required).
    subset_regex : str, optional
        Subset any raster files in the input directory using a regex, by
        default None which means no subsetting will occur.

    """
    # defend against case where the provided input dir does not exist
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"{input_dir} can not be found")

    # get tif files in directory, ensure some exist and select subset via regex
    tif_filepaths = glob.glob(f"{input_dir}/*.tif")
    if len(tif_filepaths) == 0:
        raise FileNotFoundError(f"No `*.tif` files found in {input_dir}")
    if subset_regex is not None:
        tif_filepaths = [
            fpath for fpath in tif_filepaths if re.search(subset_regex, fpath)
        ]
    print(len(tif_filepaths))


def sum_resample_file(
    input_dir: str,
    input_filename: str,
    output_dir: str,
    output_filename: str,
    resample_factor: int = 2,
) -> None:
    """Resample raster file (change grid resolution) by summing.

    Takes an input raster file, resamples to a different grid resolution by
    summing constituent cells, and writes the resampled raster to file.

    Parameters
    ----------
    input_dir : str
        Input directory of GeoTIFF file
    input_filename : str
        Input filename of GeoTIFF file
    output_dir : str
        Output directory of resample GeoTIFF file
    output_filename : str
        Output filename of resample GeoTIFF file
    resample_factor : int, optional
        Factor to resample input raster by, by default 2 which means the
        resolution will be decreased by 2 (e.g., input=100x100m then output=200
        x200m)

    """
    pass


# TODO: remove once development is completed
if __name__ == "__main__":

    # set inputs, checking with E020 in filename
    INPUT_FOLDER = os.path.join(here(), "data", "external", "population")
    subset_regex = r"(.{0,})(E2020)(.{0,})(\.tif)$"
    merge_raster_files(INPUT_FOLDER, "", "", subset_regex=subset_regex)
