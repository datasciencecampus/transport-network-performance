"""Utility functions to handle merging and resampling of raster data.

These were developed to support merging raster files together (e.g., to cover
a larger area), and resampling to a different grid size (e.g., 100x100m grids
to 200x200m grids). The original design intention is for these to form part of
gridded population data pre-processing.
"""


def merge_raster_files(
    input_dir: str, output_dir: str, output_filename: str
) -> None:
    """Merge raster files together.

    Takes an input directory and merges all `.tif` files within it. Writes
    merged raster to file.

    Parameters
    ----------
    input_dir : str
        Directory containing input raster files
    output_dir : str
        Directory to write output, merged raster file
    output_filename : str
        Filename of merged raster file (.tif extension required)

    """
    pass


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
