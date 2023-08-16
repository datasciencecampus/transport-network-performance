"""Internal helpers for testing."""
import numpy as np
import rasterio as rio
import xarray as xr


def _np_to_rioxarray(
    arr: np.ndarray,
    aff: rio.Affine,
    as_type: str = "float32",
    no_data: int = -200,
    crs: str = "ESRI:54009",
) -> xr.DataArray:
    """Convert numpy array to rioxarry.

    This function is only used within pytest, as a convinent way to build
    fixtures without duplicating code.

    Parameters
    ----------
    arr : np.ndarray
        Input numpy array
    aff : rio.Affine
        Affine transform, for input data
    as_type : str, optional
        Data type, by default "int16"
    no_data : int, optional
        Value to use for no data, by default -200
    crs : _type_, optional
        Coordinate Reference system for input data, by default "ESRI:54009"

    Returns
    -------
    xr.DataArray
        Input numpy array as an xarray.DataArray with the correct

    """
    # get geometry of input arr and generate col, row indcies
    height = arr.shape[0]
    width = arr.shape[1]
    cols = np.arange(width)
    rows = np.arange(height)

    # transform indcies to x, y coordinates
    xs, ys = rio.transform.xy(aff, rows, cols)

    # build x_array object
    x_array = (
        xr.DataArray(arr.T, dims=["x", "y"], coords=dict(x=xs, y=ys))
        .transpose("y", "x")
        .astype(as_type)
        .rio.write_nodata(no_data)
        .rio.write_transform(aff)
        .rio.set_crs(crs)
    )

    return x_array
